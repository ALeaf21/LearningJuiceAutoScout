import csv
import math
import os

from autoscout.geometry import _field_center_to_corner_xy
from util.juice_log import read_rows as read_jlog_rows
from util.juice_log import sniff_jlog


class FieldDetector:
    def __init__(self, cv2, np):
        self.cv2, self.np = cv2, np

    def detect_field(self, frame):
        cv2, np = self.cv2, self.np
        h, w = frame.shape[:2]
        edges = cv2.Canny(
            cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5,5), 0),
            30, 100)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        for cnt in sorted(cnts, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(cnt) < h * w * 0.05:
                continue
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) == 4:
                pts  = approx.reshape(4, 2).astype(np.float32)
                s    = pts.sum(axis=1)
                diff = np.diff(pts, axis=1).flatten()
                ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                                    pts[np.argmax(s)], pts[np.argmax(diff)]],
                                   dtype=np.float32)
                H, _ = cv2.findHomography(
                    ordered,
                    np.array([[0,0],[144,0],[144,144],[0,144]], np.float32))
                return ordered, H
        return None


def _project_field_points(cv2, np, H_inv, points_in):
    pts = np.array([points_in], dtype=np.float32)
    img = cv2.perspectiveTransform(pts, H_inv)[0]
    return [(float(p[0]), float(p[1])) for p in img]


def _project_image_points(cv2, np, H_2d, points_px):
    pts = np.array([points_px], dtype=np.float32)
    field = cv2.perspectiveTransform(pts, H_2d)[0]
    return [(float(p[0]), float(p[1])) for p in field]


def _load_manual_pose_rows(path: str):
    if sniff_jlog(path):
        return read_jlog_rows(path)
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _build_manual_reid_histograms(
    tracker,
    cap,
    manual_reference_csv: str,
    H_inv,
    video_fps: float,
):
    cv2, np = tracker.cv2, tracker.np
    rows = _load_manual_pose_rows(manual_reference_csv)
    if not rows:
        print("[WARN] Manual re-ID reference file is empty: {}".format(manual_reference_csv))
        return None

    refs = {i: [] for i in range(4)}
    max_samples = tracker.REID_MAX_SAMPLES_PER_ROBOT
    stride = max(1, tracker.REID_SAMPLE_STRIDE)

    for row_idx in range(0, len(rows), stride):
        row = rows[row_idx]
        frame_num = int(round(float(row["timestamp_s"]) * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        for robot_id in range(4):
            if row["robot{}_visible".format(robot_id)] != "1":
                continue
            if len(refs[robot_id]) >= max_samples:
                continue

            fx_center = float(row["robot{}_x_in".format(robot_id)])
            fy_center = float(row["robot{}_y_in".format(robot_id)])
            fx, fy = _field_center_to_corner_xy(fx_center, fy_center)
            pt = np.array([[[fx, fy]]], dtype=np.float32)
            ip = cv2.perspectiveTransform(pt, H_inv)[0][0]
            hist = tracker._extract_appearance_feature(
                frame, int(round(float(ip[0]))), int(round(float(ip[1]))))
            if hist is not None:
                refs[robot_id].append(hist)

        if all(len(refs[i]) >= max_samples for i in range(4)):
            break

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    counts = [len(refs[i]) for i in range(4)]
    if not any(counts):
        print("[WARN] Could not build manual re-ID appearance references.")
        return None

    print("[INFO] Manual re-ID samples per robot: {}".format(counts))
    return refs


def _detect_robot_edge_profile(cv2, np, frame, fg_mask, tracker, center_px):
    """
    Find the robot's floor contact point from the local foreground silhouette.

    Tracker poses are blob centers, not ground-contact centers. For the cube
    base we choose the connected foreground component nearest the tracked point,
    then use Canny edges on its lower silhouette to estimate where the robot
    touches the field.
    """
    h, w = frame.shape[:2]
    cx, cy = center_px
    pad = max(22, int(tracker._robot_max_px * 1.05))
    x0 = max(0, int(cx - pad)); x1 = min(w, int(cx + pad))
    y0 = max(0, int(cy - pad * 1.5)); y1 = min(h, int(cy + pad * 1.2))
    fallback_h = tracker._robot_max_px * 0.85
    fallback_box = (
        cx - tracker._robot_max_px * 0.35,
        cy - fallback_h,
        cx + tracker._robot_max_px * 0.35,
        cy,
    )
    if x1 <= x0 or y1 <= y0:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    roi = frame[y0:y1, x0:x1]
    fg_roi = fg_mask[y0:y1, x0:x1]
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        fg_roi, 8)
    if n_labels <= 1:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    local_cx = float(cx - x0)
    local_cy = float(cy - y0)
    center_label = labels[int(np.clip(round(local_cy), 0, labels.shape[0]-1)),
                          int(np.clip(round(local_cx), 0, labels.shape[1]-1))]
    best_lbl = None
    best_score = -1.0
    for lbl in range(1, n_labels):
        area = float(stats[lbl, cv2.CC_STAT_AREA])
        if area < max(30.0, tracker.BLOB_MIN * 0.12):
            continue
        ccx, ccy = centroids[lbl]
        dist = math.hypot(float(ccx) - local_cx, float(ccy) - local_cy)
        score = area / (1.0 + dist / max(tracker._robot_max_px, 1.0))
        if center_label == lbl:
            score *= 2.0
        if score > best_score:
            best_score = score
            best_lbl = lbl

    if best_lbl is None:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 45, 135)
    comp = (labels == best_lbl).astype(np.uint8) * 255
    edges = cv2.bitwise_and(edges, comp)

    comp_ys, comp_xs = np.where(comp > 0)
    if len(comp_xs) < 8:
        return {
            "base_px": center_px,
            "top_y": cy - fallback_h,
            "bbox": fallback_box,
        }

    ys, xs = np.where(edges > 0)
    if len(xs) >= 8:
        edge_rx = max(8.0, tracker._robot_max_px * 0.62)
        edge_ry = max(10.0, tracker._robot_max_px * 1.05)
        edge_keep = (
            ((xs.astype(np.float32) - local_cx) / edge_rx) ** 2 +
            ((ys.astype(np.float32) - local_cy) / edge_ry) ** 2
        ) <= 1.0
        if int(edge_keep.sum()) >= 8:
            xs = xs[edge_keep]
            ys = ys[edge_keep]

    comp_rx = max(8.0, tracker._robot_max_px * 0.65)
    comp_ry = max(10.0, tracker._robot_max_px * 1.10)
    comp_keep = (
        ((comp_xs.astype(np.float32) - local_cx) / comp_rx) ** 2 +
        ((comp_ys.astype(np.float32) - local_cy) / comp_ry) ** 2
    ) <= 1.0
    if int(comp_keep.sum()) >= 8:
        comp_xs = comp_xs[comp_keep]
        comp_ys = comp_ys[comp_keep]

    comp_h = max(1.0, float(comp_ys.max() - comp_ys.min() + 1))
    lower_y = comp_ys.min() + comp_h * 0.58

    lower_edges = ys >= lower_y
    if int(lower_edges.sum()) >= 5:
        edge_ys = ys[lower_edges]
        edge_xs = xs[lower_edges]
        base_y_local = float(np.percentile(edge_ys, 92))
        band = edge_ys >= base_y_local - max(3.0, comp_h * 0.16)
        base_x_local = float(np.median(edge_xs[band] if int(band.sum()) else edge_xs))
    else:
        base_y_local = float(np.percentile(comp_ys, 96))
        band = comp_ys >= base_y_local - max(3.0, comp_h * 0.14)
        base_x_local = float(np.median(comp_xs[band] if int(band.sum()) else comp_xs))

    comp_box = (
        x0 + float(np.percentile(comp_xs, 4)),
        y0 + float(np.percentile(comp_ys, 4)),
        x0 + float(np.percentile(comp_xs, 96)),
        y0 + float(np.percentile(comp_ys, 96)),
    )

    if len(xs) >= 8:
        canny_box = (
            x0 + float(np.percentile(xs, 3)),
            y0 + float(np.percentile(ys, 3)),
            x0 + float(np.percentile(xs, 97)),
            y0 + float(np.percentile(ys, 97)),
        )
        edge_box = (
            min(canny_box[0], comp_box[0]),
            min(canny_box[1], comp_box[1]),
            max(canny_box[2], comp_box[2]),
            max(canny_box[3], comp_box[3]),
        )
        top_y_local = float(np.percentile(ys, 8))
    else:
        edge_box = comp_box
        top_y_local = float(np.percentile(comp_ys, 5))

    return {
        "base_px": (x0 + base_x_local, y0 + base_y_local),
        "top_y": y0 + top_y_local,
        "bbox": edge_box,
    }


def _projected_bottom_for_side(cv2, np, H_inv, field_center, side_in):
    half = side_in / 2.0
    bottom_field = [
        (field_center[0] - half, field_center[1] - half),
        (field_center[0] + half, field_center[1] - half),
        (field_center[0] + half, field_center[1] + half),
        (field_center[0] - half, field_center[1] + half),
    ]
    return _project_field_points(cv2, np, H_inv, bottom_field)


def _choose_cube_side_in(cv2, np, H_inv, field_center, edge_bbox, tracker):
    x_min, _y_min, x_max, _y_max = edge_bbox
    target_min = x_min - 2.0
    target_max = x_max + 2.0
    lo = max(8.0, tracker.ROBOT_SIZE_IN * 0.45)
    hi = tracker.ROBOT_SIZE_IN * 1.20
    best = hi
    for _ in range(10):
        mid = (lo + hi) * 0.5
        bottom = _projected_bottom_for_side(cv2, np, H_inv, field_center, mid)
        bx_min = min(p[0] for p in bottom)
        bx_max = max(p[0] for p in bottom)
        if bx_min <= target_min and bx_max >= target_max:
            best = mid
            hi = mid
        else:
            lo = mid
    return best


def _draw_wire_cube(cv2, np, dbg, frame, fg_mask, tracker, H_inv,
                    pose, center_px, color):
    profile = _detect_robot_edge_profile(
        cv2, np, frame, fg_mask, tracker, center_px)
    base_px = profile["base_px"]
    top_y = profile["top_y"]
    edge_bbox = profile["bbox"]
    if tracker._H_2d is not None:
        field_center = _project_image_points(cv2, np, tracker._H_2d, [base_px])[0]
    else:
        field_center = (pose.x_in, pose.y_in)

    side_in = _choose_cube_side_in(cv2, np, H_inv, field_center, edge_bbox, tracker)
    bottom = _projected_bottom_for_side(cv2, np, H_inv, field_center, side_in)
    h, w = dbg.shape[:2]
    if not all(-w <= x <= 2*w and -h <= y <= 2*h for x, y in bottom):
        return

    height_px = base_px[1] - top_y
    height_px = float(np.clip(height_px,
                              tracker._robot_max_px * 0.45,
                              tracker._robot_max_px * 1.45))
    top = [(x, y - height_px) for x, y in bottom]

    bottom_i = [(int(round(x)), int(round(y))) for x, y in bottom]
    top_i = [(int(round(x)), int(round(y))) for x, y in top]

    for pts, thickness in ((bottom_i, 1), (top_i, 2)):
        for j in range(4):
            cv2.line(dbg, pts[j], pts[(j+1) % 4], color, thickness, cv2.LINE_AA)
    for j in range(4):
        cv2.line(dbg, bottom_i[j], top_i[j], color, 2, cv2.LINE_AA)


def _open_debug_video_writer(cv2, base_output_dir, frame_size, fps):
    candidates = [
        ("tracker_debug.mp4", "mp4v"),
        ("tracker_debug.mp4", "avc1"),
        ("tracker_debug.avi", "MJPG"),
        ("tracker_debug.avi", "XVID"),
    ]
    width, height = frame_size
    attempted = []
    for filename, codec in candidates:
        path = os.path.join(base_output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, path, codec
        attempted.append("{} ({})".format(path, codec))
        writer.release()
    return None, attempted, None
