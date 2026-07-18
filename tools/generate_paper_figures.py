#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autoscout.tracker import RobotTracker


PANEL_W = 320
PANEL_H = 180
TITLE_H = 28


def load_ordered_corners(corners_path: Path) -> np.ndarray:
    data = json.loads(corners_path.read_text())
    bl, br, tr, tl = data["corners_px"]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def render_panel(title: str, image: np.ndarray) -> np.ndarray:
    image = ensure_bgr(image)
    image = cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_AREA)
    panel = np.full((PANEL_H + TITLE_H, PANEL_W, 3), 250, dtype=np.uint8)
    panel[TITLE_H:, :, :] = image
    cv2.rectangle(panel, (0, 0), (PANEL_W - 1, TITLE_H - 1), (32, 32, 32), -1)
    cv2.putText(
        panel,
        title,
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return panel


def stack_panels(panels, cols: int) -> np.ndarray:
    rows = []
    for idx in range(0, len(panels), cols):
        row = panels[idx:idx + cols]
        if len(row) < cols:
            filler = np.full_like(row[0], 255)
            while len(row) < cols:
                row.append(filler.copy())
        rows.append(cv2.hconcat(row))
    return cv2.vconcat(rows)


def thresholded_foreground(frame: np.ndarray, tracker: RobotTracker) -> np.ndarray:
    diff = cv2.absdiff(frame, tracker._bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, tracker.FG_THRESH, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(fg, tracker._field_mask)


def difference_magnitude(frame: np.ndarray, tracker: RobotTracker) -> np.ndarray:
    diff = cv2.absdiff(frame, tracker._bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def contour_overlay(frame: np.ndarray, tracker: RobotTracker) -> np.ndarray:
    overlay = frame.copy()
    contours, split_centers = tracker._foreground_contours(frame)
    for contour in contours:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)
    for sx, sy in split_centers:
        cv2.drawMarker(
            overlay,
            (int(sx), int(sy)),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )
    return overlay


def field_mask_overlay(frame: np.ndarray, ordered: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    cv2.polylines(
        overlay,
        [ordered.astype(np.int32).reshape(-1, 1, 2)],
        True,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def tracking_overlay(frame: np.ndarray, tracker: RobotTracker, ordered: np.ndarray, poses) -> np.ndarray:
    debug_colors = [(220, 160, 0), (0, 220, 255), (0, 0, 220), (0, 120, 255)]
    overlay = frame.copy()
    cv2.polylines(
        overlay,
        [ordered.astype(np.int32).reshape(-1, 1, 2)],
        True,
        (0, 220, 0),
        2,
        cv2.LINE_AA,
    )

    contours, split_centers = tracker._foreground_contours(frame)
    for contour in contours:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 255), 2)
    for sx, sy in split_centers:
        cv2.drawMarker(
            overlay,
            (int(sx), int(sy)),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )

    if tracker._last_ball_mask is not None:
        ball_cnts, _ = cv2.findContours(
            tracker._last_ball_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        for contour in ball_cnts:
            if cv2.contourArea(contour) >= 20:
                cv2.drawContours(overlay, [contour], -1, (255, 0, 255), 1)

    for merge_group in tracker._merge_groups.values():
        pos_list = [
            tracker._pos_px[tid]
            for tid in merge_group.track_ids
            if tracker._pos_px[tid] is not None
        ]
        if not pos_list:
            continue
        mcx = int(sum(point[0] for point in pos_list) / len(pos_list))
        mcy = int(sum(point[1] for point in pos_list) / len(pos_list))
        ax, ay = merge_group.entry_axis
        ex = int(mcx + ax * 50)
        ey = int(mcy + ay * 50)
        if len(merge_group.track_ids) == 2:
            color = (0, 80, 255) if merge_group.crossed else (0, 220, 100)
            label = "+".join(f"R{tid}" for tid in merge_group.track_ids)
            label += " crossed" if merge_group.crossed else " ok"
        else:
            color = (255, 160, 0)
            entry_s = "".join(str(tid) for tid in merge_group.entry_order)
            current_s = "".join(str(tid) for tid in merge_group.current_order)
            label = "+".join(f"R{tid}" for tid in merge_group.track_ids)
            label += f" [{entry_s}->{current_s}]"
        cv2.arrowedLine(overlay, (mcx, mcy), (ex, ey), color, 2)
        cv2.putText(
            overlay,
            label,
            (mcx + 6, mcy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    for rid, pose in enumerate(poses):
        if not pose.visible:
            continue
        if tracker._pos_px[rid] is not None:
            ix = int(round(tracker._pos_px[rid][0]))
            iy = int(round(tracker._pos_px[rid][1]))
        else:
            point = np.array([[[pose.x_in, pose.y_in]]], dtype=np.float32)
            img = cv2.perspectiveTransform(point, tracker._H_inv)[0][0]
            ix = int(round(float(img[0])))
            iy = int(round(float(img[1])))
        color = debug_colors[rid]
        cv2.circle(overlay, (ix, iy), 16, color, -1)
        cv2.circle(overlay, (ix, iy), 19, (255, 255, 255), 2)
        cv2.putText(
            overlay,
            f"R{rid}",
            (ix - 10, iy - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return overlay


def find_representative_frames(video_path: Path, tracker: RobotTracker, ordered: np.ndarray):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base_frame = None
    base_frame_num = None
    merge_overlay_img = None
    merge_frame_num = None
    best_merge_score = None

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        poses = tracker.update(frame)
        visible = sum(1 for pose in poses if pose.visible)
        merge_active = bool(tracker._merge_groups)

        if (
            base_frame is None
            and tracker._initialized
            and frame_num >= min(300, max(total_frames // 10, 120))
            and visible == 4
            and not merge_active
        ):
            base_frame = frame.copy()
            base_frame_num = frame_num

        if (
            merge_active
            and base_frame_num is not None
            and frame_num > base_frame_num + 60
            and frame_num < min(total_frames, base_frame_num + 1200)
        ):
            frame_center = (frame.shape[1] / 2.0, frame.shape[0] / 2.0)
            score = float("-inf")
            for merge_group in tracker._merge_groups.values():
                pos_list = [
                    tracker._pos_px[tid]
                    for tid in merge_group.track_ids
                    if tracker._pos_px[tid] is not None
                ]
                if not pos_list:
                    continue
                mcx = sum(point[0] for point in pos_list) / len(pos_list)
                mcy = sum(point[1] for point in pos_list) / len(pos_list)
                dist_center = np.hypot(mcx - frame_center[0], mcy - frame_center[1])
                cardinality = len(merge_group.track_ids)
                current = cardinality * 120.0 - 0.25 * dist_center
                if cardinality >= 3:
                    current += 80.0
                score = max(score, current)
            if score != float("-inf") and (best_merge_score is None or score > best_merge_score):
                best_merge_score = score
                merge_overlay_img = tracking_overlay(frame, tracker, ordered, poses)
                merge_frame_num = frame_num

        frame_num += 1

    cap.release()

    if base_frame is None:
        raise RuntimeError("Could not find a representative non-merge frame.")
    if merge_overlay_img is None:
        raise RuntimeError("Could not find a merge frame in the example video.")

    return base_frame, base_frame_num, merge_overlay_img, merge_frame_num


def build_cv_layers_figure(frame: np.ndarray, tracker: RobotTracker, ordered: np.ndarray) -> np.ndarray:
    raw_diff = difference_magnitude(frame, tracker)
    field_fg = thresholded_foreground(frame, tracker)
    ball_mask = tracker._ball_color_mask(frame)
    if ball_mask is None:
        ball_mask = np.zeros_like(field_fg)
    final_mask = tracker._foreground_mask(frame)

    panels = [
        render_panel("Input frame", frame),
        render_panel("Median background", tracker._bg),
        render_panel("Calibrated field overlay", field_mask_overlay(frame, ordered)),
        render_panel("Background difference", raw_diff),
        render_panel("Thresholded field mask", field_fg),
        render_panel("Ball color mask", ball_mask),
        render_panel("Final robot mask", final_mask),
        render_panel("Contours and split centers", contour_overlay(frame, tracker)),
    ]
    return stack_panels(panels, cols=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from an example match video.")
    parser.add_argument("--video", default="examples/joosmatch.mp4")
    parser.add_argument("--corners", default="field_corners.json")
    parser.add_argument("--output-dir", default="assets/paper")
    args = parser.parse_args()

    video_path = Path(args.video)
    corners_path = Path(args.corners)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ordered = load_ordered_corners(corners_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    ret, sample = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Could not read the first frame from the example video.")

    tracker = RobotTracker(cv2, np)
    tracker.setup(str(video_path), ordered, sample.shape)

    base_frame, base_frame_num, merge_overlay_img, merge_frame_num = find_representative_frames(
        video_path, tracker, ordered
    )

    cv_layers = build_cv_layers_figure(base_frame, tracker, ordered)

    cv_layers_path = output_dir / "cv_layers_joosmatch.png"
    merge_path = output_dir / "merge_debug_joosmatch.png"
    meta_path = output_dir / "paper_figure_metadata.json"

    cv2.imwrite(str(cv_layers_path), cv_layers)
    cv2.imwrite(str(merge_path), merge_overlay_img)

    meta = {
        "video": str(video_path),
        "corners": str(corners_path),
        "cv_layers_frame": int(base_frame_num),
        "merge_debug_frame": int(merge_frame_num),
        "cv_layers_image": str(cv_layers_path),
        "merge_debug_image": str(merge_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
