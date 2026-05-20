import argparse
import csv
import json
import math
import os
import sys

from autoscout.geometry import _field_center_to_corner_xy
from autoscout.helpers import (
    FieldDetector,
    _build_manual_reid_histograms,
    _draw_wire_cube,
    _open_debug_video_writer,
)
from autoscout.runtime import _make_bar, _require, install_console_output
from autoscout.shot import ShotDetector
from autoscout.tracker import RobotTracker
from autoscout.wpilog import WPILogWriter
from util.juice_log import (
    CSV_COLUMNS,
    ROBOT_POSE_SCHEMA,
    JuiceLogWriter,
    csv_row_to_list,
)


SAVE_ALL_DEBUG_AROUND_MERGES = False
PROCESS_EVERY_SOURCE_FRAME = True


install_console_output()


def process_match(
    video_path,
    output_dir,
    start_offset_sec=0.0,
    sample_rate_fps=10.0,
    debug=False,
    debug_video=False,
    debug_every_n=1,
    debug_enable_hitboxes=False,
    manual_corners_px=None,
    robot_init_positions=None,
    manual_reference_csv=None,
):
    cv2 = _require("cv2", "opencv-python")
    np = _require("numpy")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Cannot open: {}".format(video_path))
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] Video: {:.1f} fps, {} frames, {:.1f}s".format(
        video_fps, total_frames, total_frames / video_fps))

    start_frame = int(start_offset_sec * video_fps)
    frame_step = max(1, int(round(video_fps / sample_rate_fps)))
    if PROCESS_EVERY_SOURCE_FRAME:
        frame_step = 1
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps / frame_step))
    if debug:
        if debug_video:
            print("[INFO] Writing debug video from frames sampled about every {} source frames".format(
                debug_every_n))
        else:
            print("[INFO] Saving debug frame about every {} source frames".format(debug_every_n))
        if debug_enable_hitboxes:
            print("[INFO] Debug robot wireframe hitboxes enabled.")

    csv_path = os.path.join(output_dir, "robot_positions.csv")
    jlog_path = os.path.join(output_dir, "robot_positions.jlog")
    wpilog_path = os.path.join(output_dir, "match_log.wpilog")
    debug_dir = os.path.join(output_dir, "tracker_debug") if (debug and not debug_video) else None
    debug_video_path = os.path.join(output_dir, "tracker_debug.mp4") if (debug and debug_video) else None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        for name in os.listdir(debug_dir):
            if name.endswith(".jpg"):
                try:
                    os.remove(os.path.join(debug_dir, name))
                except OSError:
                    pass
    if debug_video_path and os.path.exists(debug_video_path):
        try:
            os.remove(debug_video_path)
        except OSError:
            pass
    avi_fallback_path = os.path.join(output_dir, "tracker_debug.avi")
    if debug_video_path and os.path.exists(avi_fallback_path):
        try:
            os.remove(avi_fallback_path)
        except OSError:
            pass

    log = WPILogWriter(wpilog_path)
    pose_eids = [log.start_entry("Robot{}/Pose".format(i), "double[]") for i in range(4)]
    vis_eids = [log.start_entry("Robot{}/Visible".format(i), "boolean") for i in range(4)]

    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(CSV_COLUMNS)
    jlog_writer = JuiceLogWriter(jlog_path, schema=ROBOT_POSE_SCHEMA)

    tracker = RobotTracker(cv2, np)
    shot_detector = ShotDetector(cv2, np)
    field_detector = FieldDetector(cv2, np)

    ordered = None
    H_2d = None
    H_inv = None

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, sample_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read first frame.")
        sys.exit(1)
    frame_shape = sample_frame.shape
    debug_writer = None
    if debug_video_path:
        debug_h, debug_w = frame_shape[:2]
        debug_fps = max(1.0, video_fps / max(1, debug_every_n))
        debug_writer, opened_path_or_attempts, debug_codec = _open_debug_video_writer(
            cv2,
            output_dir,
            (debug_w, debug_h),
            debug_fps,
        )
        if debug_writer is None:
            print("[WARN] Could not open debug video writer. Tried:")
            for attempt in opened_path_or_attempts:
                print("  {}".format(attempt))
        else:
            debug_video_path = opened_path_or_attempts
            print("[INFO] Debug video: {} [{} @ {:.2f} fps]".format(
                debug_video_path, debug_codec, debug_fps))

    if manual_corners_px:
        ordered = np.array(manual_corners_px, dtype=np.float32)
        dst2d = np.array([[0, 0], [144, 0], [144, 144], [0, 144]], dtype=np.float32)
        H_2d, _ = cv2.findHomography(ordered, dst2d)
        H_inv = np.linalg.inv(H_2d)
        tracker.setup(video_path, ordered, frame_shape)
        shot_detector.setup(tracker)
        print("[INFO] Using manual field corners.")
        if tracker._bg is not None:
            cv2.imwrite(os.path.join(output_dir, "median_background.jpg"), tracker._bg)

    if robot_init_positions:
        if H_inv is None:
            print("[ERROR] --robot-init-positions requires field corners to be known.")
            sys.exit(1)
        if len(robot_init_positions) != 4:
            print("[ERROR] --robot-init-positions must provide exactly 4 [x,y] pairs.")
            sys.exit(1)
        lineup = []
        for idx, item in enumerate(robot_init_positions):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                print("[ERROR] Robot init entry {} is invalid: {}".format(idx, item))
                sys.exit(1)
            fx_center = float(item[0])
            fy_center = float(item[1])
            fx, fy = _field_center_to_corner_xy(fx_center, fy_center)
            pt = np.array([[[fx, fy]]], dtype=np.float32)
            ip = cv2.perspectiveTransform(pt, H_inv)[0][0]
            lineup.append((fx, fy, float(ip[0]), float(ip[1])))
        tracker._initialize_tracks_from_lineup(lineup, "manual --robot-init-positions")

    if manual_reference_csv:
        if H_inv is None:
            print("[ERROR] --manual-reference-csv requires field corners to be known.")
            sys.exit(1)
        refs = _build_manual_reid_histograms(
            tracker,
            cap,
            manual_reference_csv,
            H_inv,
            video_fps,
        )
        if refs:
            tracker.set_reid_reference_histograms(refs)
            print("[INFO] Manual re-ID appearance model enabled.")

    frame_num = start_frame
    processed = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame if start_frame > 0 else 0)

    debug_colors = [(220, 160, 0), (0, 220, 255), (0, 0, 220), (0, 120, 255)]
    frames_to_process = max(1, (total_frames - start_frame + frame_step - 1) // frame_step)
    bar = _make_bar("Processing frames", frames_to_process)
    merge_debug_hold = 0
    dense_merge_hold = 0
    next_debug_source_frame = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_num = frame_num
        match_time_s = current_frame_num / video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s * 1_000_000))
        frame_num += 1

        process_dense = dense_merge_hold > 0 or bool(tracker._merge_groups)
        if not process_dense and (current_frame_num - start_frame) % frame_step != 0:
            continue

        bar.next()

        if ordered is None:
            result = field_detector.detect_field(frame)
            if result is not None:
                ordered, H_2d = result
                H_inv = np.linalg.inv(H_2d)
                tracker.setup(video_path, ordered, frame.shape)
                shot_detector.setup(tracker)
                print("\n  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                if tracker._bg is not None:
                    cv2.imwrite(os.path.join(output_dir, "median_background.jpg"), tracker._bg)

        poses = tracker.update(frame)
        shot_events = shot_detector.update(current_frame_num, match_time_s, poses, tracker)
        merge_active = bool(tracker._merge_groups)
        recent_merge = any(v > 0 for v in tracker._merge_recent)
        if merge_active or recent_merge:
            dense_merge_hold = tracker.MERGE_DEBUG_HOLD
        elif dense_merge_hold > 0:
            dense_merge_hold -= 1
        if merge_active:
            merge_debug_hold = tracker.MERGE_DEBUG_HOLD
        elif merge_debug_hold > 0:
            merge_debug_hold -= 1

        for i, pose in enumerate(poses):
            log.write_pose2d(
                pose_eids[i],
                timestamp_us,
                pose.wpilog_x_m,
                pose.wpilog_y_m,
                pose.wpilog_heading_rad,
            )
            log.write_boolean(vis_eids[i], timestamp_us, pose.visible)

        row = {"timestamp_s": "{:.4f}".format(match_time_s)}
        for robot_id, pose in enumerate(poses):
            row["robot{}_x_in".format(robot_id)] = "{:.2f}".format(pose.x_center_in)
            row["robot{}_y_in".format(robot_id)] = "{:.2f}".format(pose.y_center_in)
            row["robot{}_heading_rad".format(robot_id)] = "{:.4f}".format(pose.heading)
            row["robot{}_visible".format(robot_id)] = "1" if pose.visible else "0"
        shot_cells = [["", "", "", ""] for _ in range(4)]
        for event in shot_events:
            if 0 <= event.shooter_id < 4:
                shot_cells[event.shooter_id] = [
                    event.result,
                    "{:.2f}".format(event.shot_x_in),
                    "{:.2f}".format(event.shot_y_in),
                    event.goal_color,
                ]
        for robot_id, cells in enumerate(shot_cells):
            row["robot{}_shot_result".format(robot_id)] = cells[0]
            row["robot{}_shot_x_in".format(robot_id)] = cells[1]
            row["robot{}_shot_y_in".format(robot_id)] = cells[2]
            row["robot{}_shot_goal".format(robot_id)] = cells[3]
        csv_writer.writerow(csv_row_to_list(row))
        jlog_writer.append_row(row)

        if debug and H_inv is not None and (debug_dir or debug_writer is not None):
            dbg = frame.copy()
            fg_debug = None
            if ordered is not None:
                cv2.polylines(dbg, [ordered.astype(np.int32).reshape(-1, 1, 2)],
                              True, (0, 200, 0), 2)
                shot_detector.draw_debug(dbg)

            if tracker._bg is not None:
                fg_debug = tracker._foreground_mask(frame)
                all_cnts, split_centers = tracker._foreground_contours(frame)
                if tracker._last_ball_mask is not None:
                    ball_cnts, _ = cv2.findContours(
                        tracker._last_ball_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )
                    for contour in ball_cnts:
                        if cv2.contourArea(contour) >= 20:
                            cv2.drawContours(dbg, [contour], -1, (255, 0, 255), 1)
                for contour in all_cnts:
                    area = cv2.contourArea(contour)
                    mask = np.zeros(tracker._field_mask.shape, np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
                    is_robot = dist.max() >= tracker.MIN_RADIUS_PX
                    color = (0, 255, 255) if is_robot else (160, 160, 160)
                    cv2.drawContours(dbg, [contour], -1, color, 2 if is_robot else 1)
                    if area >= tracker.BLOB_MIN:
                        moments = cv2.moments(contour)
                        if moments["m00"] > 0:
                            bx = int(moments["m10"] / moments["m00"])
                            by = int(moments["m01"] / moments["m00"])
                            cv2.putText(dbg, "{:.0f}".format(area),
                                        (bx - 15, by + 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                for sx, sy in split_centers[:4]:
                    cv2.drawMarker(dbg, (sx, sy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 16, 2)

            fh_d, fw_d = frame.shape[:2]
            cv2.circle(dbg, (fw_d - 40, 40),
                       int(tracker._robot_max_px / 2), (80, 80, 80), 1)
            cv2.putText(dbg, "18in", (fw_d - 74, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)

            for merge_group in tracker._merge_groups.values():
                pos_list = [tracker._pos_px[tid] for tid in merge_group.track_ids
                            if tracker._pos_px[tid] is not None]
                if pos_list:
                    mcx = int(sum(point[0] for point in pos_list) / len(pos_list))
                    mcy = int(sum(point[1] for point in pos_list) / len(pos_list))
                    ax, ay = merge_group.entry_axis
                    ex = int(mcx + ax * 50)
                    ey = int(mcy + ay * 50)
                    if len(merge_group.track_ids) == 2:
                        color = (0, 80, 255) if merge_group.crossed else (0, 220, 100)
                        label = "+".join("R{}".format(tid) for tid in merge_group.track_ids)
                        label += " CROSSED" if merge_group.crossed else " ok"
                    else:
                        color = (255, 160, 0)
                        entry_s = "".join(str(tid) for tid in merge_group.entry_order)
                        current_s = "".join(str(tid) for tid in merge_group.current_order)
                        label = "+".join("R{}".format(tid) for tid in merge_group.track_ids)
                        label += " [{}→{}]".format(entry_s, current_s)
                    cv2.arrowedLine(dbg, (mcx, mcy), (ex, ey), color, 2)
                    cv2.putText(dbg, label, (mcx + 6, mcy - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            n_drawn = 0
            fh, fw = frame.shape[:2]
            for i, pose in enumerate(poses):
                if not pose.visible:
                    continue
                pt = np.array([[[pose.x_in, pose.y_in]]], dtype=np.float32)
                img = cv2.perspectiveTransform(pt, H_inv)[0][0]
                ix, iy = int(img[0]), int(img[1])
                if not (0 <= ix < fw and 0 <= iy < fh):
                    continue
                color = debug_colors[i]
                cv2.circle(dbg, (ix, iy), 16, color, -1)
                cv2.circle(dbg, (ix, iy), 19, (255, 255, 255), 2)
                if debug_enable_hitboxes and fg_debug is not None:
                    _draw_wire_cube(cv2, np, dbg, frame, fg_debug,
                                    tracker, H_inv, pose, (ix, iy), color)
                cv2.putText(dbg, "R{}".format(i), (ix - 10, iy - 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                vx_px, vy_px = tracker._vel_px[i]
                speed_px = math.hypot(vx_px, vy_px)
                if speed_px >= 0.35:
                    scale = 28.0 / speed_px
                    dx = int(round(vx_px * scale))
                    dy = int(round(vy_px * scale))
                    cv2.arrowedLine(dbg, (ix, iy), (ix + dx, iy + dy),
                                    (255, 255, 255), 2)
                n_drawn += 1

            n_merged = sum(len(merge_group.track_ids) for merge_group in tracker._merge_groups.values())
            label = ("init" if not tracker._initialized else
                     "{}/4 ({} merged)".format(n_drawn, n_merged) if n_merged else
                     "{}/4".format(n_drawn))
            cv2.putText(dbg, "t={:.2f}s f={}  [{}]".format(
                        match_time_s, current_frame_num, label),
                        (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            base_debug_interval = max(1, debug_every_n)
            save_debug = (current_frame_num >= next_debug_source_frame
                          or (SAVE_ALL_DEBUG_AROUND_MERGES and
                              (merge_active or merge_debug_hold > 0)))
            if save_debug:
                if debug_writer is not None:
                    debug_writer.write(dbg)
                elif debug_dir is not None:
                    cv2.imwrite(os.path.join(debug_dir,
                                "frame_{:06d}.jpg".format(current_frame_num)), dbg)
                if current_frame_num >= next_debug_source_frame:
                    while current_frame_num >= next_debug_source_frame:
                        next_debug_source_frame += base_debug_interval

        for event in shot_events:
            print("[INFO] Shot {} by R{} at ({:.1f},{:.1f}) goal={}".format(
                event.result,
                event.shooter_id,
                event.shot_x_in,
                event.shot_y_in,
                event.goal_color or "unknown",
            ))

        processed += 1

    bar.finish()
    if debug_writer is not None:
        debug_writer.release()
    log.close()
    csv_file.close()
    jlog_writer.close()
    cap.release()
    print("\n[DONE] {} frames processed.".format(processed))
    print("  CSV:    {}".format(csv_path))
    print("  JLOG:   {}".format(jlog_path))
    print("  WPILOG: {}".format(wpilog_path))
    if debug:
        print("  Debug:  {}".format(debug_video_path if debug_video_path else debug_dir))
    _print_instructions()


def _print_instructions():
    print("""
+------------------------------------------------------------------+
|         AdvantageScope Visualization Instructions                |
+------------------------------------------------------------------+
|  1. Open AdvantageScope                                          |
|  2. File > Open Log(s) ... > select match_log.wpilog             |
|  3. "+" tab -> 2D Field -> set Field to FTC DECODE season        |
|  4. Drag Robot0/Pose into Poses                                  |
|  5. Click icon LEFT of the field name ->                         |
|       Format: Pose2d   Units: Meters + Radians                   |
|  6. Repeat for Robot1, Robot2, Robot3                            |
|  7. Press play!                                                  |
|                                                                  |
|  Robot IDs: assigned left-to-right at match start.               |
|  Debug overlay: 2-robot merges show CROSSED/ok; 3+ show          |
|  [entry_order->current_order] permutation string.                |
+------------------------------------------------------------------+
""")


def download_video(url, output_dir):
    import subprocess

    out = os.path.join(output_dir, "match_video.mp4")
    print("[INFO] Downloading:", url)
    bar = _make_bar("Downloading", 100)
    last_pct = 0
    proc = subprocess.Popen(
        ["yt-dlp", "-f", "best[ext=mp4]", "-o", out, url,
         "--progress-template", "%(progress._percent_str)s"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.stdout:
        for line in proc.stdout:
            try:
                pct = int(float(line.strip().rstrip("%")))
                while last_pct < pct:
                    bar.next()
                    last_pct += 1
            except ValueError:
                pass
    proc.wait()
    while last_pct < 100:
        bar.next()
        last_pct += 1
    bar.finish()
    if proc.returncode != 0:
        print("[ERROR] yt-dlp failed")
        sys.exit(1)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Track FTC DECODE robots from a match video.")
    parser.add_argument("url", nargs="?", help="YouTube URL")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--start-offset", type=float, default=0.0,
                        help="Seconds to skip before match timer starts")
    parser.add_argument("--sample-rate", type=float, default=10.0,
                        help="Frames per second to process (default 10)")
    parser.add_argument("--debug", action="store_true",
                        help="Save annotated debug frames to tracker_debug/")
    parser.add_argument("--debug-video", action="store_true",
                        help="Write annotated debug output to tracker_debug.mp4 instead of tracker_debug/ images; implies --debug")
    parser.add_argument("--debug-every", type=int, default=1,
                        help="Save 1 debug frame every N processed frames (default 1)")
    parser.add_argument("--debug-enable-hitboxes", action="store_true",
                        help="Draw robot wireframe hitboxes in debug frames")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--corners", default=None,
                        help="field_corners.json from ftc_calibrate.py")
    parser.add_argument("--robot-init-positions", default=None,
                        help=(
                            "JSON file or inline JSON string with starting field positions "
                            "for all 4 robots, sorted by robot id. "
                            "Format: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] in inches "
                            "with (0,0) at field center. "
                            "Bypasses blob-based init (required when robots start under "
                            "corner structures or are otherwise hard to detect at t=0). "
                            "Example: --robot-init-positions "
                            "'[[-52.7,-70.4],[-12.6,53.0],[15.8,60.6],[57.5,-55.7]]'"))
    parser.add_argument("--manual-reference-csv", default=None,
                        help=(
                            "Manual robot_positions-style CSV or JLOG used to build a supervised "
                            "appearance re-ID model for this video. "
                            "Coordinates are interpreted with (0,0) at field center. "
                            "Useful for tuning tracker identity assignment against "
                            "hand-labeled data."
                        ))
    args = parser.parse_args()

    if args.no_download:
        if not args.video_path:
            parser.error("--no-download requires --video-path")
        video_path = args.video_path
    else:
        if not args.url:
            parser.error("YouTube URL required (or --no-download --video-path)")
        os.makedirs(args.output_dir, exist_ok=True)
        video_path = download_video(args.url, args.output_dir)

    print("[INFO] Output:", args.output_dir)
    corners = None
    if args.corners:
        with open(args.corners) as file:
            corners = json.load(file)["corners_px"]
        print("[INFO] Corners:", args.corners)

    robot_init = None
    if args.robot_init_positions:
        raw = args.robot_init_positions.strip()
        if os.path.isfile(raw):
            with open(raw) as file:
                robot_init = json.load(file)
        else:
            robot_init = json.loads(raw)
        print("[INFO] Robot init positions:", robot_init)

    process_match(
        video_path=video_path,
        output_dir=args.output_dir,
        start_offset_sec=args.start_offset,
        sample_rate_fps=args.sample_rate,
        debug=(args.debug or args.debug_video),
        debug_video=args.debug_video,
        debug_every_n=args.debug_every,
        debug_enable_hitboxes=args.debug_enable_hitboxes,
        manual_corners_px=corners,
        robot_init_positions=robot_init,
        manual_reference_csv=args.manual_reference_csv,
    )


if __name__ == "__main__":
    main()
