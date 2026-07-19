"""
AutoScout Main Entry Point

Orchestrates the complete tracking pipeline for FTC match videos:
1. Validates and downloads video (from local path or YouTube URL)
2. Auto-detects field corners (or uses provided calibration)
3. Initializes tracker and shot detector
4. Processes video frame-by-frame
5. Exports results to multiple formats (CSV, JLOG, WPILog)
6. Generates debug video/images if requested

Command-Line Usage:
    python auto_scout.py EVENT_CODE MATCH_ID [options]
    
    Typical:
    >>> python auto_scout.py 2024txho qm27
    >>> python auto_scout.py 2024txho qm27 --debug --debug-video
    >>> python auto_scout.py https://youtube.com/watch?v=xyz --video-path local.mp4

    Field Calibration:
    >>> python tools/calibrate.py match.mp4 --output field_corners.json --frame 100
    >>> python auto_scout.py 2024txho qm27 --corners field_corners.json
    
    Manual Re-ID Bootstrapping:
    >>> python auto_scout.py 2024txho qm27 --manual-reference-csv manual_poses.csv

Key Options:
    --no-download: Use local video file (skip YouTube fetch)
    --video-path: Local video file path
    --corners: JSON file with field corner calibration
    --robot-init-positions: Initial robot guess (JSON)
    --manual-reference-csv: Re-ID histogram training data
    --debug: Enable debug output and frame overlay
    --debug-video: Save debug video with overlays
    --sample-rate-fps: Downsampling rate (skip frames)
    --output-dir: Output directory (default: current dir)

Output Files (in output_dir):
    - robot_positions.csv: Full tracking data (48 columns)
    - robot_positions.jlog: Compact binary format (~90% smaller)
    - match_log.wpilog: AdvantageScope-compatible format
    - debug_background.jpg: Detected background (if --debug)
    - debug_frames/: Debug frame overlays (if --debug)
    - debug_video.mp4: Debug video (if --debug-video)

Performance:
    - Real-time: ~10 fps on modern 4-core CPU (640×360 video)
    - Memory: ~5-10 MB footprint
    - File size: CSV ~2-5 MB → JLOG ~200-500 KB (90% reduction)

Dependencies:
    - opencv-python: Video I/O, computer vision
    - numpy: Array operations
    - scipy (optional): Hungarian algorithm optimization
    - yt-dlp (optional): YouTube downloads
    - progress (optional): Console progress bars

Dashboard Integration:
    - If AUTOSCOUT_DASHBOARD_MODE=1: Emit JSON events to stdout
    - Real-time progress: file:// URLs for preview frames
    - Compatible with dashboard_server.py for web UI
"""

import argparse
import base64
import csv
import json
import math
import os
import sys
from pathlib import Path

from autoscout.geometry import _field_center_to_corner_xy
from autoscout.helpers import (
    FieldDetector,
    _build_manual_reid_histograms,
    _draw_wire_cube,
    _open_debug_video_writer,
)
from autoscout.match_timer import scan_match_bounds
from autoscout.runtime import _make_bar, _require, install_console_output
from autoscout.shot import ShotDetector
from autoscout.tracker import RobotTracker
from autoscout.wpilog import WPILogWriter
from autoscout.ytdlp import (
    auth_help_message,
    build_download_attempts,
    find_yt_dlp_command,
    get_yt_dlp_command_warning,
    output_indicates_rate_limit,
    rate_limit_help_message,
    tls_help_message,
    yt_dlp_network_backoff_args,
)
from util.juice_log import (
    CSV_COLUMNS,
    ROBOT_POSE_SCHEMA,
    JuiceLogWriter,
    csv_row_to_list,
)


SAVE_ALL_DEBUG_AROUND_MERGES = False
PROCESS_EVERY_SOURCE_FRAME = True
DASHBOARD_EVENT_PREFIX = "@@AUTOSCOUT_DASHBOARD@@"


install_console_output()


def _dashboard_mode_enabled():
    return os.environ.get("AUTOSCOUT_DASHBOARD_MODE") == "1"


def _dashboard_preview_interval():
    raw = os.environ.get("AUTOSCOUT_DASHBOARD_PREVIEW_EVERY", "5").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 5


def _emit_dashboard_event(kind, **payload):
    if not _dashboard_mode_enabled():
        return
    event = {"kind": kind}
    event.update(payload)
    print(DASHBOARD_EVENT_PREFIX + json.dumps(event, separators=(",", ":")), flush=True)


def _emit_dashboard_stage(stage, status, detail=""):
    _emit_dashboard_event("stage", stage=stage, status=status, detail=detail)


def _emit_dashboard_stage_progress(stage, detail, current, total):
    total = max(1, int(total))
    current = max(0, min(int(current), total))
    _emit_dashboard_event(
        "stage",
        stage=stage,
        status="in_progress",
        detail=detail,
        progress_current=current,
        progress_total=total,
        progress_fraction=(current / total),
    )


def _emit_dashboard_preview(cv2, frame, frame_num, match_time_s):
    if not _dashboard_mode_enabled():
        return
    ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), 70],
    )
    if not ok:
        return
    _emit_dashboard_event(
        "preview",
        frame_num=int(frame_num),
        match_time_s=round(float(match_time_s), 2),
        jpeg_b64=base64.b64encode(encoded.tobytes()).decode("ascii"),
    )


def _manual_corners_file_to_tracker_order(manual_corners_px):
    if not manual_corners_px or len(manual_corners_px) != 4:
        return manual_corners_px
    # Saved calibration files are persisted as [BL, BR, TR, TL].
    # The tracker homography expects [TL, TR, BR, BL].
    bl, br, tr, tl = manual_corners_px
    return [tl, tr, br, bl]


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
    auto_match_bounds=False,
):
    """
    Orchestrate complete tracking pipeline for an FTC match video.
    
    Main Processing Steps:
        1. Load video file, validate format and frame count
        2. Optionally auto-detect match bounds (via broadcast timer OCR)
        3. Auto-detect field corners (or use calibration if provided)
        4. Initialize tracker with background model and homography
        5. Frame-by-frame tracking loop:
            - Extract and assign blobs to robot tracks
            - Detect ball launches and goal entries
            - Manage robot collisions (merge groups)
        6. Export results: CSV, JLOG (compressed), WPILog (AdvantageScope)
        7. Generate debug outputs (overlays, video)
    
    Frame Sampling:
        - Processes frames at sample_rate_fps (e.g., 10 fps from 30 fps video)
        - Converts to frame_step for efficient skipping
        - Can process every frame if sample_rate_fps >= video_fps
    
    Field Calibration:
        - Primary: Auto-detect field corners via edge detection
        - Fallback: Use manual_corners_px (CLI: --corners file.json)
        - Validates corner order and homography quality
    
    Tracker Initialization:
        - Builds median background from N samples across video
        - Computes perspective homography (pixel ↔ field coords)
        - Initializes merge group state machine
    
    Output Generation:
        - CSV: Full 48-column tracking data (human-readable)
        - JLOG: Compressed binary format (~90% smaller than CSV)
        - WPILog: AdvantageScope-compatible binary (for replay in dashboards)
        - Debug frames/video: Robot overlays (if --debug)
    
    Args:
        video_path (str): Path to match video file (.mp4, .mov, etc.)
        output_dir (str): Directory for output files (created if missing)
        start_offset_sec (float): Match start time (skip opening ceremonies)
        sample_rate_fps (float): Frame sampling rate (10 fps typical)
        debug (bool): Generate debug overlays (frames/images)
        debug_video (bool): Generate debug video (slower, larger files)
        debug_every_n (int): For debug: only show every Nth frame
        debug_enable_hitboxes (bool): Draw robot/ball bounding boxes
        manual_corners_px (Optional[Dict]): Field corner calibration
            {"tl": [x,y], "tr": [x,y], "br": [x,y], "bl": [x,y]}
        robot_init_positions (Optional[Dict]): Initial robot guesses (for bootstrap)
        manual_reference_csv (Optional[str]): Re-ID histogram training data (hand-labeled)
        auto_match_bounds (bool): Auto-detect match start/end via timer OCR
    
    Returns:
        None (writes output files to output_dir)
    
    Output Files Created:
        - robot_positions.csv: Tracking data (columns match util.juice_log.CSV_COLUMNS)
        - robot_positions.jlog: Compressed JLOG format
        - match_log.wpilog: WPILog v1 format for AdvantageScope
        - median_background.jpg: Background subtraction reference (if --debug)
        - debug_frames/: Individual frames with overlays (if --debug)
        - debug_video.mp4: Video with real-time overlays (if --debug-video)
    
    Exceptions:
        - SystemExit: If video cannot be opened or dependencies missing
        - RuntimeError: If field corners cannot be detected
    
    Dashboard Integration:
        - If AUTOSCOUT_DASHBOARD_MODE=1: Emits JSON events to stdout
        - Events include: stage progress, preview frames (as data: URLs), final stats
        - Enables real-time dashboard updates during processing
    
    Performance Notes:
        - Processing time: ~5-10 seconds per minute of video (downsampled to 10 fps)
        - Memory: ~5-10 MB for full tracking state
        - Output size: ~2-5 MB CSV → ~200-500 KB JLOG (90% reduction)
    """
    cv2 = _require("cv2", "opencv-python")
    np = _require("numpy")
    dashboard_stream_previews = _dashboard_mode_enabled() and debug and not debug_video

    os.makedirs(output_dir, exist_ok=True)

    _emit_dashboard_stage("Loading Match Footage", "in_progress", "Opening match footage")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _emit_dashboard_stage("Loading Match Footage", "failed", "Unable to open match footage")
        print("[ERROR] Cannot open: {}".format(video_path))
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("[INFO] Video: {:.1f} fps, {} frames, {:.1f}s".format(
        video_fps, total_frames, total_frames / video_fps))

    start_frame = int(start_offset_sec * video_fps)
    stop_frame = total_frames - 1
    if auto_match_bounds:
        timer_scan = scan_match_bounds(
            video_path,
            cv2,
            np,
            progress_callback=lambda current, total: _emit_dashboard_stage_progress(
                "Loading Match Footage",
                "Scanning broadcast timer for match bounds",
                current,
                total,
            ),
        )
        auto_start_frame = timer_scan.get("start_frame")
        auto_stop_frame = timer_scan.get("stop_frame")
        if auto_start_frame is not None:
            start_frame = max(0, int(auto_start_frame))
            start_offset_sec = start_frame / video_fps
            print("[INFO] Auto-detected match start at {:.2f}s (frame {}).".format(
                start_offset_sec,
                start_frame,
            ))
        else:
            start_offset_sec = 0.0
            start_frame = 0
            print("[WARN] Could not auto-detect the match start timer. Starting from frame 0.")
        if auto_stop_frame is not None:
            stop_frame = int(auto_stop_frame)
            stop_time_s = stop_frame / video_fps
            print("[INFO] Auto-detected match end at {:.2f}s (frame {}).".format(
                stop_time_s,
                stop_frame,
            ))
        else:
            print("[WARN] Could not auto-detect the end of match timer. Processing until the video ends.")
    else:
        print("[INFO] Using explicit start offset of {:.2f}s (frame {}).".format(
            start_offset_sec,
            start_frame,
        ))

    frame_step = max(1, int(round(video_fps / sample_rate_fps)))
    if PROCESS_EVERY_SOURCE_FRAME:
        frame_step = 1
    print("[INFO] Processing every {} frames (~{:.1f} fps output)".format(
        frame_step, video_fps / frame_step))
    _emit_dashboard_stage("Loading Match Footage", "completed", "Match footage is ready")
    _emit_dashboard_stage("Calibrating Trackers", "in_progress", "Preparing tracker calibration")
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
    debug_dir = os.path.join(output_dir, "tracker_debug") if (debug and not debug_video and not dashboard_stream_previews) else None
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
        ordered = np.array(_manual_corners_file_to_tracker_order(manual_corners_px), dtype=np.float32)
        dst2d = np.array([[0, 0], [144, 0], [144, 144], [0, 144]], dtype=np.float32)
        H_2d, _ = cv2.findHomography(ordered, dst2d)
        H_inv = np.linalg.inv(H_2d)
        tracker.setup(video_path, ordered, frame_shape)
        shot_detector.setup(tracker)
        print("[INFO] Using manual field corners.")
        _emit_dashboard_stage("Calibrating Trackers", "completed", "Using manual field corners")
        _emit_dashboard_stage("Tracking Match", "in_progress", "Tracking robots across the match")
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
    end_frame_exclusive = min(total_frames, stop_frame + 1)
    frames_to_process = max(1, (end_frame_exclusive - start_frame + frame_step - 1) // frame_step)
    bar = _make_bar("Processing frames", frames_to_process)
    merge_debug_hold = 0
    dense_merge_hold = 0
    next_debug_source_frame = start_frame
    dashboard_preview_every = max(debug_every_n, _dashboard_preview_interval()) if dashboard_stream_previews else max(1, debug_every_n)
    calibration_completed = H_inv is not None
    tracking_started = H_inv is not None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_num = frame_num
        if current_frame_num > stop_frame:
            break
        match_time_s = current_frame_num / video_fps - start_offset_sec
        timestamp_us = max(0, int(match_time_s * 1_000_000))
        frame_num += 1

        process_dense = dense_merge_hold > 0 or bool(tracker._merge_groups)
        if not process_dense and (current_frame_num - start_frame) % frame_step != 0:
            continue

        bar.next()
        if processed == 0 or processed % 30 == 0 or processed + 1 >= frames_to_process:
            _emit_dashboard_stage_progress(
                "Tracking Match",
                "Tracking robots across the match",
                processed + 1,
                frames_to_process,
            )

        if ordered is None:
            result = field_detector.detect_field(frame)
            if result is not None:
                ordered, H_2d = result
                H_inv = np.linalg.inv(H_2d)
                tracker.setup(video_path, ordered, frame.shape)
                shot_detector.setup(tracker)
                print("\n  [t={:.1f}s] Field auto-detected.".format(match_time_s))
                calibration_completed = True
                _emit_dashboard_stage("Calibrating Trackers", "completed", "Field auto-detected")
                if not tracking_started:
                    tracking_started = True
                    _emit_dashboard_stage("Tracking Match", "in_progress", "Tracking robots across the match")
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

        if debug and H_inv is not None and (dashboard_stream_previews or debug_dir or debug_writer is not None):
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
            base_debug_interval = dashboard_preview_every
            save_debug = (current_frame_num >= next_debug_source_frame
                          or (SAVE_ALL_DEBUG_AROUND_MERGES and
                              (merge_active or merge_debug_hold > 0)))
            if save_debug:
                if debug_writer is not None:
                    debug_writer.write(dbg)
                elif dashboard_stream_previews:
                    _emit_dashboard_preview(cv2, dbg, current_frame_num, match_time_s)
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

    if not calibration_completed:
        _emit_dashboard_stage("Calibrating Trackers", "failed", "Field calibration could not be established")
    elif tracking_started:
        _emit_dashboard_stage("Tracking Match", "completed", "Tracking loop finished")
    _emit_dashboard_stage("Cleaning Up", "in_progress", "Finalizing outputs")
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
        debug_location = debug_video_path if debug_video_path else (debug_dir or "dashboard stream")
        print("  Debug:  {}".format(debug_location))
    _emit_dashboard_stage("Cleaning Up", "completed", "Outputs are ready")
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
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print("[INFO] Reusing previously downloaded match video:", out)
        _emit_dashboard_stage("Loading Match Footage", "completed", "Reusing previously downloaded match footage")
        return out
    print("[INFO] Downloading:", url)
    _emit_dashboard_stage("Loading Match Footage", "in_progress", "Downloading match footage from YouTube")
    yt_dlp_cmd = find_yt_dlp_command()
    if not yt_dlp_cmd:
        _emit_dashboard_stage("Loading Match Footage", "failed", "yt-dlp is not available")
        print("[ERROR] yt-dlp is not installed for this Python and was not found on PATH.")
        sys.exit(1)
    yt_dlp_warning = get_yt_dlp_command_warning(yt_dlp_cmd)
    if yt_dlp_warning:
        print(yt_dlp_warning)
    attempts, attempt_notes = build_download_attempts(
        [Path.cwd(), Path(__file__).resolve().parent, Path(output_dir).resolve()]
    )
    for note in attempt_notes:
        print(note)
    last_recent_output = []

    for attempt_index, attempt in enumerate(attempts, start=1):
        if attempt_index > 1:
            print("[INFO] Retrying yt-dlp with {}...".format(attempt["label"]))
        bar = _make_bar("Downloading", 100)
        last_pct = 0
        recent_output = []
        proc = subprocess.Popen(
            [
                *yt_dlp_cmd,
                "--no-playlist",
                "--merge-output-format", "mp4",
                *yt_dlp_network_backoff_args(),
                "-o", out,
                *attempt["args"],
                url,
                "--progress-template", "%(progress._percent_str)s",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True)
        if proc.stdout:
            for line in proc.stdout:
                stripped = line.strip()
                if stripped:
                    recent_output.append(stripped)
                    recent_output = recent_output[-12:]
                try:
                    pct = int(float(stripped.rstrip("%")))
                    while last_pct < pct:
                        bar.next()
                        last_pct += 1
                    _emit_dashboard_stage_progress(
                        "Loading Match Footage",
                        "Downloading match footage from YouTube",
                        last_pct,
                        100,
                    )
                except ValueError:
                    pass
        proc.wait()
        while last_pct < 100:
            bar.next()
            last_pct += 1
            _emit_dashboard_stage_progress(
                "Loading Match Footage",
                "Downloading match footage from YouTube",
                last_pct,
                100,
            )
        bar.finish()
        last_recent_output = recent_output
        if proc.returncode == 0 and os.path.exists(out):
            _emit_dashboard_stage("Loading Match Footage", "completed", "Downloaded match footage from YouTube")
            return out
        if output_indicates_rate_limit(recent_output):
            print("[WARN] yt-dlp hit a YouTube 429 / Too Many Requests response. Stopping further retries to avoid additional throttling.")
            break

    if last_recent_output:
        _emit_dashboard_stage("Loading Match Footage", "failed", "Unable to download match footage from YouTube")
        print("[ERROR] yt-dlp failed. Last output:")
        for item in last_recent_output:
            print("  {}".format(item))
        joined_output = "\n".join(item.lower() for item in last_recent_output)
        blocked_by_youtube = (
            "sign in to confirm you're not a bot" in joined_output
            or "sign in to confirm you’re not a bot" in joined_output
            or "precondition check failed" in joined_output
            or "po token" in joined_output
        )
        tls_failed = (
            "certificate_verify_failed" in joined_output
            or "unable to get local issuer certificate" in joined_output
        )
        storyboards_only = (
            "only images are available for download" in joined_output
            or "requested format is not available" in joined_output
        )
        if tls_failed:
            print("[ERROR] Python TLS certificate verification failed. {}".format(
                tls_help_message()
            ))
        elif blocked_by_youtube:
            print("[ERROR] YouTube is challenging this session/IP. The anonymous retry paths were exhausted. {}".format(
                auth_help_message()
            ))
        elif storyboards_only:
            print("[ERROR] YouTube only exposed storyboard/blocked formats to the anonymous clients we tried. {}".format(
                auth_help_message()
            ))
        elif "nsig extraction failed" in joined_output:
            print("[ERROR] Your installed yt-dlp may be outdated. Updating yt-dlp often fixes this YouTube extractor error.")
        if output_indicates_rate_limit(last_recent_output):
            print("[ERROR] {}".format(rate_limit_help_message()))
        if "operation not permitted" in joined_output and "cookies" in joined_output:
            print("[ERROR] yt-dlp could not read a protected browser cookie store. {}".format(auth_help_message()))
    else:
        _emit_dashboard_stage("Loading Match Footage", "failed", "Unable to download match footage from YouTube")
        print("[ERROR] yt-dlp failed")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Track FTC DECODE robots from a match video.")
    parser.add_argument("url", nargs="?", help="YouTube URL")
    parser.add_argument("--output-dir", default="./output")
    parser.add_argument("--start-offset", type=float, default=0.0,
                        help="Seconds to skip before match timer starts")
    parser.add_argument("--auto-match-bounds", action="store_true",
                        help="Auto-detect match start/end from the broadcast timer")
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
        auto_match_bounds=args.auto_match_bounds,
    )


if __name__ == "__main__":
    main()
