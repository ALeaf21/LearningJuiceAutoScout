"""
Shot Detection Engine for FTC Matches

This module detects ball shots (launches and goals) in FTC match videos by:
1. Segmenting ball pixels from background using HSV color masks (green/purple)
2. Extracting ball contours and associating them into continuous tracks
3. Detecting shot launch events (acceleration, minimum speed, upward motion)
4. Tracking goal entry via polygon-based approach detection
5. Classifying shots as made/missed based on goal entry confirmation

Key Concepts:
    - Ball Detection: HSV thresholding for green (alliance) and purple (opponent) balls
    - Ball Tracks: Temporal associations of ball positions, indexed by track_id
    - Shot Launch: Triggered when ball accelerates away from a robot (shooter)
    - Goal Entry: Confirmed when ball enters goal opening polygon and stays for N frames
    - Deduplication: Suppress duplicate events within 0.35s (same shot, multiple detections)

Main Workflow (per frame):
    1. Extract ball contours via HSV color masking
    2. Associate detections into persistent ball tracks
    3. For each track: check launch criteria, track goal entry
    4. When track is lost (>12 frames missing): resolve as made/missed shot event
    5. Deduplicate events by shooter_id and time window

Configuration Constants:
    - BALL_MIN/MAX_CONTOUR_AREA_PX: Size filtering (8-420 px²)
    - SHOT_MIN_SPEED_PX: Minimum launch velocity (2 px/frame)
    - SHOT_MIN_UPWARD_PX: Minimum upward component (3 pixels)
    - SHOT_MIN_NEGATIVE_STEP_RATIO: Acceleration threshold (35% speed increase)
    - GOAL_*: Goal polygon detection and approach zone parameters

Dependencies:
    - models.py: BallTrack, ShotEvent dataclasses
    - geometry.py: Coordinate transformations (_field_center_to_corner_xy)
    - OpenCV (cv2): Contour detection, morphology, perspective transforms
    - NumPy (np): Array operations, distance computations
"""

import math
from typing import Dict, List, Optional, Tuple

from autoscout.geometry import _field_center_to_corner_xy
from autoscout.models import BallTrack, ShotEvent


class ShotDetector:
    """
    Ball tracking and shot detection engine.
    
    Detects ball launches from robots and determines if shots successfully enter the goal.
    Uses HSV color masking for ball segmentation and geometric polygon detection for goal
    entry confirmation.
    
    Attributes:
        _tracks (Dict[int, BallTrack]): Active ball tracks keyed by track_id
        _goal_openings (Dict[str, np.ndarray]): Goal opening polygons {"blue": poly, "red": poly}
        _goal_approaches (Dict[str, np.ndarray]): Approach zones for goal entry detection
        _last_event_time_by_robot (Dict[int, float]): Timestamp of last shot for deduplication
        _ready (bool): False until setup() completes goal detection
    
    Lifecycle:
        1. __init__(cv2, np): Create detector with CV2/NumPy handles
        2. setup(tracker): Extract goal positions from field background image
        3. update(frame_num, match_time_s, poses, tracker) → List[ShotEvent]: Per-frame detection
        4. draw_debug(dbg): Overlay goal zones and ball tracks on debug frame
    
    Example:
        >>> detector = ShotDetector(cv2, np)
        >>> detector.setup(tracker)
        >>> for frame_num, frame in enumerate(video_frames):
        ...     events = detector.update(frame_num, match_time_s, poses, tracker)
        ...     for event in events:
        ...         print(f"Shot by R{event.shooter_id}: {'made' if event.made else 'missed'}")
    """
    BALL_MIN_CONTOUR_AREA_PX = 8.0
    BALL_MAX_CONTOUR_AREA_PX = 420.0
    BALL_ASSOC_DIST_PX = 34.0
    BALL_ASSOC_DIST_FAST_PX = 124.0
    BALL_MAX_MISSING = 12
    SHOT_MAX_START_DIST_PX = 84.0
    SHOT_MIN_DISP_PX = 8.0
    SHOT_MIN_SPEED_PX = 2.0
    SHOT_MIN_UPWARD_PX = 3.0
    SHOT_MIN_LIFE_FRAMES = 2
    SHOT_MIN_NEGATIVE_STEP_RATIO = 0.35
    SHOT_MIN_AWAY_FROM_SHOOTER_PX = 2.0
    SHOT_DEDUP_TIME_S = 0.35
    GOAL_OPENING_TOP_FRAC = 0.58
    GOAL_APPROACH_SIDE_PAD_FRAC = 0.30
    GOAL_APPROACH_DOWN_FRAC = 0.95
    GOAL_CONFIRM_MIN_FRAMES = 2
    GOAL_ENTRY_MAX_MISSING = 5
    GOAL_OPENING_SHRINK_X = 0.84
    GOAL_OPENING_SHRINK_Y = 0.80
    GOAL_BLUE_HSV_LO = (95, 80, 45)
    GOAL_BLUE_HSV_HI = (135, 255, 255)
    GOAL_RED1_HSV_LO = (0, 90, 45)
    GOAL_RED1_HSV_HI = (12, 255, 255)
    GOAL_RED2_HSV_LO = (170, 90, 45)
    GOAL_RED2_HSV_HI = (179, 255, 255)
    GOAL_MIN_AREA_PX = 900.0

    def __init__(self, cv2, np):
        """Initialize ShotDetector with CV2 and NumPy handles.
        
        Args:
            cv2 (module): OpenCV library handle
            np (module): NumPy library handle
        """
        self._next_track_id = 1
        self._tracks: Dict[int, BallTrack] = {}
        self._goal_openings: Dict[str, object] = {}
        self._goal_approaches: Dict[str, object] = {}
        self._last_event_time_by_robot: Dict[int, float] = {}
        self._ready = False

    def setup(self, tracker) -> None:
        """
        Initialize goal detection from field background image.
        
        Attempts to auto-detect goal openings (blue & red) via HSV color masking on the
        background image. If auto-detection fails, falls back to geometric assumptions
        based on field coordinates.
        
        Goal Opening Detection:
            - Masks background image by goal colors (blue HSV, red HSV ranges)
            - Extracts contours and filters by minimum area (900 px²)
            - Fits polygons to goal openings for shot entry detection
        
        Approach Zones:
            - Shrunk versions of goal opening polygons
            - Used for continuous goal entry confirmation
            - Prevents noise from bounces or partial entries
        
        Args:
            tracker (RobotTracker): Initialized tracker with background image and field mask
                - tracker._bg: Background image (h, w, 3) for color detection
                - tracker._field_mask: Binary field boundary mask
                - tracker._H_2d: Homography matrix for field coordinate transforms
        
        Side Effects:
            - Sets _ready = True if goal openings successfully detected
            - Populates _goal_openings, _goal_approaches dicts
            - Prints status messages to console
        """
            return
        openings = self._detect_goal_openings(tracker._bg, tracker._field_mask)
        if not openings:
            openings = self._fallback_goal_openings(tracker)
        if not openings:
            print("[WARN] Shot detector could not determine goal openings.")
            return
        self._goal_openings = openings
        self._goal_approaches = {
            color: self._build_goal_approach(poly)
            for color, poly in openings.items()
        }
        self._ready = True
        print("[INFO] Shot detector goal openings ready: {}".format(
            ", ".join(sorted(openings.keys()))))

    def update(self, frame_num: int, match_time_s: float, poses, tracker) -> List[ShotEvent]:
        """
        Perform one frame of ball tracking and shot detection.
        
        Per-frame processing:
        1. Extract ball positions from HSV color masks
        2. Associate detections into persistent ball tracks
        3. Update goal entry state for each active track
        4. Check for shot launch events
        5. Finalize and emit shot events when tracks are lost
        
        Ball Segmentation:
            - Uses tracker._last_ball_mask (computed during tracking)
            - Mask created from HSV ranges for green (alliance) and purple (opponent) balls
            - Extracts contours and computes centroids for each ball
        
        Track Association:
            - Matches detections to existing tracks via nearest-neighbor distance
            - Creates new tracks for unmatched detections near robots
            - Removes lost tracks after BALL_MAX_MISSING (12) frames
        
        Shot Launch Detection:
            - Triggered when ball velocity exceeds SHOT_MIN_SPEED_PX (2 px/frame)
            - Requires minimum upward component: SHOT_MIN_UPWARD_PX (3 pixels)
            - Requires acceleration (negative step ratio): 35% speed increase
            - Must originate near a robot position (within SHOT_MAX_START_DIST_PX)
        
        Goal Entry Confirmation:
            - Ball enters goal approach zone → sets entered_goal flag
            - Confirms with minimum consecutive frames in zone (GOAL_CONFIRM_MIN_FRAMES=2)
            - Deduplicates by storing last event time per robot
        
        Args:
            frame_num (int): Current frame index
            match_time_s (float): Elapsed match time in seconds
            poses (List[RobotPose]): Current robot poses from tracker
            tracker (RobotTracker): Tracker instance with:
                - _last_ball_mask: Ball detection mask for current frame
                - _H_inv: Inverse homography for pixel ↔ field transforms
                - _pos_px: Robot image positions
        
        Returns:
            List[ShotEvent]: Shot events that were resolved this frame
                Each ShotEvent contains: shooter_id, x_in, y_in, made (bool), timestamp_s
        
        Side Effects:
            - Updates _tracks: active ball track states
            - Updates _last_event_time_by_robot: deduplication tracking
            - Removes tracks from _tracks when resolved
        """
            return []

        detections = self._extract_ball_detections(tracker._last_ball_mask)
        robot_refs = self._robot_refs(poses, tracker)
        self._associate_tracks(detections, robot_refs, frame_num)

        events = []
        for track in list(self._tracks.values()):
            if not track.samples or track.resolved:
                continue
            self._update_goal_state(track)
            self._maybe_mark_launched(track, robot_refs)
            if track.missing_frames > self.BALL_MAX_MISSING:
                event = self._resolve_track(track, frame_num, match_time_s)
                if event is not None and not self._is_duplicate_event(event):
                    events.append(event)
                del self._tracks[track.track_id]
        return events

    def draw_debug(self, dbg) -> None:
        if not self._ready:
            return
        cv2 = self.cv2
        for color, poly in self._goal_approaches.items():
            cv_color = (255, 180, 60) if color == "blue" else (80, 180, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 1, cv2.LINE_AA)
        for color, poly in self._goal_openings.items():
            cv_color = (255, 80, 0) if color == "blue" else (0, 80, 255)
            cv2.polylines(dbg, [poly], True, cv_color, 2, cv2.LINE_AA)
        for track in self._tracks.values():
            pts = [(int(round(x)), int(round(y))) for _f, x, y in track.samples[-8:]]
            if len(pts) >= 2:
                cv2.polylines(
                    dbg,
                    [self.np.array(pts, dtype=self.np.int32).reshape(-1, 1, 2)],
                    False,
                    (0, 255, 255) if track.launched else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
            if pts:
                px, py = pts[-1]
                label = "S{}".format(track.shooter_id) if track.launched and track.shooter_id is not None else "b"
                if track.entered_goal:
                    label += ":goal"
                cv2.putText(dbg, label, (px + 4, py - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def _extract_ball_detections(self, ball_mask):
        cv2 = self.cv2
        cnts, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in cnts:
            area = float(cv2.contourArea(c))
            if area < self.BALL_MIN_CONTOUR_AREA_PX or area > self.BALL_MAX_CONTOUR_AREA_PX:
                continue
            M = cv2.moments(c)
            if M["m00"] <= 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
            detections.append((cx, cy, area))
        return detections

    def _robot_refs(self, poses, tracker):
        refs = []
        for rid, pose in enumerate(poses):
            if not pose.visible:
                continue
            pt = self.np.array([[[pose.x_in, pose.y_in]]], dtype=self.np.float32)
            ip = self.cv2.perspectiveTransform(pt, tracker._H_inv)[0][0]
            refs.append({
                "id": rid,
                "img": (float(ip[0]), float(ip[1])),
                "center": (float(pose.x_center_in), float(pose.y_center_in)),
            })
        return refs

    def _associate_tracks(self, detections, robot_refs, frame_num: int) -> None:
        unmatched_tracks = set(self._tracks.keys())
        unmatched_dets = set(range(len(detections)))
        pairs = []
        for tid, track in self._tracks.items():
            pred_x, pred_y = self._predict_track_point(track)
            max_dist = self.BALL_ASSOC_DIST_FAST_PX if track.launched else self.BALL_ASSOC_DIST_PX
            for di, (cx, cy, _area) in enumerate(detections):
                d = math.hypot(cx - pred_x, cy - pred_y)
                if d <= max_dist:
                    pairs.append((d, tid, di))
        pairs.sort()
        for _d, tid, di in pairs:
            if tid not in unmatched_tracks or di not in unmatched_dets:
                continue
            cx, cy, _area = detections[di]
            track = self._tracks[tid]
            track.samples.append((frame_num, cx, cy))
            track.missing_frames = 0
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

        for tid in unmatched_tracks:
            self._tracks[tid].missing_frames += 1

        for di in unmatched_dets:
            cx, cy, _area = detections[di]
            nearest_id = None
            nearest_dist = None
            nearest_img = None
            nearest_center = None
            for ref in robot_refs:
                dist = math.hypot(cx - ref["img"][0], cy - ref["img"][1])
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = ref["id"]
                    nearest_img = ref["img"]
                    nearest_center = ref["center"]
            self._tracks[self._next_track_id] = BallTrack(
                track_id=self._next_track_id,
                samples=[(frame_num, cx, cy)],
                shooter_id=nearest_id,
                shooter_img=nearest_img,
                shooter_dist_px=nearest_dist,
                shooter_pos_center_in=nearest_center,
            )
            self._next_track_id += 1

    def _predict_track_point(self, track: BallTrack) -> Tuple[float, float]:
        if len(track.samples) < 2:
            _f, x, y = track.samples[-1]
            return x, y
        _f1, x1, y1 = track.samples[-1]
        _f0, x0, y0 = track.samples[-2]
        return x1 + (x1 - x0), y1 + (y1 - y0)

    def _update_goal_state(self, track: BallTrack) -> None:
        frame_idx, x, y = track.samples[-1]
        for color, poly in self._goal_approaches.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.approached_goal = True
                if not track.goal_color:
                    track.goal_color = color
        for color, poly in self._goal_openings.items():
            if self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0:
                track.entered_goal = True
                track.entered_goal_frames += 1
                if track.first_goal_entry_frame is None:
                    track.first_goal_entry_frame = frame_idx
                track.last_goal_entry_point = (float(x), float(y))
                track.goal_color = color
                break

    def _maybe_mark_launched(self, track: BallTrack, robot_refs) -> None:
        if track.launched or len(track.samples) < self.SHOT_MIN_LIFE_FRAMES:
            return
        first = track.first_point
        last = track.last_point
        if first is None or last is None:
            return
        shooter_ref, shooter_score = self._select_shooter(track, robot_refs)
        if shooter_ref is None or shooter_score > self.SHOT_MAX_START_DIST_PX:
            return
        f0, x0, y0 = first
        f1, x1, y1 = last
        dt = max(f1 - f0, 1)
        disp = math.hypot(x1 - x0, y1 - y0)
        avg_speed = disp / dt
        if disp < self.SHOT_MIN_DISP_PX or avg_speed < self.SHOT_MIN_SPEED_PX:
            return
        if (y1 - y0) > -self.SHOT_MIN_UPWARD_PX:
            return
        if not self._has_consistent_upward_motion(track):
            return
        if not self._moving_away_from_shooter(track):
            return
        track.shooter_id = shooter_ref["id"]
        track.shooter_img = shooter_ref["img"]
        track.shooter_dist_px = shooter_score
        track.shooter_pos_center_in = shooter_ref["center"]
        track.launched = True

    def _resolve_track(self, track: BallTrack, frame_num: int, match_time_s: float) -> Optional[ShotEvent]:
        if not track.launched or track.shooter_id is None or track.shooter_pos_center_in is None:
            return None
        if self._confirmed_goal_make(track):
            result = "made"
        elif track.approached_goal or self._track_reached_top(track):
            result = "missed"
        else:
            return None
        sx, sy = track.shooter_pos_center_in
        track.resolved = True
        return ShotEvent(
            shooter_id=track.shooter_id,
            result=result,
            shot_x_in=float(sx),
            shot_y_in=float(sy),
            frame_num=frame_num,
            timestamp_s=match_time_s,
            goal_color=track.goal_color,
        )

    def _has_consistent_upward_motion(self, track: BallTrack) -> bool:
        if len(track.samples) < 3:
            return False
        negative_steps = 0
        total_steps = 0
        for (_f0, _x0, y0), (_f1, _x1, y1) in zip(track.samples[:-1], track.samples[1:]):
            total_steps += 1
            if y1 < y0:
                negative_steps += 1
        return total_steps > 0 and (negative_steps / total_steps) >= self.SHOT_MIN_NEGATIVE_STEP_RATIO

    def _select_shooter(self, track: BallTrack, robot_refs):
        if not robot_refs:
            return None, float("inf")
        first = track.first_point
        if first is None:
            return None, float("inf")
        _f0, x0, y0 = first
        origin_x, origin_y = x0, y0
        if len(track.samples) >= 2:
            deltas = []
            for (_fa, xa, ya), (_fb, xb, yb) in zip(track.samples[:-1], track.samples[1:]):
                deltas.append((xb - xa, yb - ya))
                if len(deltas) >= 3:
                    break
            if deltas:
                avg_dx = sum(dx for dx, _dy in deltas) / len(deltas)
                avg_dy = sum(dy for _dx, dy in deltas) / len(deltas)
                origin_x = x0 - 1.35 * avg_dx
                origin_y = y0 - 1.35 * avg_dy

        best_ref = None
        best_score = float("inf")
        for ref in robot_refs:
            rx, ry = ref["img"]
            dist_origin = math.hypot(origin_x - rx, origin_y - ry)
            dist_first = math.hypot(x0 - rx, y0 - ry)
            above_penalty = max(0.0, (y0 - ry) - 6.0) * 0.75
            score = 0.7 * dist_origin + 0.3 * dist_first + above_penalty
            if score < best_score:
                best_score = score
                best_ref = ref
        return best_ref, best_score

    def _moving_away_from_shooter(self, track: BallTrack) -> bool:
        if track.shooter_img is None or len(track.samples) < 2:
            return False
        first = track.first_point
        last = track.last_point
        if first is None or last is None:
            return False
        sx, sy = track.shooter_img
        _f0, x0, y0 = first
        _f1, x1, y1 = last
        start_dist = math.hypot(x0 - sx, y0 - sy)
        last_dist = math.hypot(x1 - sx, y1 - sy)
        return last_dist >= start_dist + self.SHOT_MIN_AWAY_FROM_SHOOTER_PX

    def _confirmed_goal_make(self, track: BallTrack) -> bool:
        if not track.entered_goal:
            return False
        if track.entered_goal_frames >= self.GOAL_CONFIRM_MIN_FRAMES:
            return True
        if track.first_goal_entry_frame is None:
            return False
        if track.missing_frames <= self.GOAL_ENTRY_MAX_MISSING and self._track_finished_inside_goal(track):
            return True
        return False

    def _track_finished_inside_goal(self, track: BallTrack) -> bool:
        if not track.goal_color or track.last_goal_entry_point is None:
            return False
        poly = self._goal_openings.get(track.goal_color)
        if poly is None:
            return False
        x, y = track.last_goal_entry_point
        return self.cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def _is_duplicate_event(self, event: ShotEvent) -> bool:
        last_time = self._last_event_time_by_robot.get(event.shooter_id)
        if last_time is not None and (event.timestamp_s - last_time) < self.SHOT_DEDUP_TIME_S:
            return True
        self._last_event_time_by_robot[event.shooter_id] = event.timestamp_s
        return False

    def _track_reached_top(self, track: BallTrack) -> bool:
        if not track.samples:
            return False
        top_y = min(y for _f, _x, y in track.samples)
        goal_top = min(
            min(poly[:, 0, 1]) for poly in self._goal_approaches.values()
        ) if self._goal_approaches else 0.0
        goal_bottom = max(
            max(poly[:, 0, 1]) for poly in self._goal_approaches.values()
        ) if self._goal_approaches else 0.0
        return top_y <= goal_bottom + max(14.0, 0.2 * max(goal_bottom - goal_top, 1.0))

    def _detect_goal_openings(self, background, field_mask):
        cv2, np = self.cv2, self.np
        hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        h, w = background.shape[:2]
        top_band = np.zeros((h, w), dtype=np.uint8)
        top_band[:max(1, int(round(h * 0.55))), :] = 255

        masks = {
            "blue": cv2.inRange(
                hsv,
                np.array(self.GOAL_BLUE_HSV_LO, dtype=np.uint8),
                np.array(self.GOAL_BLUE_HSV_HI, dtype=np.uint8),
            ),
            "red": cv2.bitwise_or(
                cv2.inRange(
                    hsv,
                    np.array(self.GOAL_RED1_HSV_LO, dtype=np.uint8),
                    np.array(self.GOAL_RED1_HSV_HI, dtype=np.uint8),
                ),
                cv2.inRange(
                    hsv,
                    np.array(self.GOAL_RED2_HSV_LO, dtype=np.uint8),
                    np.array(self.GOAL_RED2_HSV_HI, dtype=np.uint8),
                ),
            ),
        }

        openings = {}
        for color, mask in masks.items():
            mask = cv2.bitwise_and(mask, top_band)
            if color == "blue":
                mask[:, w // 2:] = 0
            else:
                mask[:, :w // 2] = 0
            kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            chosen = None
            for c in cnts:
                area = float(cv2.contourArea(c))
                if area < self.GOAL_MIN_AREA_PX:
                    continue
                x, y, ww, hh = cv2.boundingRect(c)
                if y > int(0.22 * h):
                    continue
                chosen = c
                break
            if chosen is None:
                continue
            x, y, ww, hh = cv2.boundingRect(chosen)
            pts = chosen.reshape(-1, 2)
            top_pts = pts[pts[:, 1] <= y + hh * self.GOAL_OPENING_TOP_FRAC]
            if len(top_pts) >= 3:
                hull = cv2.convexHull(top_pts.reshape(-1, 1, 2).astype(np.int32))
            else:
                hull = cv2.convexHull(chosen)
            openings[color] = self._shrink_polygon(
                hull,
                shrink_x=self.GOAL_OPENING_SHRINK_X,
                shrink_y=self.GOAL_OPENING_SHRINK_Y,
            )
        return openings

    def _fallback_goal_openings(self, tracker):
        if tracker._H_inv is None:
            return {}
        cv2, np = self.cv2, self.np
        field_points = {
            "blue": [(-8.0, -72.0), (-8.0, -40.0), (-36.0, -56.0)],
            "red": [(8.0, -72.0), (36.0, -56.0), (8.0, -40.0)],
        }
        openings = {}
        for color, pts_center in field_points.items():
            pts_corner = [_field_center_to_corner_xy(x, y) for x, y in pts_center]
            arr = np.array([pts_corner], dtype=np.float32)
            img = cv2.perspectiveTransform(arr, tracker._H_inv)[0]
            openings[color] = np.round(img).astype(np.int32).reshape(-1, 1, 2)
        return openings

    def _build_goal_approach(self, poly):
        pts = poly.reshape(-1, 2).astype(self.np.float32)
        x_min = float(self.np.min(pts[:, 0]))
        x_max = float(self.np.max(pts[:, 0]))
        y_min = float(self.np.min(pts[:, 1]))
        y_max = float(self.np.max(pts[:, 1]))
        w = max(x_max - x_min, 1.0)
        h = max(y_max - y_min, 1.0)
        pad_x = w * self.GOAL_APPROACH_SIDE_PAD_FRAC
        down = h * self.GOAL_APPROACH_DOWN_FRAC
        rect = self.np.array([
            [x_min - pad_x, y_min],
            [x_max + pad_x, y_min],
            [x_max + pad_x, y_max + down],
            [x_min - pad_x, y_max + down],
        ], dtype=self.np.int32)
        return rect.reshape(-1, 1, 2)

    def _shrink_polygon(self, poly, shrink_x: float, shrink_y: float):
        pts = poly.reshape(-1, 2).astype(self.np.float32)
        cx = float(self.np.mean(pts[:, 0]))
        cy = float(self.np.mean(pts[:, 1]))
        scaled = pts.copy()
        scaled[:, 0] = cx + (scaled[:, 0] - cx) * shrink_x
        scaled[:, 1] = cy + (scaled[:, 1] - cy) * shrink_y
        return self.np.round(scaled).astype(self.np.int32).reshape(-1, 1, 2)
