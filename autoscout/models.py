"""Data models for robot pose tracking and shot detection.

This module defines the core dataclasses used throughout the AutoScout pipeline:
- RobotPose: Robot position, orientation, and visibility in a single frame
- MergeGroup: State management for multiple robots sharing a foreground blob
- BallTrack: Ball trajectory and shot association
- ShotEvent: Resolved shot result (made/missed)

Coordinate Systems:
- Internal (corner-origin): (0,0) to (144,144) inches, used by tracker
- Public API (center-origin): (-72,-72) to (72,72) inches, used in outputs
- WPILog: Swapped axes, meters, robotics heading convention
"""

import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple

from autoscout.geometry import FIELD_CENTER_OFFSET_IN, _normalize_angle_rad


@dataclass
class RobotPose:
    """Represents a robot's pose (position, orientation, visibility) at a single frame.
    
    Stores position in internal corner-origin coordinate system (0-144 inches).
    Provides computed properties for conversion to center-origin, meters, and WPILog formats.
    
    Attributes:
        x_in (float): Field X coordinate in inches (corner-origin, default=72)
        y_in (float): Field Y coordinate in inches (corner-origin, default=72)
        heading (float): Robot heading/rotation in radians (default=0.0)
        visible (bool): Track visibility in current frame (default=False)
    
    Properties:
        x_center_in, y_center_in: Center-origin field coordinates (subtract 72)
        x_m, y_m: Meters (center-origin, multiply by 0.0254)
        wpilog_x_m, wpilog_y_m: WPILog format (axes swapped, center-origin)
        wpilog_heading_rad: WPILog heading (π/2 - heading, radians)
    
    Example:
        >>> pose = RobotPose(x_in=100, y_in=80, heading=0.785, visible=True)
        >>> pose.x_center_in  # 28.0 (100 - 72)
        >>> pose.x_m          # 0.7112 (28 * 0.0254)
    """
    x_in:    float = FIELD_CENTER_OFFSET_IN
    y_in:    float = FIELD_CENTER_OFFSET_IN
    heading: float = 0.0
    visible: bool  = False

    @property
    def x_center_in(self): return self.x_in - FIELD_CENTER_OFFSET_IN

    @property
    def y_center_in(self): return self.y_in - FIELD_CENTER_OFFSET_IN

    @property
    def x_m(self): return self.x_center_in * 0.0254

    @property
    def y_m(self): return self.y_center_in * 0.0254

    @property
    def wpilog_x_m(self): return self.y_center_in * 0.0254

    @property
    def wpilog_y_m(self): return self.x_center_in * 0.0254

    @property
    def wpilog_heading_rad(self): return _normalize_angle_rad((math.pi / 2.0) - self.heading)


# ─────────────────────────────────────────────────────────────────────────────
# MergeGroup  (v3.2: adds entry_order / current_order for permutation tracking)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class MergeGroup:
    """
    State for 2 (or more) tracks sharing a single foreground blob.

    track_ids       : tracks in the group (unordered set, kept as list).
    entry_axis      : unit vector (ax, ay) in IMAGE pixel space pointing from
                      the "low" end to the "high" end at merge time.
    parent_id       : contour parent_id of the current merged blob.

    -- 2-robot fields (unchanged from v3.1) --
    crossed         : True if the two robots have swapped sides of entry_axis.

    -- 3+ robot fields (new in v3.2) --
    entry_order     : track IDs sorted by projection onto entry_axis at the
                      moment the merge was created.  Immutable after creation.
                      entry_order[0] is the robot that was most "negative"
                      along entry_axis at merge start.
    current_order   : track IDs sorted by projection of their most-recent
                      dist-transform peak onto entry_axis.  Updated every
                      frame.  At separation, entry_order[k]→current_order[k]
                      gives the permutation to apply.
    peak_assignment : {track_id: (px, py)} — current dist-transform peak per
                      track.  Used both for live position updates (fix e) and
                      for re-anchoring on separation (fix h).
    """
    track_ids      : List[int]             = dc_field(default_factory=list)
    entry_axis     : Tuple[float, float]   = (1.0, 0.0)
    parent_id      : int                   = -1
    # 2-robot
    crossed        : bool                  = False
    # 3+ robot (v3.2)
    entry_order    : List[int]             = dc_field(default_factory=list)
    current_order  : List[int]             = dc_field(default_factory=list)
    peak_assignment: dict                  = dc_field(default_factory=dict)
    order_votes    : dict                  = dc_field(default_factory=dict)
    entry_features : dict                  = dc_field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Shot detection
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ShotEvent:
    """Represents a resolved shot event (made or missed goal attempt).
    
    Created when a ball track is finalized and either enters the goal region
    or is lost without goal entry. Exported to CSV/JLOG output files.
    
    Attributes:
        shooter_id (int): Robot ID (0-3) that launched the shot
        result (str): Shot outcome - "made" or "missed"
        shot_x_in (float): Ball X coordinate at launch (inches, center-origin)
        shot_y_in (float): Ball Y coordinate at launch (inches, center-origin)
        frame_num (int): Frame number when shot was launched
        timestamp_s (float): Match time in seconds when shot was launched
        goal_color (str): Target goal color - "blue" or "red" (default="")
    
    Note:
        Deduplication: Events from same robot within 0.35s are suppressed
        to avoid duplicate detection of bounces/rebounds.
    """
    shooter_id: int
    result: str
    shot_x_in: float
    shot_y_in: float
    frame_num: int
    timestamp_s: float
    goal_color: str = ""


@dataclass
class BallTrack:
    """Tracks a ball's trajectory and associates it with shot events.
    
    Maintains centroid history, handles temporary occlusions (missing frames),
    detects launch velocity/angle, associates with shooter robot, and tracks
    goal entry/exit. Finalized when ball is missing too long or goal entry
    is confirmed.
    
    Attributes:
        track_id (int): Unique identifier for this ball track
        samples (List[(frame, x, y)]): Ball centroid positions over time (pixel space)
        missing_frames (int): Consecutive frames without detection
        launched (bool): Has this shot been launched (velocity threshold met)?
        shooter_id (Optional[int]): Robot ID (0-3) if launched, else None
        shooter_img (Optional[(px, py)]): Robot image position at launch
        shooter_dist_px (Optional[float]): Distance from shooter at launch (pixels)
        shooter_pos_center_in (Optional[(x, y)]): Shooter field position at launch
        goal_color (str): "blue" or "red" for target goal
        entered_goal (bool): Did ball enter goal region?
        entered_goal_frames (int): Consecutive frames inside goal
        first_goal_entry_frame (Optional[int]): Frame number of first entry
        last_goal_entry_point (Optional[(px, py)]): Last position in goal (pixels)
        approached_goal (bool): Did ball approach goal region?
        resolved (bool): Has this track been finalized?
    
    Properties:
        last_point: Most recent sample (frame, x, y) or None
        first_point: Oldest sample (frame, x, y) or None
    
    Lifecycle:
        1. Created on ball detection (unassociated)
        2. Associated to track via nearest-neighbor search
        3. Marked as launched when velocity thresholds met
        4. Goal entry tracked as ball approaches/enters goal region
        5. Resolved when missing >12 frames or match ends
    """
    track_id: int
    samples: List[Tuple[int, float, float]] = dc_field(default_factory=list)
    missing_frames: int = 0
    launched: bool = False
    shooter_id: Optional[int] = None
    shooter_img: Optional[Tuple[float, float]] = None
    shooter_dist_px: Optional[float] = None
    shooter_pos_center_in: Optional[Tuple[float, float]] = None
    goal_color: str = ""
    entered_goal: bool = False
    entered_goal_frames: int = 0
    first_goal_entry_frame: Optional[int] = None
    last_goal_entry_point: Optional[Tuple[float, float]] = None
    approached_goal: bool = False
    resolved: bool = False

    @property
    def last_point(self) -> Optional[Tuple[int, float, float]]:
        return self.samples[-1] if self.samples else None

    @property
    def first_point(self) -> Optional[Tuple[int, float, float]]:
        return self.samples[0] if self.samples else None
