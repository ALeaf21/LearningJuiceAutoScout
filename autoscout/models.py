import math
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple

from autoscout.geometry import FIELD_CENTER_OFFSET_IN, _normalize_angle_rad


@dataclass
class RobotPose:
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
    shooter_id: int
    result: str
    shot_x_in: float
    shot_y_in: float
    frame_num: int
    timestamp_s: float
    goal_color: str = ""


@dataclass
class BallTrack:
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
