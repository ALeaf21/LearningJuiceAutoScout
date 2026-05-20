import math
from typing import Tuple


FIELD_SIZE_IN = 144.0
FIELD_CENTER_OFFSET_IN = FIELD_SIZE_IN / 2.0


def _field_corner_to_center_xy(x_in: float, y_in: float) -> Tuple[float, float]:
    return float(x_in) - FIELD_CENTER_OFFSET_IN, float(y_in) - FIELD_CENTER_OFFSET_IN


def _field_center_to_corner_xy(x_in: float, y_in: float) -> Tuple[float, float]:
    return float(x_in) + FIELD_CENTER_OFFSET_IN, float(y_in) + FIELD_CENTER_OFFSET_IN


def _normalize_angle_rad(angle: float) -> float:
    while angle <= -math.pi:
        angle += math.tau
    while angle > math.pi:
        angle -= math.tau
    return angle
