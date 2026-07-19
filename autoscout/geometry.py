"""Coordinate system transformations and geometry utilities.

Provides conversions between:
- Internal (corner-origin): (0,0) to (144,144) inches - used by tracker
- Public API (center-origin): (-72,-72) to (72,72) inches - used in outputs
- Angle normalization for heading consistency

The tracker uses internal corner-origin coordinates for homography calculations.
All external APIs (CSV, JLOG, manual inputs) use center-origin coordinates.

Key Constants:
    FIELD_SIZE_IN: 144 inches (FTC field dimension)
    FIELD_CENTER_OFFSET_IN: 72 inches (offset for centering)
"""

import math
from typing import Tuple


FIELD_SIZE_IN = 144.0
FIELD_CENTER_OFFSET_IN = FIELD_SIZE_IN / 2.0


def _field_corner_to_center_xy(x_in: float, y_in: float) -> Tuple[float, float]:
    """Convert field coordinates from corner-origin to center-origin.
    
    Internal tracker uses corner-origin (0,0 at top-left) for perspective transform.
    External APIs use center-origin (0,0 at field center) for intuitive positioning.
    
    Args:
        x_in (float): X coordinate in corner-origin system (0-144 inches)
        y_in (float): Y coordinate in corner-origin system (0-144 inches)
    
    Returns:
        Tuple[float, float]: (x_center, y_center) in center-origin system (-72 to 72)
    
    Example:
        >>> _field_corner_to_center_xy(72, 100)  # (0, 28)
    """
    return float(x_in) - FIELD_CENTER_OFFSET_IN, float(y_in) - FIELD_CENTER_OFFSET_IN


def _field_center_to_corner_xy(x_in: float, y_in: float) -> Tuple[float, float]:
    """Convert field coordinates from center-origin to corner-origin.
    
    Inverse of _field_corner_to_center_xy. Used when processing user inputs
    (--robot-init-positions, manual CSVs) before passing to tracker.
    
    Args:
        x_in (float): X coordinate in center-origin system (-72 to 72 inches)
        y_in (float): Y coordinate in center-origin system (-72 to 72 inches)
    
    Returns:
        Tuple[float, float]: (x_corner, y_corner) in corner-origin system (0-144)
    
    Example:
        >>> _field_center_to_corner_xy(0, 28)  # (72, 100)
    """
    return float(x_in) + FIELD_CENTER_OFFSET_IN, float(y_in) + FIELD_CENTER_OFFSET_IN


def _normalize_angle_rad(angle: float) -> float:
    """Normalize angle to (-π, π] range using wrapping.
    
    Ensures consistent angle representation by adding/subtracting 2π as needed.
    Used for robot heading to prevent angle drift and enable simple comparisons.
    
    Args:
        angle (float): Angle in radians
    
    Returns:
        float: Normalized angle in (-π, π] range
    
    Example:
        >>> _normalize_angle_rad(3.5 * math.pi)  # ~-0.64 (0.5π - 2π)
    """
    while angle <= -math.pi:
        angle += math.tau
    while angle > math.pi:
        angle -= math.tau
    return angle
