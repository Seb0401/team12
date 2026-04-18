"""Compute joint angles from YOLO detection positions."""

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

from solution.detection.detector import FrameDetections

_MAX_INTERPOLATION_GAP = 5


@dataclass
class JointAngles:
    """Computed angles between detected shovel components."""

    frame_idx: int
    arm_angle_deg: Optional[float] = None
    bucket_angle_deg: Optional[float] = None
    boom_angle_deg: Optional[float] = None
    valid: bool = False
    interpolated: bool = False


_last_valid_angles: Optional["JointAngles"] = None
_frames_since_valid: int = 0


def _angle_between_points(
    p1: tuple[float, float],
    vertex: tuple[float, float],
    p2: tuple[float, float],
) -> float:
    """
    Compute angle at vertex formed by p1-vertex-p2, in degrees.

    Uses atan2 for full 360° range, returns 0–180°.

    @param p1 - First endpoint
    @param vertex - Vertex point (angle measured here)
    @param p2 - Second endpoint
    @returns Angle in degrees [0, 180]
    """
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0

    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _angle_from_vertical(
    point: tuple[float, float],
    reference: tuple[float, float],
) -> float:
    """
    Compute angle of the vector (reference→point) from vertical (downward).

    @param point - Target point
    @param reference - Reference/pivot point
    @returns Angle from vertical in degrees [0, 360)
    """
    dx = point[0] - reference[0]
    dy = point[1] - reference[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def compute_joint_angles(detections: FrameDetections) -> JointAngles:
    """
    Compute shovel joint angles from detection bounding box centers.

    Falls back to the last valid angles when detections are missing
    for up to _MAX_INTERPOLATION_GAP consecutive frames.

    @param detections - Frame detections with bucket, arm_joint, boom
    @returns JointAngles with computed values (valid=True if all 3 detected)
    """
    global _last_valid_angles, _frames_since_valid

    bucket_det = detections.get_by_class("bucket")
    arm_det = detections.get_by_class("arm_joint")
    boom_det = detections.get_by_class("boom")

    angles = JointAngles(frame_idx=detections.frame_idx)

    if not all([bucket_det, arm_det, boom_det]):
        _frames_since_valid += 1
        if _last_valid_angles is not None and _frames_since_valid <= _MAX_INTERPOLATION_GAP:
            angles.arm_angle_deg = _last_valid_angles.arm_angle_deg
            angles.bucket_angle_deg = _last_valid_angles.bucket_angle_deg
            angles.boom_angle_deg = _last_valid_angles.boom_angle_deg
            angles.valid = True
            angles.interpolated = True
        return angles

    p_bucket = bucket_det.center
    p_arm = arm_det.center
    p_boom = boom_det.center

    angles.arm_angle_deg = _angle_between_points(p_bucket, p_arm, p_boom)
    angles.bucket_angle_deg = _angle_from_vertical(p_bucket, p_arm)
    angles.boom_angle_deg = _angle_from_vertical(p_boom, p_arm)
    angles.valid = True

    _last_valid_angles = angles
    _frames_since_valid = 0

    return angles


def reset_interpolation_state() -> None:
    """Reset the interpolation state between pipeline runs."""
    global _last_valid_angles, _frames_since_valid
    _last_valid_angles = None
    _frames_since_valid = 0
