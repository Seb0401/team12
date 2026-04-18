"""3D joint position estimation via stereo triangulation."""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np

from solution.detection.detector import FrameDetections, Detection

logger = logging.getLogger(__name__)


@dataclass
class Joint3D:
    """3D position of a detected joint in camera coordinates (meters)."""

    class_name: str
    x: float
    y: float
    z: float


@dataclass
class Frame3DKinematics:
    """3D joint positions for a single stereo frame pair."""

    frame_idx: int
    joints: List[Joint3D]

    def get_joint(self, class_name: str) -> Optional[Joint3D]:
        for j in self.joints:
            if j.class_name == class_name:
                return j
        return None


def triangulate_point(
    uv_left: Tuple[float, float],
    uv_right: Tuple[float, float],
    focal_length_px: float,
    baseline_m: float,
    cx: float,
    cy: float,
) -> Tuple[float, float, float]:
    """
    Triangulate a 3D point from stereo correspondences.

    X = baseline × (u_left - cx) / disparity
    Y = baseline × (v_left - cy) / disparity
    Z = baseline × focal_length / disparity

    @param uv_left - (u, v) pixel coords in left image
    @param uv_right - (u, v) pixel coords in right image
    @param focal_length_px - Focal length in pixels
    @param baseline_m - Stereo baseline in meters
    @param cx - Principal point x (usually width/2)
    @param cy - Principal point y (usually height/2)
    @returns (X, Y, Z) in meters, camera coordinate frame
    """
    disparity = uv_left[0] - uv_right[0]
    if abs(disparity) < 1.0:
        return (0.0, 0.0, float("inf"))

    z = (focal_length_px * baseline_m) / disparity
    x = baseline_m * (uv_left[0] - cx) / disparity
    y = baseline_m * (uv_left[1] - cy) / disparity

    return (x, y, z)


def estimate_3d_kinematics(
    left_detections: FrameDetections,
    right_detections: FrameDetections,
    focal_length_px: float = 700.0,
    baseline_m: float = 0.12,
    image_width: int = 1280,
    image_height: int = 720,
) -> Frame3DKinematics:
    """
    Compute 3D positions of all detected joints using stereo triangulation.

    Matches detections by class name between left and right frames.

    @param left_detections - Detections from left camera
    @param right_detections - Detections from right camera
    @param focal_length_px - Estimated focal length
    @param baseline_m - Stereo baseline
    @param image_width - Image width for principal point
    @param image_height - Image height for principal point
    @returns Frame3DKinematics with triangulated joint positions
    """
    cx = image_width / 2.0
    cy = image_height / 2.0
    joints: List[Joint3D] = []

    for class_name in ["bucket", "arm_joint", "boom", "truck"]:
        left_det = left_detections.get_by_class(class_name)
        right_det = right_detections.get_by_class(class_name)

        if left_det is None or right_det is None:
            continue

        x, y, z = triangulate_point(
            left_det.center,
            right_det.center,
            focal_length_px,
            baseline_m,
            cx,
            cy,
        )

        if np.isfinite(z) and z > 0:
            joints.append(Joint3D(class_name=class_name, x=x, y=y, z=z))

    return Frame3DKinematics(
        frame_idx=left_detections.frame_idx,
        joints=joints,
    )
