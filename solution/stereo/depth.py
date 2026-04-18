"""Stereo disparity and depth map computation using StereoSGBM."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from solution.config import (
    STEREO_BLOCK_SIZE,
    STEREO_MIN_DISPARITY,
    STEREO_NUM_DISPARITIES,
)

logger = logging.getLogger(__name__)


class StereoDepthEstimator:
    """
    Compute depth maps from rectified stereo image pairs using StereoSGBM.

    Assumes cameras are pre-rectified (common for industrial stereo rigs).
    """

    def __init__(
        self,
        num_disparities: int = STEREO_NUM_DISPARITIES,
        block_size: int = STEREO_BLOCK_SIZE,
        min_disparity: int = STEREO_MIN_DISPARITY,
    ) -> None:
        self._stereo = cv2.StereoSGBM_create(
            minDisparity=min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

    def compute_disparity(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Compute disparity map from a stereo pair.

        @param left_frame - Left BGR image
        @param right_frame - Right BGR image
        @returns Disparity map as float32 (higher = closer)
        """
        left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

        disparity = self._stereo.compute(left_gray, right_gray)
        # StereoSGBM returns disparity * 16 as int16
        return disparity.astype(np.float32) / 16.0

    def disparity_to_depth(
        self,
        disparity: np.ndarray,
        focal_length_px: float,
        baseline_m: float,
    ) -> np.ndarray:
        """
        Convert disparity map to metric depth.

        depth = (focal_length × baseline) / disparity

        @param disparity - Disparity map from compute_disparity
        @param focal_length_px - Camera focal length in pixels
        @param baseline_m - Stereo baseline distance in meters
        @returns Depth map in meters (inf where disparity ≤ 0)
        """
        safe_disp = np.where(disparity > 0, disparity, np.inf)
        return (focal_length_px * baseline_m) / safe_disp

    def compute_depth(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        focal_length_px: float = 700.0,
        baseline_m: float = 0.12,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full pipeline: stereo pair → (disparity, depth_meters).

        @param left_frame - Left BGR image
        @param right_frame - Right BGR image
        @param focal_length_px - Estimated focal length (default 700px for 1280w)
        @param baseline_m - Estimated baseline (default 0.12m)
        @returns Tuple of (disparity_map, depth_map_meters)
        """
        disparity = self.compute_disparity(left_frame, right_frame)
        depth = self.disparity_to_depth(disparity, focal_length_px, baseline_m)
        return disparity, depth
