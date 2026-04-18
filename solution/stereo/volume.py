"""Bucket fill volume estimation from stereo depth maps."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from solution.detection.detector import Detection

logger = logging.getLogger(__name__)


def estimate_bucket_fill(
    depth_map: np.ndarray,
    bucket_detection: Optional[Detection],
    empty_bucket_depth: Optional[float] = None,
    pixel_to_meter: float = 0.002,
) -> Optional[float]:
    """
    Estimate material volume inside the bucket from depth map.

    Compares depth within bucket bbox against a reference (empty bucket)
    to estimate fill volume.

    Volume = Σ (reference_depth - actual_depth) × pixel_area_m²
    for all pixels where actual_depth < reference_depth (material present).

    @param depth_map - Metric depth map (H, W) in meters
    @param bucket_detection - YOLO detection of bucket bbox
    @param empty_bucket_depth - Reference depth for empty bucket (auto-estimated if None)
    @param pixel_to_meter - Spatial resolution in meters/pixel
    @returns Estimated fill volume in m³, or None if estimation fails
    """
    if bucket_detection is None:
        return None

    x1 = max(0, int(bucket_detection.x1))
    y1 = max(0, int(bucket_detection.y1))
    x2 = min(depth_map.shape[1], int(bucket_detection.x2))
    y2 = min(depth_map.shape[0], int(bucket_detection.y2))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth_map[y1:y2, x1:x2]

    valid_mask = np.isfinite(roi) & (roi > 0) & (roi < 50.0)
    valid_depths = roi[valid_mask]

    if len(valid_depths) < 10:
        return None

    if empty_bucket_depth is None:
        empty_bucket_depth = np.percentile(valid_depths, 95)

    fill_mask = valid_depths < (empty_bucket_depth * 0.95)
    fill_depths = valid_depths[fill_mask]

    if len(fill_depths) == 0:
        return 0.0

    depth_diff = empty_bucket_depth - fill_depths
    pixel_area_m2 = pixel_to_meter ** 2
    volume_m3 = float(np.sum(depth_diff) * pixel_area_m2)

    return max(0.0, volume_m3)


def compute_fill_percentage(
    estimated_volume_m3: float,
    bucket_capacity_m3: float,
) -> float:
    """
    Compute bucket fill as percentage of nominal capacity.

    @param estimated_volume_m3 - Estimated fill volume
    @param bucket_capacity_m3 - Nominal bucket capacity (29 m³ for EX-5600)
    @returns Fill percentage [0, 100+] (can exceed 100 if heaped)
    """
    if bucket_capacity_m3 <= 0:
        return 0.0
    return (estimated_volume_m3 / bucket_capacity_m3) * 100.0
