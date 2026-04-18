"""Synchronization utilities for aligning video frames with IMU timestamps."""

import numpy as np


def imu_timestamp_to_seconds(timestamps: np.ndarray) -> np.ndarray:
    """
    Convert epoch nanosecond timestamps to relative seconds from start.

    @param timestamps - Array of epoch timestamps in nanoseconds
    @returns Array of relative time in seconds (starts at 0.0)
    """
    if len(timestamps) == 0:
        return np.array([], dtype=np.float64)
    return (timestamps - timestamps[0]) / 1e9


def frame_index_to_seconds(frame_idx: int, fps: float) -> float:
    """
    Convert a video frame index to time in seconds.

    @param frame_idx - Zero-based frame index
    @param fps - Video frames per second
    @returns Time in seconds
    """
    return frame_idx / fps


def seconds_to_frame_index(seconds: float, fps: float) -> int:
    """
    Convert a time in seconds to the nearest video frame index.

    @param seconds - Time in seconds
    @param fps - Video frames per second
    @returns Nearest frame index (clamped to >= 0)
    """
    return max(0, round(seconds * fps))


def align_imu_to_video(
    imu_timestamps: np.ndarray,
    video_fps: float,
    total_frames: int,
) -> np.ndarray:
    """
    Build a mapping from each video frame to the closest IMU sample index.

    Uses np.searchsorted for O(N log M) alignment.

    @param imu_timestamps - Raw epoch nanosecond timestamps from IMU col 0
    @param video_fps - Video FPS
    @param total_frames - Total number of video frames
    @returns Array of shape (total_frames,) with IMU sample indices
    """
    if len(imu_timestamps) == 0 or total_frames == 0:
        return np.array([], dtype=np.int64)

    imu_sec = imu_timestamp_to_seconds(imu_timestamps)
    frame_times = np.arange(total_frames) / video_fps

    indices = np.searchsorted(imu_sec, frame_times, side="left")
    indices = np.clip(indices, 0, len(imu_sec) - 1)

    left_idx = np.clip(indices - 1, 0, len(imu_sec) - 1)
    right_idx = indices

    left_dist = np.abs(frame_times - imu_sec[left_idx])
    right_dist = np.abs(frame_times - imu_sec[right_idx])

    return np.where(left_dist <= right_dist, left_idx, right_idx)


def get_imu_window(
    imu_data: np.ndarray,
    frame_idx: int,
    alignment: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Get a window of IMU samples centered around a video frame.

    @param imu_data - Full IMU array (N, 11)
    @param frame_idx - Video frame index
    @param alignment - Frame-to-IMU mapping from align_imu_to_video
    @param window_size - Number of IMU samples on each side of center
    @returns IMU data slice of shape (up to 2*window_size+1, 11)
    """
    if frame_idx < 0 or frame_idx >= len(alignment):
        return np.empty((0, imu_data.shape[1]), dtype=imu_data.dtype)

    center = alignment[frame_idx]
    start = max(0, center - window_size)
    end = min(len(imu_data), center + window_size + 1)
    return imu_data[start:end]
