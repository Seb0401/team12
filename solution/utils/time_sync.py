"""Synchronization utilities for aligning video frames with IMU timestamps.

Key finding from cross-correlation analysis:
- IMU and video share the same sample count (1:1 mapping).
- IMU timestamps reflect wall-clock time with gaps (sensor pauses/drops).
- Video records continuously at a fixed FPS.
- The relationship is: imu_sample_index = frame_index + offset.
- Cross-correlation of gyro magnitude vs video motion energy yields the offset.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import correlate

logger = logging.getLogger(__name__)

DEFAULT_OFFSET_SAMPLES = -3
MAX_CROSS_CORR_LAG = 100


@dataclass
class AlignmentResult:
    """Result of IMU-to-video temporal alignment.

    @param offset_samples - IMU leads video by this many samples (negative = IMU leads)
    @param offset_sec - Offset in seconds
    @param correlation_score - Peak normalized cross-correlation value
    @param method - How the offset was determined ('cross_correlation' or 'default')
    @param segment_offsets - Per-segment offsets for drift analysis (list of (seg_idx, offset) tuples)
    """

    offset_samples: int
    offset_sec: float
    correlation_score: float
    method: str
    segment_offsets: list


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


def imu_sample_to_frame(sample_idx: int, alignment: "AlignmentResult") -> int:
    """
    Map an IMU sample index to the corresponding video frame index.

    @param sample_idx - Zero-based IMU sample index
    @param alignment - Alignment result from compute_alignment
    @returns Corresponding video frame index
    """
    return sample_idx - alignment.offset_samples


def frame_to_imu_sample(frame_idx: int, alignment: "AlignmentResult") -> int:
    """
    Map a video frame index to the corresponding IMU sample index.

    @param frame_idx - Zero-based video frame index
    @param alignment - Alignment result from compute_alignment
    @returns Corresponding IMU sample index
    """
    return frame_idx + alignment.offset_samples


def _normalize(signal: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance normalization.

    @param signal - Input signal array
    @returns Normalized signal (or zeros if std ≈ 0)
    """
    std = signal.std()
    if std < 1e-8:
        return np.zeros_like(signal)
    return (signal - signal.mean()) / std


def compute_alignment_cross_correlation(
    gyro_magnitude: np.ndarray,
    motion_energy: np.ndarray,
    fps: float,
    max_lag: int = MAX_CROSS_CORR_LAG,
) -> AlignmentResult:
    """
    Compute IMU-to-video offset via cross-correlation of gyro magnitude and video motion energy.

    Both signals must have the same length (1:1 sample-to-frame mapping).

    @param gyro_magnitude - IMU gyro magnitude per sample, shape (N,)
    @param motion_energy - Video frame-to-frame motion energy, shape (N,)
    @param fps - Video FPS (for converting offset to seconds)
    @param max_lag - Maximum lag in samples to search (default 100 = ±6.7s at 15fps)
    @returns AlignmentResult with offset and diagnostics
    """
    if len(gyro_magnitude) != len(motion_energy):
        logger.warning(
            "Signal length mismatch: gyro=%d, motion=%d. Using min length.",
            len(gyro_magnitude), len(motion_energy),
        )
        n = min(len(gyro_magnitude), len(motion_energy))
        gyro_magnitude = gyro_magnitude[:n]
        motion_energy = motion_energy[:n]

    gm_norm = _normalize(gyro_magnitude)
    me_norm = _normalize(motion_energy)

    corr = correlate(me_norm, gm_norm, mode="full")
    center = len(gm_norm) - 1
    lags = np.arange(-len(gm_norm) + 1, len(me_norm))

    lo = max(0, center - max_lag)
    hi = min(len(corr), center + max_lag + 1)
    corr_window = corr[lo:hi]
    lag_window = lags[lo:hi]

    best_idx = int(np.argmax(corr_window))
    best_lag = int(lag_window[best_idx])
    best_score = float(corr_window[best_idx])

    zero_lag_score = float(corr[center])
    norm_score = best_score / (len(gm_norm) + 1e-8)

    segment_offsets = _compute_segment_offsets(
        gm_norm, me_norm, num_segments=4, max_lag=max_lag,
    )

    logger.info(
        "Cross-correlation: offset=%d samples (%.3fs), score=%.2f, zero_lag=%.2f",
        best_lag, best_lag / fps, best_score, zero_lag_score,
    )
    for seg_idx, seg_offset in segment_offsets:
        logger.info("  Segment %d: offset=%d samples", seg_idx, seg_offset)

    return AlignmentResult(
        offset_samples=best_lag,
        offset_sec=best_lag / fps,
        correlation_score=norm_score,
        method="cross_correlation",
        segment_offsets=segment_offsets,
    )


def _compute_segment_offsets(
    signal_a: np.ndarray,
    signal_b: np.ndarray,
    num_segments: int = 4,
    max_lag: int = 50,
) -> list:
    """
    Compute per-segment cross-correlation offsets for drift analysis.

    @param signal_a - Normalized signal A (gyro)
    @param signal_b - Normalized signal B (motion energy)
    @param num_segments - Number of segments to split into
    @param max_lag - Maximum lag per segment
    @returns List of (segment_index, offset_samples) tuples
    """
    n = len(signal_a)
    seg_size = n // num_segments
    results = []

    for seg_idx in range(num_segments):
        start = seg_idx * seg_size
        end = start + seg_size if seg_idx < num_segments - 1 else n

        seg_a = signal_a[start:end]
        seg_b = signal_b[start:end]

        corr = correlate(seg_b, seg_a, mode="full")
        center = len(seg_a) - 1

        lo = max(0, center - max_lag)
        hi = min(len(corr), center + max_lag + 1)
        corr_window = corr[lo:hi]
        lag_window = np.arange(lo - center, hi - center)

        best_idx = int(np.argmax(corr_window))
        best_lag = int(lag_window[best_idx])

        results.append((seg_idx + 1, best_lag))

    return results


def compute_alignment_default(fps: float) -> AlignmentResult:
    """
    Return the default alignment (no cross-correlation data available).

    Uses the empirically determined offset from dev dataset analysis.

    @param fps - Video FPS
    @returns AlignmentResult with default offset
    """
    return AlignmentResult(
        offset_samples=DEFAULT_OFFSET_SAMPLES,
        offset_sec=DEFAULT_OFFSET_SAMPLES / fps,
        correlation_score=0.0,
        method="default",
        segment_offsets=[],
    )


def align_imu_to_video(
    imu_timestamps: np.ndarray,
    video_fps: float,
    total_frames: int,
) -> np.ndarray:
    """
    Build a mapping from each video frame to the closest IMU sample index.

    This is the legacy interface used by main.py. It applies a simple 1:1
    mapping with the default offset correction.

    @param imu_timestamps - Raw epoch nanosecond timestamps from IMU col 0
    @param video_fps - Video FPS
    @param total_frames - Total number of video frames
    @returns Array of shape (total_frames,) with IMU sample indices
    """
    if len(imu_timestamps) == 0 or total_frames == 0:
        return np.array([], dtype=np.int64)

    num_imu = len(imu_timestamps)
    offset = DEFAULT_OFFSET_SAMPLES

    frame_indices = np.arange(total_frames)
    imu_indices = frame_indices + offset
    imu_indices = np.clip(imu_indices, 0, num_imu - 1)

    return imu_indices.astype(np.int64)


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
