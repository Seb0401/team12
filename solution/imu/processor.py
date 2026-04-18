"""IMU signal processing: filtering and feature extraction."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import signal

from solution.config import IMU_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ProcessedIMU:
    """Filtered IMU signals ready for cycle detection."""

    timestamps_sec: np.ndarray
    accel_filtered: np.ndarray      # (N, 3) low-pass filtered
    gyro_filtered: np.ndarray       # (N, 3) band-pass filtered
    accel_magnitude: np.ndarray     # (N,)
    gyro_magnitude: np.ndarray      # (N,)
    quaternions: np.ndarray         # (N, 4) raw quaternions


def _butterworth_lowpass(data: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Apply Butterworth low-pass filter along axis 0.

    @param data - Input signal (N,) or (N, C)
    @param cutoff_hz - Cutoff frequency in Hz
    @param fs - Sampling frequency in Hz
    @param order - Filter order
    @returns Filtered signal, same shape as input
    """
    nyq = fs / 2.0
    normalized_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(order, normalized_cutoff, btype="low")
    return signal.filtfilt(b, a, data, axis=0)


def _butterworth_bandpass(
    data: np.ndarray,
    low_hz: float,
    high_hz: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """
    Apply Butterworth band-pass filter along axis 0.

    @param data - Input signal (N,) or (N, C)
    @param low_hz - Lower cutoff frequency in Hz
    @param high_hz - Upper cutoff frequency in Hz
    @param fs - Sampling frequency in Hz
    @param order - Filter order
    @returns Filtered signal, same shape as input
    """
    nyq = fs / 2.0
    low = max(low_hz / nyq, 0.01)
    high = min(high_hz / nyq, 0.99)
    b, a = signal.butter(order, [low, high], btype="band")
    return signal.filtfilt(b, a, data, axis=0)


def process_imu(raw_data: np.ndarray) -> ProcessedIMU:
    """
    Filter raw IMU data and compute derived signals.

    @param raw_data - Raw IMU array (N, 11)
    @returns ProcessedIMU with filtered signals and magnitudes
    """
    cfg = IMU_CONFIG

    timestamps_ns = raw_data[:, cfg.timestamp_col]
    timestamps_sec = (timestamps_ns - timestamps_ns[0]) / 1e9

    accel_raw = raw_data[:, list(cfg.accel_cols)]
    gyro_raw = raw_data[:, list(cfg.gyro_cols)]
    quaternions = raw_data[:, list(cfg.quat_cols)]

    actual_fs = _estimate_sample_rate(timestamps_sec)
    if abs(actual_fs - cfg.sample_rate_hz) > 2.0:
        logger.warning(
            "IMU sample rate %.1f Hz differs from expected %.1f Hz",
            actual_fs,
            cfg.sample_rate_hz,
        )

    accel_filtered = _butterworth_lowpass(accel_raw, cfg.accel_lowpass_cutoff_hz, actual_fs)
    gyro_filtered = _butterworth_bandpass(
        gyro_raw, cfg.gyro_bandpass_low_hz, cfg.gyro_bandpass_high_hz, actual_fs
    )

    accel_magnitude = np.linalg.norm(accel_filtered, axis=1)
    gyro_magnitude = np.linalg.norm(gyro_filtered, axis=1)

    logger.info(
        "IMU processed: %d samples, %.1f Hz, %.1fs duration",
        len(timestamps_sec),
        actual_fs,
        timestamps_sec[-1],
    )

    return ProcessedIMU(
        timestamps_sec=timestamps_sec,
        accel_filtered=accel_filtered,
        gyro_filtered=gyro_filtered,
        accel_magnitude=accel_magnitude,
        gyro_magnitude=gyro_magnitude,
        quaternions=quaternions,
    )


def _estimate_sample_rate(timestamps_sec: np.ndarray) -> float:
    """
    Estimate actual sample rate from timestamp differences.

    @param timestamps_sec - Relative timestamps in seconds
    @returns Estimated sample rate in Hz
    """
    if len(timestamps_sec) < 2:
        return IMU_CONFIG.sample_rate_hz
    dt = np.median(np.diff(timestamps_sec))
    return 1.0 / dt if dt > 0 else IMU_CONFIG.sample_rate_hz
