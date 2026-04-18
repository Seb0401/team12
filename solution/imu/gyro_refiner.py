"""Gyro-Y signal refiner for phase classification support.

Gyro-Y (column 5 of IMU data) correlates with left-right swing motion
of the shovel body. High |gyro_y| = transporting (swinging to dump or dig).
Low |gyro_y| = stationary (digging, dumping, or idle).
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

TRANSPORT_THRESHOLD = 5.0
STATIONARY_THRESHOLD = 2.0
SMOOTHING_WINDOW = 15


class GyroYRefiner:
    """Per-frame gyro-Y signal processor for phase refinement.

    Maintains a rolling smoothed |gyro_y| and classifies each frame as
    transporting, stationary, or ambiguous.
    """

    def __init__(
        self,
        transport_threshold: float = TRANSPORT_THRESHOLD,
        stationary_threshold: float = STATIONARY_THRESHOLD,
        smoothing_window: int = SMOOTHING_WINDOW,
    ) -> None:
        self._transport_thresh = transport_threshold
        self._stationary_thresh = stationary_threshold
        self._buffer: deque[float] = deque(maxlen=smoothing_window)
        self._raw_value: float = 0.0
        self._smoothed_abs: float = 0.0

    @property
    def raw_gyro_y(self) -> float:
        return self._raw_value

    @property
    def smoothed_abs_gyro_y(self) -> float:
        return self._smoothed_abs

    @property
    def is_transporting(self) -> bool:
        return self._smoothed_abs > self._transport_thresh

    @property
    def is_stationary(self) -> bool:
        return self._smoothed_abs < self._stationary_thresh

    def update(self, gyro_y_value: float) -> None:
        """Feed one frame of gyro-Y data.

        @param gyro_y_value - Raw gyro Y-axis value for current frame/sample
        """
        self._raw_value = gyro_y_value
        self._buffer.append(abs(gyro_y_value))
        self._smoothed_abs = sum(self._buffer) / len(self._buffer)

    def reset(self) -> None:
        self._buffer.clear()
        self._raw_value = 0.0
        self._smoothed_abs = 0.0


def precompute_gyro_y_smooth(imu_raw: np.ndarray, window: int = SMOOTHING_WINDOW) -> np.ndarray:
    """Precompute smoothed |gyro_y| for entire IMU dataset.

    @param imu_raw - Full IMU array (N, 11)
    @param window - Smoothing window size
    @returns Smoothed |gyro_y| array, shape (N,)
    """
    gyro_y = imu_raw[:, 5]
    abs_gy = np.abs(gyro_y)
    kernel = np.ones(window) / window
    smoothed = np.convolve(abs_gy, kernel, mode="same")
    return smoothed
