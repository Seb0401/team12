"""IMU-only cycle detection using gyroscope peak analysis."""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks

from solution.config import IMU_CONFIG
from solution.imu.processor import ProcessedIMU

logger = logging.getLogger(__name__)


@dataclass
class IMUCycleEvent:
    """A swing event detected from IMU gyroscope data."""

    sample_idx: int
    timestamp_sec: float
    gyro_magnitude: float
    direction: str  # "to_dump" or "to_dig" based on gyro sign


@dataclass
class IMUCycle:
    """A pair of swing events forming one dig-dump cycle."""

    cycle_id: int
    swing_to_dump: IMUCycleEvent
    swing_to_dig: IMUCycleEvent

    @property
    def duration_sec(self) -> float:
        return self.swing_to_dig.timestamp_sec - self.swing_to_dump.timestamp_sec


def detect_swing_peaks(processed: ProcessedIMU) -> List[IMUCycleEvent]:
    """
    Detect swing events from gyroscope magnitude peaks.

    Swings are the most distinctive IMU signal: large rotation
    around the Z-axis (yaw) when the shovel body rotates.

    @param processed - Filtered IMU data
    @returns List of swing events ordered by time
    """
    cfg = IMU_CONFIG
    gyro_z = processed.gyro_filtered[:, 2]

    positive_peaks, pos_props = find_peaks(
        gyro_z,
        distance=cfg.peak_min_distance_samples,
        prominence=cfg.peak_min_prominence,
    )

    negative_peaks, neg_props = find_peaks(
        -gyro_z,
        distance=cfg.peak_min_distance_samples,
        prominence=cfg.peak_min_prominence,
    )

    events: List[IMUCycleEvent] = []

    for idx in positive_peaks:
        events.append(IMUCycleEvent(
            sample_idx=int(idx),
            timestamp_sec=float(processed.timestamps_sec[idx]),
            gyro_magnitude=float(processed.gyro_magnitude[idx]),
            direction="to_dump",
        ))

    for idx in negative_peaks:
        events.append(IMUCycleEvent(
            sample_idx=int(idx),
            timestamp_sec=float(processed.timestamps_sec[idx]),
            gyro_magnitude=float(processed.gyro_magnitude[idx]),
            direction="to_dig",
        ))

    events.sort(key=lambda e: e.timestamp_sec)

    logger.info(
        "IMU swing detection: %d peaks (%d to_dump, %d to_dig)",
        len(events),
        len(positive_peaks),
        len(negative_peaks),
    )

    return events


def pair_swing_events(events: List[IMUCycleEvent]) -> List[IMUCycle]:
    """
    Pair consecutive to_dump → to_dig swings into complete cycles.

    @param events - Sorted swing events from detect_swing_peaks
    @returns List of paired cycles
    """
    cycles: List[IMUCycle] = []
    cycle_id = 0
    pending_dump: IMUCycleEvent | None = None

    for event in events:
        if event.direction == "to_dump":
            pending_dump = event
        elif event.direction == "to_dig" and pending_dump is not None:
            cycle_id += 1
            cycle = IMUCycle(
                cycle_id=cycle_id,
                swing_to_dump=pending_dump,
                swing_to_dig=event,
            )

            if cycle.duration_sec > IMU_CONFIG.peak_min_distance_samples / IMU_CONFIG.sample_rate_hz:
                cycles.append(cycle)
                logger.debug(
                    "IMU cycle %d: %.1f–%.1fs (%.1fs)",
                    cycle.cycle_id,
                    pending_dump.timestamp_sec,
                    event.timestamp_sec,
                    cycle.duration_sec,
                )
            pending_dump = None

    logger.info("IMU cycle detection: %d complete cycles", len(cycles))
    return cycles
