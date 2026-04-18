"""Fuse IMU-detected cycles with video-detected cycles."""

import logging
from dataclasses import dataclass
from typing import List, Tuple

from solution.config import FUSION_TOLERANCE_SEC, FPS
from solution.imu.cycle_detector import IMUCycle
from solution.kinematics.cycle_detector import Cycle, Phase

logger = logging.getLogger(__name__)


@dataclass
class FusedCycle:
    """A cycle confirmed by both video and IMU, or by one source alone."""

    cycle_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    video_cycle: Cycle | None = None
    imu_cycle: IMUCycle | None = None
    source: str = "fused"  # "fused", "video_only", "imu_only"


def fuse_cycles(
    video_cycles: List[Cycle],
    imu_cycles: List[IMUCycle],
    tolerance_sec: float = FUSION_TOLERANCE_SEC,
) -> List[FusedCycle]:
    """
    Merge video and IMU cycle detections into a unified timeline.

    Matching: video cycle start within ±tolerance of IMU swing_to_dump timestamp.
    Unmatched cycles kept with source label.

    @param video_cycles - Cycles from FSM
    @param imu_cycles - Cycles from IMU gyro peaks
    @param tolerance_sec - Max time offset for matching
    @returns Merged list of FusedCycle, sorted by start time
    """
    matched_video: set[int] = set()
    matched_imu: set[int] = set()
    fused: List[FusedCycle] = []
    fused_id = 0

    for vc in video_cycles:
        vc_start = vc.start_frame / FPS
        best_imu = None
        best_dist = float("inf")

        for ic in imu_cycles:
            if ic.cycle_id in matched_imu:
                continue
            dist = abs(vc_start - ic.swing_to_dump.timestamp_sec)
            if dist < best_dist and dist <= tolerance_sec:
                best_dist = dist
                best_imu = ic

        fused_id += 1
        vc_end = vc.end_frame / FPS

        if best_imu is not None:
            matched_video.add(vc.cycle_id)
            matched_imu.add(best_imu.cycle_id)
            fused.append(FusedCycle(
                cycle_id=fused_id,
                start_sec=vc_start,
                end_sec=vc_end,
                duration_sec=vc_end - vc_start,
                video_cycle=vc,
                imu_cycle=best_imu,
                source="fused",
            ))
        else:
            matched_video.add(vc.cycle_id)
            fused.append(FusedCycle(
                cycle_id=fused_id,
                start_sec=vc_start,
                end_sec=vc_end,
                duration_sec=vc_end - vc_start,
                video_cycle=vc,
                source="video_only",
            ))

    for ic in imu_cycles:
        if ic.cycle_id in matched_imu:
            continue
        fused_id += 1
        fused.append(FusedCycle(
            cycle_id=fused_id,
            start_sec=ic.swing_to_dump.timestamp_sec,
            end_sec=ic.swing_to_dig.timestamp_sec,
            duration_sec=ic.duration_sec,
            imu_cycle=ic,
            source="imu_only",
        ))

    fused.sort(key=lambda c: c.start_sec)

    fused_count = sum(1 for c in fused if c.source == "fused")
    video_only = sum(1 for c in fused if c.source == "video_only")
    imu_only = sum(1 for c in fused if c.source == "imu_only")

    logger.info(
        "Fusion: %d total (%d fused, %d video-only, %d imu-only)",
        len(fused), fused_count, video_only, imu_only,
    )

    return fused
