"""Fused phase detector: BucketPhaseDetector (vision) refined by Gyro-Y (IMU).

Vision provides primary phase classification. Gyro-Y refines timing:
- Vision TRANSPORTE + gyro moving    -> TRANSPORTE (confirmed)
- Vision TRANSPORTE + gyro still     -> INACTIVO (override)
- Vision INACTIVO + gyro moving      -> TRANSPORTE (override)
- Vision EXCAVANDO + gyro still      -> EXCAVANDO (confirmed)
- Vision BOTANDO_CARGA + gyro still  -> BOTANDO_CARGA (confirmed)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from solution.config import FPS
from solution.detection.detector import FrameDetections
from solution.imu.gyro_refiner import GyroYRefiner
from solution.kinematics.bucket_phase import BucketPhase, BucketPhaseDetector

logger = logging.getLogger(__name__)


@dataclass
class RefinedPhaseEvent:
    """A phase event with refined start/end from gyro-Y."""

    phase: BucketPhase
    start_frame: int
    end_frame: Optional[int] = None

    @property
    def duration_sec(self) -> Optional[float]:
        if self.end_frame is None:
            return None
        return (self.end_frame - self.start_frame) / FPS


@dataclass
class RefinedCycle:
    """A complete excavation cycle with gyro-refined phase timings."""

    cycle_id: int
    start_frame: int
    end_frame: int
    phases: List[RefinedPhaseEvent] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame) / FPS

    def phase_duration(self, phase: BucketPhase) -> float:
        total = 0.0
        for event in self.phases:
            dur = event.duration_sec
            if event.phase == phase and dur is not None:
                total += dur
        return total


class FusedPhaseDetector:
    """Combines BucketPhaseDetector output with GyroYRefiner for accurate phase timing.

    Vision is authoritative for DIG and DUMP phases.
    Gyro-Y refines TRANSPORT vs IDLE classification and sharpens transition boundaries.
    """

    def __init__(self) -> None:
        self._vision = BucketPhaseDetector()
        self._gyro = GyroYRefiner()
        self._current_phase = BucketPhase.INACTIVO
        self._phase_start_frame = 0
        self._min_phase_frames = 8

        self._events: List[RefinedPhaseEvent] = []
        self._cycles: List[RefinedCycle] = []
        self._current_cycle_phases: List[RefinedPhaseEvent] = []
        self._cycle_count = 0

    @property
    def current_phase(self) -> BucketPhase:
        return self._current_phase

    @property
    def cycles(self) -> List[RefinedCycle]:
        return list(self._cycles)

    @property
    def events(self) -> List[RefinedPhaseEvent]:
        return list(self._events)

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def gyro_refiner(self) -> GyroYRefiner:
        return self._gyro

    @property
    def vision_detector(self) -> BucketPhaseDetector:
        return self._vision

    def update(self, detections: FrameDetections, gyro_y_value: float) -> BucketPhase:
        """Classify one frame using vision + gyro-Y fusion.

        @param detections - YOLO detections for this frame
        @param gyro_y_value - Raw gyro Y-axis value for aligned IMU sample
        @returns Refined phase classification
        """
        frame_idx = detections.frame_idx
        vision_phase = self._vision.update(detections)
        self._gyro.update(gyro_y_value)

        refined = self._refine(vision_phase)

        if refined != self._current_phase:
            frames_in = frame_idx - self._phase_start_frame
            if frames_in >= self._min_phase_frames:
                self._transition(refined, frame_idx)

        return self._current_phase

    def _refine(self, vision_phase: BucketPhase) -> BucketPhase:
        """Apply gyro-Y refinement rules to vision phase.

        @param vision_phase - Phase from BucketPhaseDetector
        @returns Refined phase
        """
        if vision_phase == BucketPhase.TRANSPORTE and self._gyro.is_stationary:
            return BucketPhase.INACTIVO

        if vision_phase == BucketPhase.INACTIVO and self._gyro.is_transporting:
            return BucketPhase.TRANSPORTE

        return vision_phase

    def _transition(self, new_phase: BucketPhase, frame_idx: int) -> None:
        event = RefinedPhaseEvent(
            phase=self._current_phase,
            start_frame=self._phase_start_frame,
            end_frame=frame_idx,
        )
        self._events.append(event)
        self._current_cycle_phases.append(event)

        if self._current_phase == BucketPhase.BOTANDO_CARGA:
            dump_duration = (frame_idx - self._phase_start_frame) / FPS
            if dump_duration >= 3.0:
                self._complete_cycle(frame_idx)

        self._current_phase = new_phase
        self._phase_start_frame = frame_idx

    def _complete_cycle(self, end_frame: int) -> None:
        if not self._current_cycle_phases:
            return

        start_frame = self._current_cycle_phases[0].start_frame
        duration = (end_frame - start_frame) / FPS

        if duration < 10.0:
            self._current_cycle_phases = []
            return

        self._cycle_count += 1
        cycle = RefinedCycle(
            cycle_id=self._cycle_count,
            start_frame=start_frame,
            end_frame=end_frame,
            phases=list(self._current_cycle_phases),
        )
        self._cycles.append(cycle)
        self._current_cycle_phases = []

        logger.info(
            "Refined cycle %d: %.1fs (frames %d-%d)",
            cycle.cycle_id, cycle.duration_sec, start_frame, end_frame,
        )

    def finalize(self, last_frame: int) -> None:
        """Close open phase/cycle at end of video.

        @param last_frame - Final frame index
        """
        if self._phase_start_frame < last_frame:
            event = RefinedPhaseEvent(
                phase=self._current_phase,
                start_frame=self._phase_start_frame,
                end_frame=last_frame,
            )
            self._events.append(event)

        self._vision.finalize(last_frame)
