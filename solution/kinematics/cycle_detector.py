"""Finite state machine for detecting shovel dig-swing-dump cycles from joint angles."""

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from solution.config import CYCLE_FSM_CONFIG, FPS
from solution.kinematics.joint_angles import JointAngles

logger = logging.getLogger(__name__)


class Phase(Enum):
    """Shovel operational phases."""

    IDLE = auto()
    DIG = auto()
    SWING_TO_DUMP = auto()
    DUMP = auto()
    SWING_TO_DIG = auto()


@dataclass
class PhaseEvent:
    """A phase transition event with timing info."""

    phase: Phase
    start_frame: int
    end_frame: Optional[int] = None

    @property
    def start_sec(self) -> float:
        return self.start_frame / FPS

    @property
    def end_sec(self) -> Optional[float]:
        if self.end_frame is None:
            return None
        return self.end_frame / FPS

    @property
    def duration_sec(self) -> Optional[float]:
        if self.end_frame is None:
            return None
        return (self.end_frame - self.start_frame) / FPS


@dataclass
class Cycle:
    """A complete dig→swing→dump→return cycle."""

    cycle_id: int
    phases: List[PhaseEvent] = field(default_factory=list)
    start_frame: int = 0
    end_frame: int = 0

    @property
    def duration_sec(self) -> float:
        return (self.end_frame - self.start_frame) / FPS

    def phase_duration(self, phase: Phase) -> float:
        """
        Total seconds spent in a given phase within this cycle.

        @param phase - Phase to query
        @returns Duration in seconds (0 if phase not found)
        """
        total = 0.0
        for event in self.phases:
            if event.phase == phase and event.duration_sec is not None:
                total += event.duration_sec
        return total


class CycleFSM:
    """
    Finite state machine that classifies shovel phases from joint angle time series.

    Transition logic based on angle rates and absolute angle thresholds.
    """

    def __init__(self) -> None:
        self._config = CYCLE_FSM_CONFIG
        self._current_phase = Phase.IDLE
        self._phase_start_frame = 0
        self._prev_angles: Optional[JointAngles] = None
        self._events: List[PhaseEvent] = []
        self._cycles: List[Cycle] = []
        self._current_cycle_phases: List[PhaseEvent] = []
        self._cycle_count = 0
        self._min_phase_frames = int(self._config.min_phase_duration_sec * FPS)

    def update(self, angles: JointAngles, truck_visible: bool = False) -> Phase:
        """
        Feed one frame of joint angles and return current phase.

        @param angles - Joint angles for current frame
        @param truck_visible - Whether truck detected in this frame
        @returns Current classified phase
        """
        self._truck_visible = truck_visible

        if not angles.valid:
            return self._current_phase

        if self._prev_angles is not None and self._prev_angles.valid:
            new_phase = self._classify_transition(angles)
            if new_phase != self._current_phase:
                frames_in_phase = angles.frame_idx - self._phase_start_frame
                if frames_in_phase >= self._min_phase_frames:
                    self._transition_to(new_phase, angles.frame_idx)

        self._prev_angles = angles
        return self._current_phase

    def _classify_transition(self, angles: JointAngles) -> Phase:
        """
        Determine target phase from angle changes between consecutive frames.

        Uses arm angle rate, boom angle rate, and bucket tilt to classify.

        @param angles - Current frame angles
        @returns Target phase based on angle analysis
        """
        arm_rate = abs(angles.arm_angle_deg - self._prev_angles.arm_angle_deg) * FPS
        boom_rate = abs(angles.boom_angle_deg - self._prev_angles.boom_angle_deg) * FPS

        is_idle = (
            arm_rate < self._config.idle_max_angle_rate_deg_s
            and boom_rate < self._config.idle_max_angle_rate_deg_s
        )

        is_digging = arm_rate >= self._config.dig_min_arm_curl_rate_deg_s
        is_swinging = boom_rate > self._config.angle_change_threshold_deg

        bucket_tilt = angles.bucket_angle_deg or 0.0
        is_dumping = bucket_tilt > self._config.dump_bucket_tilt_threshold_deg

        if is_idle:
            return Phase.IDLE

        truck_here = getattr(self, "_truck_visible", False)

        if self._current_phase == Phase.IDLE:
            if is_digging:
                return Phase.DIG
            if is_swinging:
                return Phase.SWING_TO_DUMP

        elif self._current_phase == Phase.DIG:
            if is_swinging:
                return Phase.SWING_TO_DUMP

        elif self._current_phase == Phase.SWING_TO_DUMP:
            if is_dumping or truck_here or (not is_swinging and not is_digging):
                return Phase.DUMP

        elif self._current_phase == Phase.DUMP:
            if is_swinging or (not truck_here and not is_dumping):
                return Phase.SWING_TO_DIG

        elif self._current_phase == Phase.SWING_TO_DIG:
            if is_digging:
                return Phase.DIG
            if is_idle:
                return Phase.IDLE

        return self._current_phase

    def _transition_to(self, new_phase: Phase, frame_idx: int) -> None:
        """Record phase transition and detect completed cycles."""
        event = PhaseEvent(
            phase=self._current_phase,
            start_frame=self._phase_start_frame,
            end_frame=frame_idx,
        )
        self._events.append(event)
        self._current_cycle_phases.append(event)

        if (
            self._current_phase == Phase.SWING_TO_DIG
            and new_phase in (Phase.DIG, Phase.IDLE)
        ):
            self._complete_cycle(frame_idx)

        logger.debug(
            "Phase %s → %s at frame %d (%.1fs)",
            self._current_phase.name,
            new_phase.name,
            frame_idx,
            frame_idx / FPS,
        )

        self._current_phase = new_phase
        self._phase_start_frame = frame_idx

    def _complete_cycle(self, end_frame: int) -> None:
        """Finalize current cycle and add to results."""
        if not self._current_cycle_phases:
            return

        self._cycle_count += 1
        cycle = Cycle(
            cycle_id=self._cycle_count,
            phases=list(self._current_cycle_phases),
            start_frame=self._current_cycle_phases[0].start_frame,
            end_frame=end_frame,
        )
        self._cycles.append(cycle)
        self._current_cycle_phases = []

        logger.info(
            "Cycle %d completed: %.1fs (frames %d–%d)",
            cycle.cycle_id,
            cycle.duration_sec,
            cycle.start_frame,
            cycle.end_frame,
        )

    def finalize(self, last_frame: int) -> None:
        """
        Close any open phase/cycle at end of video.

        @param last_frame - Final frame index
        """
        if self._phase_start_frame < last_frame:
            event = PhaseEvent(
                phase=self._current_phase,
                start_frame=self._phase_start_frame,
                end_frame=last_frame,
            )
            self._events.append(event)

    @property
    def cycles(self) -> List[Cycle]:
        return list(self._cycles)

    @property
    def events(self) -> List[PhaseEvent]:
        return list(self._events)

    @property
    def current_phase(self) -> Phase:
        return self._current_phase
