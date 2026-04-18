"""Phase classification based on bucket Y-position, X-velocity, and truck visibility."""

from collections import deque
from enum import Enum, auto
from typing import List, Optional

from solution.config import BUCKET_PHASE_CONFIG, FPS
from solution.detection.detector import Detection, FrameDetections


class BucketPhase(Enum):
    """Phases derived from bucket position + truck presence."""

    EXCAVANDO = auto()
    TRANSPORTE = auto()
    BOTANDO_CARGA = auto()
    INACTIVO = auto()


PHASE_DISPLAY = {
    BucketPhase.EXCAVANDO: "EXCAVANDO",
    BucketPhase.TRANSPORTE: "TRANSPORTE",
    BucketPhase.BOTANDO_CARGA: "BOTANDO CARGA",
    BucketPhase.INACTIVO: "INACTIVO",
}

PHASE_COLORS_BGR = {
    BucketPhase.EXCAVANDO: (0, 200, 0),
    BucketPhase.TRANSPORTE: (0, 220, 255),
    BucketPhase.BOTANDO_CARGA: (0, 0, 220),
    BucketPhase.INACTIVO: (128, 128, 128),
}


class BucketPhaseDetector:
    """
    Classifies the shovel phase each frame using bucket Y, X-velocity, and truck detection.

    Rules:
      - bucket low  (cy > bucket_low_y)                        -> EXCAVANDO
      - bucket high + truck visible                             -> BOTANDO CARGA
      - bucket NOT low + moving on X-axis (|dx| > threshold)   -> TRANSPORTE
      - bucket NOT low + stationary X for >= 4s                 -> INACTIVO (alert)
    """

    def __init__(self) -> None:
        cfg = BUCKET_PHASE_CONFIG
        self._low_y = cfg.bucket_low_y
        self._high_y = cfg.bucket_high_y
        self._smooth_win = cfg.smoothing_window
        self._min_phase_frames = cfg.min_phase_frames
        self._x_vel_threshold = cfg.x_velocity_threshold
        self._transport_x_vel_threshold = cfg.transport_x_velocity_threshold
        self._x_smooth_win = cfg.x_smoothing_window
        self._min_dump_frames = int(cfg.min_dump_duration_sec * FPS)
        self._min_idle_frames = int(cfg.min_idle_duration_sec * FPS)

        self._y_buffer: deque[float] = deque(maxlen=self._smooth_win)
        self._x_buffer: deque[float] = deque(maxlen=self._x_smooth_win)
        self._y_vel_buffer: deque[float] = deque(maxlen=self._x_smooth_win)
        self._prev_bucket_x: Optional[float] = None
        self._x_velocity: float = 0.0
        self._y_velocity: float = 0.0
        self._prev_raw_y: Optional[float] = None

        self._current_phase = BucketPhase.INACTIVO
        self._phase_start_frame = 0
        self._prev_bucket_y: Optional[float] = None

        self._idle_candidate_start: Optional[int] = None
        self._idle_confirmed = False
        self._idle_start_frame: Optional[int] = None

        self._idle_events: List[dict] = []
        self._last_truck_y: Optional[float] = None

        self._events: List[dict] = []
        self._cycles: List[dict] = []
        self._current_cycle_phases: List[dict] = []
        self._cycle_count = 0

    @property
    def x_velocity(self) -> float:
        """Current smoothed X-axis velocity (px/frame)."""
        return self._x_velocity

    @property
    def y_velocity(self) -> float:
        return self._y_velocity

    @property
    def is_idle_alert(self) -> bool:
        return self._idle_confirmed

    @property
    def idle_duration_sec(self) -> float:
        if not self._idle_confirmed or self._idle_start_frame is None:
            return 0.0
        return (self._last_frame_idx - self._idle_start_frame) / FPS

    def update(self, detections: FrameDetections) -> BucketPhase:
        """
        Classify one frame.

        @param detections - YOLO detections for this frame
        @returns Current phase
        """
        self._last_frame_idx = detections.frame_idx
        bucket = detections.get_by_class("bucket")
        truck = detections.get_by_class("truck")

        if bucket is None:
            return self._current_phase

        if truck is not None:
            self._last_truck_y = truck.center[1]

        bucket_cx, bucket_cy = bucket.center

        self._y_buffer.append(bucket_cy)
        smoothed_y = sum(self._y_buffer) / len(self._y_buffer)
        self._prev_bucket_y = smoothed_y

        self._update_x_velocity(bucket_cx)
        self._update_y_velocity(bucket_cy)

        raw_phase = self._classify_raw(smoothed_y, bucket, truck)
        effective_phase = self._apply_idle_grace(raw_phase, detections.frame_idx)

        if effective_phase != self._current_phase:
            frames_in = detections.frame_idx - self._phase_start_frame
            if frames_in >= self._min_phase_frames:
                self._transition(effective_phase, detections.frame_idx)

        return self._current_phase

    def _update_x_velocity(self, bucket_cx: float) -> None:
        if self._prev_bucket_x is not None:
            dx = abs(bucket_cx - self._prev_bucket_x)
            self._x_buffer.append(dx)
            self._x_velocity = sum(self._x_buffer) / len(self._x_buffer)
        self._prev_bucket_x = bucket_cx

    def _update_y_velocity(self, bucket_cy: float) -> None:
        if self._prev_raw_y is not None:
            dy = abs(bucket_cy - self._prev_raw_y)
            self._y_vel_buffer.append(dy)
            self._y_velocity = sum(self._y_vel_buffer) / len(self._y_vel_buffer)
        self._prev_raw_y = bucket_cy

    def _is_moving_x(self) -> bool:
        return self._x_velocity > self._x_vel_threshold

    def _is_moving_y(self) -> bool:
        return self._y_velocity > self._x_vel_threshold

    def _is_transporting_x(self) -> bool:
        return self._x_velocity > self._transport_x_vel_threshold

    def _bucket_overlaps_truck_x(self, bucket: Detection, truck: Detection) -> bool:
        overlap_start = max(bucket.x1, truck.x1)
        overlap_end = min(bucket.x2, truck.x2)
        if overlap_end <= overlap_start:
            return False
        overlap_width = overlap_end - overlap_start
        bucket_width = bucket.x2 - bucket.x1
        return overlap_width > bucket_width * 0.3

    def _is_bucket_above_truck(self, bucket_y: float) -> bool:
        """Check if bucket is above truck height. Y increases downward, so bucket_y < truck_y = above."""
        if self._last_truck_y is None:
            return bucket_y < self._high_y
        return bucket_y < self._last_truck_y

    def _classify_raw(
        self,
        bucket_y: float,
        bucket: Detection,
        truck: Optional[Detection],
    ) -> BucketPhase:
        if bucket_y > self._low_y:
            return BucketPhase.EXCAVANDO

        if (
            truck is not None
            and self._is_bucket_above_truck(bucket_y)
            and self._bucket_overlaps_truck_x(bucket, truck)
        ):
            return BucketPhase.BOTANDO_CARGA

        if self._is_bucket_above_truck(bucket_y) and self._is_transporting_x():
            return BucketPhase.TRANSPORTE

        return BucketPhase.INACTIVO

    def _apply_idle_grace(self, raw_phase: BucketPhase, frame_idx: int) -> BucketPhase:
        """
        Suppress INACTIVO until raw classification stays INACTIVO for >= min_idle_duration_sec
        AND both X and Y axes are truly stationary. Any axis movement resets the grace timer.
        """
        has_any_movement = self._is_moving_x() or self._is_moving_y()

        if raw_phase != BucketPhase.INACTIVO or has_any_movement:
            if self._idle_confirmed:
                self._close_idle_event(frame_idx)
            self._idle_candidate_start = None
            self._idle_confirmed = False
            self._idle_start_frame = None
            if raw_phase != BucketPhase.INACTIVO:
                return raw_phase
            return self._current_phase

        if self._idle_candidate_start is None:
            self._idle_candidate_start = frame_idx

        elapsed_idle_frames = frame_idx - self._idle_candidate_start

        if elapsed_idle_frames >= self._min_idle_frames:
            if not self._idle_confirmed:
                self._idle_confirmed = True
                self._idle_start_frame = self._idle_candidate_start
            return BucketPhase.INACTIVO

        return self._current_phase

    def _close_idle_event(self, end_frame: int) -> None:
        if self._idle_start_frame is None:
            return
        duration = (end_frame - self._idle_start_frame) / FPS
        self._idle_events.append({
            "start_frame": self._idle_start_frame,
            "end_frame": end_frame,
            "start_sec": round(self._idle_start_frame / FPS, 2),
            "end_sec": round(end_frame / FPS, 2),
            "duration_sec": round(duration, 2),
        })

    def _transition(self, new_phase: BucketPhase, frame_idx: int) -> None:
        event = {
            "phase": self._current_phase,
            "start_frame": self._phase_start_frame,
            "end_frame": frame_idx,
            "duration_sec": (frame_idx - self._phase_start_frame) / FPS,
        }
        self._events.append(event)
        self._current_cycle_phases.append(event)

        if self._current_phase == BucketPhase.BOTANDO_CARGA:
            dump_duration_frames = frame_idx - self._phase_start_frame
            if dump_duration_frames >= self._min_dump_frames:
                self._complete_cycle(frame_idx)

        self._current_phase = new_phase
        self._phase_start_frame = frame_idx

    def _complete_cycle(self, end_frame: int) -> None:
        if not self._current_cycle_phases:
            return

        start_frame = self._current_cycle_phases[0]["start_frame"]
        duration = (end_frame - start_frame) / FPS

        if duration < 10.0:
            self._current_cycle_phases = []
            return

        self._cycle_count += 1

        phase_breakdown: dict[str, float] = {}
        for ev in self._current_cycle_phases:
            pname = ev["phase"].name
            phase_breakdown[pname] = phase_breakdown.get(pname, 0) + ev["duration_sec"]

        cycle = {
            "cycle_id": self._cycle_count,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_sec": start_frame / FPS,
            "end_sec": end_frame / FPS,
            "duration_sec": duration,
            "phase_breakdown": {k: round(v, 2) for k, v in phase_breakdown.items()},
            "phases": list(self._current_cycle_phases),
        }
        self._cycles.append(cycle)
        self._current_cycle_phases = []

    def finalize(self, last_frame: int) -> None:
        """Close open phase and idle event at end of video."""
        if self._phase_start_frame < last_frame:
            event = {
                "phase": self._current_phase,
                "start_frame": self._phase_start_frame,
                "end_frame": last_frame,
                "duration_sec": (last_frame - self._phase_start_frame) / FPS,
            }
            self._events.append(event)

        if self._idle_confirmed:
            self._close_idle_event(last_frame)

    @property
    def cycles(self) -> List[dict]:
        return list(self._cycles)

    @property
    def events(self) -> List[dict]:
        return list(self._events)

    @property
    def idle_events(self) -> List[dict]:
        return list(self._idle_events)

    @property
    def current_phase(self) -> BucketPhase:
        return self._current_phase

    @property
    def cycle_count(self) -> int:
        return self._cycle_count
