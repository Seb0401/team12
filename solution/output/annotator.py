"""Draw detection overlays, cycle HUD, and metrics on video frames."""

import logging
from typing import Optional

import cv2
import numpy as np

from solution.config import (
    COLORS,
    OUTPUTS_DIR,
    ANNOTATED_VIDEO_FILENAME,
    ANNOTATED_VIDEO_CODEC,
    FPS,
    RESOLUTION,
)
from solution.detection.detector import FrameDetections
from solution.kinematics.cycle_detector import Phase
from solution.kinematics.joint_angles import JointAngles

logger = logging.getLogger(__name__)

CLASS_COLOR_MAP = {
    "bucket": COLORS.bucket,
    "arm_joint": COLORS.arm_joint,
    "boom": COLORS.boom,
    "truck": COLORS.truck,
}

PHASE_COLORS = {
    Phase.IDLE: (128, 128, 128),
    Phase.DIG: (0, 200, 0),
    Phase.SWING_TO_DUMP: (0, 220, 255),
    Phase.DUMP: (0, 0, 220),
    Phase.SWING_TO_DIG: (220, 180, 0),
}

PHASE_LABELS = {
    Phase.IDLE: "INACTIVO",
    Phase.DIG: "EXCAVANDO",
    Phase.SWING_TO_DUMP: "GIRO > DESCARGA",
    Phase.DUMP: "DESCARGA",
    Phase.SWING_TO_DIG: "GIRO > EXCAVAR",
}

CLASS_DISPLAY_NAMES = {
    "bucket": "balde",
    "arm_joint": "articulacion",
    "boom": "pluma",
    "truck": "camion",
}

HUD_WIDTH = 310
HUD_HEIGHT = 180
HUD_MARGIN = 12
HUD_ALPHA = 0.65


class VideoAnnotator:
    """Draws bounding boxes, cycle HUD panel, and metrics onto video frames."""

    def __init__(self, fps: float = FPS, resolution: tuple = RESOLUTION) -> None:
        self._fps = fps
        self._width, self._height = resolution
        self._writer: Optional[cv2.VideoWriter] = None
        self._prev_phase: Optional[Phase] = None
        self._phase_start_frame: int = 0
        self._last_cycle_time_sec: Optional[float] = None
        self._prev_cycle_count: int = 0
        self._cycle_start_frame: int = 0

    def open(self) -> None:
        """Open the output video writer."""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = str(OUTPUTS_DIR / ANNOTATED_VIDEO_FILENAME)
        fourcc = cv2.VideoWriter_fourcc(*ANNOTATED_VIDEO_CODEC)
        self._writer = cv2.VideoWriter(
            output_path, fourcc, self._fps, (self._width, self._height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
        logger.info("Video writer opened: %s", output_path)

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: Optional[FrameDetections] = None,
        phase: Optional[Phase] = None,
        angles: Optional[JointAngles] = None,
        cycle_count: int = 0,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """
        Draw all annotations on a frame and write to output video.

        @param frame - BGR image to annotate
        @param detections - YOLO detections for this frame
        @param phase - Current cycle phase
        @param angles - Joint angles for this frame
        @param cycle_count - Running cycle counter
        @param frame_idx - Current frame index
        @returns Annotated frame
        """
        annotated = frame.copy()

        self._track_transitions(phase, cycle_count, frame_idx)

        if detections:
            self._draw_detections(annotated, detections)

        self._draw_cycle_hud(annotated, phase, angles, cycle_count, frame_idx)
        self._draw_info_bar(annotated, cycle_count, frame_idx)

        if self._writer is not None:
            self._writer.write(annotated)

        return annotated

    def close(self) -> None:
        """Release the video writer."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Annotated video saved")

    def _track_transitions(
        self, phase: Optional[Phase], cycle_count: int, frame_idx: int
    ) -> None:
        if phase is not None and phase != self._prev_phase:
            self._phase_start_frame = frame_idx
            self._prev_phase = phase

        if cycle_count > self._prev_cycle_count:
            cycle_duration = (frame_idx - self._cycle_start_frame) / self._fps
            if cycle_duration > 0:
                self._last_cycle_time_sec = cycle_duration
            self._cycle_start_frame = frame_idx
            self._prev_cycle_count = cycle_count

    def _draw_detections(self, frame: np.ndarray, detections: FrameDetections) -> None:
        for det in detections.detections:
            color = CLASS_COLOR_MAP.get(det.class_name, (128, 128, 128))
            pt1 = (int(det.x1), int(det.y1))
            pt2 = (int(det.x2), int(det.y2))
            cv2.rectangle(frame, pt1, pt2, color, 2)

            display_name = CLASS_DISPLAY_NAMES.get(det.class_name, det.class_name)
            label = f"{display_name} {det.confidence:.2f}"
            label_y = max(pt1[1] - 8, 16)
            cv2.putText(
                frame, label, (pt1[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    def _draw_cycle_hud(
        self,
        frame: np.ndarray,
        phase: Optional[Phase],
        angles: Optional[JointAngles],
        cycle_count: int,
        frame_idx: int,
    ) -> None:
        x0 = self._width - HUD_WIDTH - HUD_MARGIN
        y0 = HUD_MARGIN
        x1 = self._width - HUD_MARGIN
        y1 = y0 + HUD_HEIGHT

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, HUD_ALPHA, frame, 1 - HUD_ALPHA, 0, frame)

        cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)

        tx = x0 + 14
        ty = y0 + 32

        display_cycle = cycle_count if cycle_count > 0 else "-"
        cv2.putText(
            frame, f"CICLO {display_cycle}", (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLORS.text, 2,
        )
        ty += 38

        current_phase = phase or Phase.IDLE
        phase_color = PHASE_COLORS.get(current_phase, (128, 128, 128))
        phase_label = PHASE_LABELS.get(current_phase, "DESCONOCIDO")

        cv2.circle(frame, (tx + 8, ty - 6), 8, phase_color, -1)
        cv2.putText(
            frame, phase_label, (tx + 24, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_color, 2,
        )
        ty += 30

        phase_elapsed = (frame_idx - self._phase_start_frame) / self._fps
        cv2.putText(
            frame, f"Fase: {phase_elapsed:.1f}s", (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
        )
        ty += 24

        if self._last_cycle_time_sec is not None:
            cv2.putText(
                frame, f"Ultimo ciclo: {self._last_cycle_time_sec:.1f}s", (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
            )
            ty += 24

        if angles and angles.valid:
            for name, val in [("Brazo", angles.arm_angle_deg), ("Balde", angles.bucket_angle_deg)]:
                if val is not None:
                    cv2.putText(
                        frame, f"{name}: {val:.1f}\xb0", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1,
                    )
                    ty += 18

    def _draw_info_bar(self, frame: np.ndarray, cycle_count: int, frame_idx: int) -> None:
        time_sec = frame_idx / self._fps
        text = f"Ciclos: {cycle_count} | Tiempo: {time_sec:.1f}s | Cuadro: {frame_idx}"
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 30), (w, h), COLORS.phase_bg, -1)
        cv2.putText(
            frame, text, (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.text, 1,
        )
