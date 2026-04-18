"""Draw detection overlays, phase labels, and metrics on video frames."""

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
from solution.detection.detector import FrameDetections, Detection
from solution.kinematics.cycle_detector import Phase
from solution.kinematics.joint_angles import JointAngles

logger = logging.getLogger(__name__)

CLASS_COLOR_MAP = {
    "bucket": COLORS.bucket,
    "arm_joint": COLORS.arm_joint,
    "boom": COLORS.boom,
    "truck": COLORS.truck,
}


class VideoAnnotator:
    """Draws bounding boxes, phase labels, and metrics onto video frames."""

    def __init__(self, fps: float = FPS, resolution: tuple = RESOLUTION) -> None:
        self._fps = fps
        self._width, self._height = resolution
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> None:
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

        @param frame - BGR image to annotate (modified in place)
        @param detections - YOLO detections for this frame
        @param phase - Current cycle phase
        @param angles - Joint angles for this frame
        @param cycle_count - Running cycle counter
        @param frame_idx - Current frame index
        @returns Annotated frame
        """
        annotated = frame.copy()

        if detections:
            self._draw_detections(annotated, detections)

        if phase:
            self._draw_phase_label(annotated, phase)

        if angles and angles.valid:
            self._draw_angles(annotated, angles)

        self._draw_info_bar(annotated, cycle_count, frame_idx)

        if self._writer is not None:
            self._writer.write(annotated)

        return annotated

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Annotated video saved")

    def _draw_detections(self, frame: np.ndarray, detections: FrameDetections) -> None:
        for det in detections.detections:
            color = CLASS_COLOR_MAP.get(det.class_name, (128, 128, 128))
            pt1 = (int(det.x1), int(det.y1))
            pt2 = (int(det.x2), int(det.y2))
            cv2.rectangle(frame, pt1, pt2, color, 2)

            label = f"{det.class_name} {det.confidence:.2f}"
            label_y = max(pt1[1] - 8, 16)
            cv2.putText(
                frame, label, (pt1[0], label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

    def _draw_phase_label(self, frame: np.ndarray, phase: Phase) -> None:
        text = f"Phase: {phase.name}"
        cv2.rectangle(frame, (10, 10), (250, 45), COLORS.phase_bg, -1)
        cv2.putText(
            frame, text, (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS.text, 2,
        )

    def _draw_angles(self, frame: np.ndarray, angles: JointAngles) -> None:
        y_offset = 60
        for name, val in [
            ("Arm", angles.arm_angle_deg),
            ("Bucket", angles.bucket_angle_deg),
            ("Boom", angles.boom_angle_deg),
        ]:
            if val is not None:
                text = f"{name}: {val:.1f} deg"
                cv2.putText(
                    frame, text, (15, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS.text, 1,
                )
                y_offset += 20

    def _draw_info_bar(self, frame: np.ndarray, cycle_count: int, frame_idx: int) -> None:
        time_sec = frame_idx / self._fps
        text = f"Cycles: {cycle_count} | Time: {time_sec:.1f}s | Frame: {frame_idx}"
        h = frame.shape[0]
        cv2.rectangle(frame, (0, h - 30), (400, h), COLORS.phase_bg, -1)
        cv2.putText(
            frame, text, (10, h - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.text, 1,
        )
