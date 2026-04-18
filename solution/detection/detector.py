"""YOLOv8 inference wrapper for shovel component detection."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from solution.config import MODEL_PATH, CONFIDENCE_THRESHOLD, CLASS_NAMES

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single bounding box detection result."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        """Bounding box center point (cx, cy)."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class FrameDetections:
    """All detections for a single frame."""

    frame_idx: int
    detections: List[Detection] = field(default_factory=list)

    def get_by_class(self, class_name: str) -> Optional[Detection]:
        """
        Get highest-confidence detection for a given class.

        @param class_name - One of: bucket, arm_joint, boom
        @returns Best detection or None if class not found
        """
        matches = [d for d in self.detections if d.class_name == class_name]
        if not matches:
            return None
        return max(matches, key=lambda d: d.confidence)

    REQUIRED_JOINTS = {"bucket", "arm_joint"}

    @property
    def has_all_joints(self) -> bool:
        """True if bucket and arm_joint both detected."""
        found = {d.class_name for d in self.detections}
        return self.REQUIRED_JOINTS.issubset(found)


class ShovelDetector:
    """
    YOLOv8-based detector for shovel components.

    Wraps ultralytics YOLO model for inference on video frames.
    Detects: bucket, arm_joint, boom.
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        confidence: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        """
        @param model_path - Path to fine-tuned YOLOv8 .pt weights
        @param confidence - Minimum detection confidence threshold
        """
        self._model_path = model_path
        self._confidence = confidence
        self._model = None

    def load(self) -> None:
        """Load the YOLO model into memory. Call once before detect()."""
        from ultralytics import YOLO

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {self._model_path}. "
                "Run training first (see SPEC.md §14)."
            )

        self._model = YOLO(str(self._model_path))
        logger.info("YOLO model loaded: %s", self._model_path.name)

    def detect(self, frame: np.ndarray, frame_idx: int = 0) -> FrameDetections:
        """
        Run inference on a single BGR frame.

        @param frame - BGR image array (H, W, 3)
        @param frame_idx - Frame index for tracking
        @returns FrameDetections with all detections above confidence threshold
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self._model(frame, conf=self._confidence, verbose=False)
        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()

                cls_name = (
                    CLASS_NAMES[cls_id]
                    if cls_id < len(CLASS_NAMES)
                    else f"unknown_{cls_id}"
                )

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        x1=x1, y1=y1,
                        x2=x2, y2=y2,
                    )
                )

        return FrameDetections(frame_idx=frame_idx, detections=detections)
