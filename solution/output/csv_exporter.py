"""Per-frame YOLO detection CSV writer."""

import csv
import logging
from typing import Optional

from solution.config import OUTPUTS_DIR, DETECTIONS_CSV_FILENAME
from solution.detection.detector import FrameDetections

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "frame_idx",
    "timestamp_sec",
    "class_name",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "center_x",
    "center_y",
    "width",
    "height",
]


class DetectionCSVWriter:
    """
    Streams per-frame YOLO detections to a CSV file.

    Each row represents one bounding box detection. Frames with
    zero detections are silently skipped.
    """

    def __init__(self) -> None:
        self._file = None
        self._writer: Optional[csv.writer] = None
        self._row_count = 0

    def open(self) -> None:
        """Open the CSV file and write the header row."""
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUTS_DIR / DETECTIONS_CSV_FILENAME
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        logger.info("Detection CSV opened: %s", path)

    def write_frame(self, detections: FrameDetections, fps: float) -> None:
        """
        Write all detections for a single frame.

        @param detections - YOLO detections for one frame
        @param fps - Video frame rate (used to compute timestamp)
        """
        if self._writer is None:
            return

        if not detections.detections:
            return

        timestamp = detections.frame_idx / fps

        for det in detections.detections:
            cx, cy = det.center
            self._writer.writerow([
                detections.frame_idx,
                f"{timestamp:.4f}",
                det.class_name,
                det.class_id,
                f"{det.confidence:.4f}",
                f"{det.x1:.1f}",
                f"{det.y1:.1f}",
                f"{det.x2:.1f}",
                f"{det.y2:.1f}",
                f"{cx:.1f}",
                f"{cy:.1f}",
                f"{det.width:.1f}",
                f"{det.height:.1f}",
            ])
            self._row_count += 1

    def close(self) -> None:
        """Flush and close the CSV file."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            logger.info("Detection CSV closed: %d rows written", self._row_count)
