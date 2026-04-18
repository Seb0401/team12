"""
Central configuration for the EX-5600 shovel productivity analysis pipeline.

All paths, constants, thresholds, and machine specs live here.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, List


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUTS_DIR = Path("./inputs")
OUTPUTS_DIR = Path("./outputs")

LEFT_VIDEO = "shovel_left.mp4"
RIGHT_VIDEO = "shovel_right.mp4"
IMU_GLOB = "*imu*.npy"

# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

FPS = 15
RESOLUTION: Tuple[int, int] = (1280, 720)

# ---------------------------------------------------------------------------
# YOLO Detection
# ---------------------------------------------------------------------------

MODEL_PATH = Path("solution/detection/model/best.pt")
CONFIDENCE_THRESHOLD = 0.5
CLASS_NAMES: List[str] = ["arm_joint", "boom", "bucket", "truck"]

# ---------------------------------------------------------------------------
# Stereo Vision
# ---------------------------------------------------------------------------

STEREO_SAMPLE_RATE = 15  # compute depth every Nth frame (1/sec at 15fps)
STEREO_NUM_DISPARITIES = 128  # must be divisible by 16
STEREO_BLOCK_SIZE = 9
STEREO_MIN_DISPARITY = 0

# ---------------------------------------------------------------------------
# EX-5600 Machine Specs
# ---------------------------------------------------------------------------

BUCKET_VOLUME_M3 = 29.0
MATERIAL_DENSITY_TPM3 = 1.8  # typical rock density (tonnes/m³)
NOMINAL_PAYLOAD_TONNES = BUCKET_VOLUME_M3 * MATERIAL_DENSITY_TPM3  # ~52.2t

# ---------------------------------------------------------------------------
# IMU Processing
# ---------------------------------------------------------------------------

@dataclass
class IMUConfig:
    """IMU signal processing parameters."""

    accel_cols: Tuple[int, int, int] = (1, 2, 3)
    gyro_cols: Tuple[int, int, int] = (4, 5, 6)
    quat_cols: Tuple[int, int, int, int] = (7, 8, 9, 10)
    timestamp_col: int = 0

    accel_lowpass_cutoff_hz: float = 2.0
    gyro_bandpass_low_hz: float = 0.05
    gyro_bandpass_high_hz: float = 2.0
    sample_rate_hz: float = 15.0

    peak_min_distance_samples: int = 300  # ~20 sec between swing peaks (real cycles 30-60s)
    peak_min_prominence: float = 0.8  # gyro magnitude prominence (lowered for bandpass signal)


IMU_CONFIG = IMUConfig()

# ---------------------------------------------------------------------------
# Cycle Detection FSM
# ---------------------------------------------------------------------------

@dataclass
class CycleFSMConfig:
    """Finite state machine thresholds for cycle phase transitions."""

    angle_change_threshold_deg: float = 5.0
    idle_max_angle_rate_deg_s: float = 2.0
    dig_min_arm_curl_rate_deg_s: float = 3.0
    swing_min_gyro_magnitude: float = 1.0
    dump_bucket_tilt_threshold_deg: float = 120.0
    min_phase_duration_sec: float = 2.0
    max_cycle_duration_sec: float = 120.0
    min_cycle_duration_sec: float = 15.0
    max_phase_duration_sec: float = 25.0
    angle_smoothing_window: int = 5


CYCLE_FSM_CONFIG = CycleFSMConfig()

# ---------------------------------------------------------------------------
# Bucket-Position Phase Detection (simple CV rules)
# ---------------------------------------------------------------------------

@dataclass
class BucketPhaseConfig:
    """Phase detection based on bucket Y-position, X-velocity, and truck presence."""

    bucket_low_y: float = 350.0
    bucket_high_y: float = 250.0
    smoothing_window: int = 8
    min_phase_frames: int = 15
    x_velocity_threshold: float = 3.0
    x_smoothing_window: int = 5
    min_dump_duration_sec: float = 3.0
    min_idle_duration_sec: float = 4.0


BUCKET_PHASE_CONFIG = BucketPhaseConfig()

# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------

FUSION_TOLERANCE_SEC = 0.5  # max offset between IMU and video cycle events

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

ANNOTATED_VIDEO_FILENAME = "annotated_video.mp4"
METRICS_JSON_FILENAME = "metrics.json"
DETECTIONS_CSV_FILENAME = "detections.csv"
ANNOTATED_VIDEO_CODEC = "mp4v"

# ---------------------------------------------------------------------------
# Annotation Colors (BGR for OpenCV)
# ---------------------------------------------------------------------------

@dataclass
class AnnotationColors:
    """BGR color palette for video annotations."""

    bucket: Tuple[int, int, int] = (0, 255, 0)      # green
    arm_joint: Tuple[int, int, int] = (0, 255, 255)  # yellow
    boom: Tuple[int, int, int] = (255, 0, 0)         # blue
    truck: Tuple[int, int, int] = (0, 165, 255)      # orange
    text: Tuple[int, int, int] = (255, 255, 255)     # white
    phase_bg: Tuple[int, int, int] = (0, 0, 0)       # black background


COLORS = AnnotationColors()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure structured logging for the pipeline.

    @param level - Logging level (default INFO)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
