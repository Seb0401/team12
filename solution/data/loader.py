"""Data loading utilities for video and IMU inputs."""

import glob
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np

from solution.config import INPUTS_DIR, LEFT_VIDEO, RIGHT_VIDEO, IMU_GLOB

logger = logging.getLogger(__name__)

EXPECTED_IMU_COLS = 11


def _find_video(filename: str) -> Path:
    """
    Locate a video file in the inputs directory, falling back to glob patterns.

    @param filename - Expected video filename (e.g. 'shovel_left.mp4')
    @returns Resolved path to the video file
    @raises FileNotFoundError if no matching video found
    """
    exact = INPUTS_DIR / filename
    if exact.exists():
        return exact

    side = "left" if "left" in filename else "right"
    pattern = str(INPUTS_DIR / f"*_{side}.mp4")
    matches = glob.glob(pattern)
    if matches:
        path = Path(matches[0])
        logger.info("Fallback match for %s: %s", filename, path.name)
        return path

    raise FileNotFoundError(
        f"Video '{filename}' not found in {INPUTS_DIR}. "
        f"Glob '{pattern}' also matched nothing."
    )


def _find_imu() -> Path:
    """
    Locate the IMU .npy file in the inputs directory.

    @returns Resolved path to the IMU numpy file
    @raises FileNotFoundError if no matching file found
    """
    pattern = str(INPUTS_DIR / IMU_GLOB)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(
            f"No IMU file matching '{IMU_GLOB}' in {INPUTS_DIR}"
        )
    if len(matches) > 1:
        logger.warning("Multiple IMU files found, using first: %s", matches[0])
    return Path(matches[0])


def load_left_video() -> cv2.VideoCapture:
    """
    Open the left stereo camera video.

    @returns OpenCV VideoCapture for the left video
    @raises FileNotFoundError if video file not found
    @raises RuntimeError if video cannot be opened
    """
    path = _find_video(LEFT_VIDEO)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    logger.info("Loaded left video: %s", path.name)
    return cap


def load_right_video() -> cv2.VideoCapture:
    """
    Open the right stereo camera video.

    @returns OpenCV VideoCapture for the right video
    @raises FileNotFoundError if video file not found
    @raises RuntimeError if video cannot be opened
    """
    path = _find_video(RIGHT_VIDEO)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    logger.info("Loaded right video: %s", path.name)
    return cap


def find_imu_path() -> Path:
    """
    Find the IMU .npy file path without loading it.

    @returns Path to the IMU .npy file
    @raises FileNotFoundError if no matching file found
    """
    return _find_imu()


def load_imu_data() -> np.ndarray:
    """
    Load IMU data from the .npy file and validate its shape.

    Expected shape: (N, 11) where columns are:
    [timestamp_ns, ax, ay, az, gx, gy, gz, qw, qx, qy, qz]

    @returns IMU data array with shape (N, 11)
    @raises FileNotFoundError if IMU file not found
    @raises ValueError if data shape is invalid
    """
    path = _find_imu()
    data = np.load(str(path))

    if data.ndim != 2 or data.shape[1] != EXPECTED_IMU_COLS:
        raise ValueError(
            f"IMU data shape {data.shape} invalid. "
            f"Expected (N, {EXPECTED_IMU_COLS})."
        )

    logger.info(
        "Loaded IMU: %d samples, %.1f sec duration",
        data.shape[0],
        (data[-1, 0] - data[0, 0]) / 1e9,
    )
    return data


def get_video_metadata(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """
    Extract metadata from an opened VideoCapture.

    @param cap - OpenCV VideoCapture object
    @returns Dict with keys: fps, frame_count, duration_sec, width, height
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = frame_count / fps if fps > 0 else 0.0

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
        "width": width,
        "height": height,
    }


def validate_inputs() -> bool:
    """
    Check that all required input files exist.

    @returns True if all inputs are present, False otherwise
    """
    try:
        _find_video(LEFT_VIDEO)
        _find_video(RIGHT_VIDEO)
        _find_imu()
        return True
    except FileNotFoundError as exc:
        logger.error("Input validation failed: %s", exc)
        return False
