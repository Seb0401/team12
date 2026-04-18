"""Alignment viewer: video with YOLO detections + synchronized IMU 3D signals.

Layout:
    ┌─────────────────────────────────┐
    │  Video + YOLO bounding boxes    │  720px
    ├─────────────────────────────────┤
    │  Info bar (frame/sample/values) │   30px
    ├─────────────────────────────────┤
    │  Gyro  X / Y / Z  (3 axes)     │  150px
    ├─────────────────────────────────┤
    │  Accel X / Y / Z  (3 axes)     │  150px
    └─────────────────────────────────┘

Usage:
    .venv/bin/python3 -m solution.alignment_viewer [--speed N] [--no-yolo]

Controls:
    SPACE  - Pause / resume
    Q/ESC  - Quit
    +/=    - Speed up (2x steps)
    -      - Slow down
    F      - Step forward one frame (while paused)
    R      - Restart from beginning
    G      - Jump to next gyro peak
    D      - Toggle YOLO detection overlay
"""

import argparse
import logging
import time
from typing import Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from solution.config import setup_logging, COLORS
from solution.data.loader import load_left_video, load_imu_data, get_video_metadata
from solution.detection.detector import ShovelDetector, FrameDetections
from solution.utils.time_sync import (
    AlignmentResult,
    frame_to_imu_sample,
    imu_sample_to_frame,
    DEFAULT_OFFSET_SAMPLES,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "EX-5600 Alignment Viewer"
STRIP_HEIGHT = 150
STRIP_WIDTH = 1280
VIDEO_HEIGHT = 720
PEAK_JUMP_MIN_PROMINENCE = 30.0

CLASS_COLOR_MAP = {
    "bucket": COLORS.bucket,
    "arm_joint": COLORS.arm_joint,
    "boom": COLORS.boom,
    "truck": COLORS.truck,
}

GYRO_COLORS = ["#00ddff", "#44ff88", "#ff44dd"]
ACCEL_COLORS = ["#ff6644", "#ffcc00", "#aa88ff"]
AXIS_LABELS = ["X", "Y", "Z"]


def _prerender_imu_strip(
    signal: np.ndarray,
    total_samples: int,
    colors: list,
    title: str,
) -> tuple[np.ndarray, int, int]:
    """Pre-render 3-axis IMU signal as BGR image.

    @param signal - IMU signal array, shape (N, 3)
    @param total_samples - Total sample count
    @param colors - List of 3 hex color strings for X/Y/Z
    @param title - Y-axis label prefix ('Gyro' or 'Accel')
    @returns Tuple of (BGR image, plot_x_left_px, plot_x_right_px) where the
             pixel offsets define the data area within the resized image
    """
    fig, axes = plt.subplots(
        3, 1, figsize=(STRIP_WIDTH / 100, STRIP_HEIGHT / 100),
        dpi=100, sharex=True,
    )
    fig.patch.set_facecolor("#1a1a1a")
    x = np.arange(total_samples)

    for i, ax in enumerate(axes):
        ax.plot(x, signal[:, i], color=colors[i], linewidth=0.4, alpha=0.9)
        ax.set_ylabel(f"{title} {AXIS_LABELS[i]}", color="#aaaaaa", fontsize=6)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#666666", labelsize=5)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, total_samples)

    plt.tight_layout(pad=0.2)
    fig.canvas.draw()

    renderer = fig.canvas.get_renderer()
    ax_ref = axes[0]
    bbox = ax_ref.get_window_extent(renderer)
    fig_w, fig_h = fig.canvas.get_width_height()
    plot_left_frac = bbox.x0 / fig_w
    plot_right_frac = bbox.x1 / fig_w

    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    strip_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    strip_bgr = cv2.resize(strip_bgr, (STRIP_WIDTH, STRIP_HEIGHT))

    plot_x_left = int(plot_left_frac * STRIP_WIDTH)
    plot_x_right = int(plot_right_frac * STRIP_WIDTH)

    return strip_bgr, plot_x_left, plot_x_right


def _draw_time_marker(
    strip: np.ndarray,
    sample_idx: int,
    total_samples: int,
    plot_x_left: int,
    plot_x_right: int,
) -> np.ndarray:
    """Draw vertical green time marker on strip copy, within plot data area.

    @param strip - Pre-rendered strip image (BGR)
    @param sample_idx - Current IMU sample index
    @param total_samples - Total samples for scaling
    @param plot_x_left - Left pixel boundary of matplotlib data area
    @param plot_x_right - Right pixel boundary of matplotlib data area
    @returns Copy with marker drawn
    """
    marked = strip.copy()
    plot_width = plot_x_right - plot_x_left
    x_pos = plot_x_left + int(sample_idx / total_samples * plot_width)
    x_pos = max(plot_x_left, min(x_pos, plot_x_right - 1))
    cv2.line(marked, (x_pos, 0), (x_pos, marked.shape[0]), (0, 255, 0), 2)
    return marked


def _draw_detections(frame: np.ndarray, detections: FrameDetections) -> None:
    """Draw YOLO bounding boxes and labels on frame in-place.

    @param frame - BGR image (modified in-place)
    @param detections - YOLO detections for this frame
    """
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


def _draw_info_bar(
    frame_idx: int,
    sample_idx: int,
    total_frames: int,
    fps: float,
    gyro_xyz: np.ndarray,
    accel_xyz: np.ndarray,
    paused: bool,
    speed: float,
    yolo_enabled: bool,
    det_count: int,
) -> np.ndarray:
    """Render info bar showing sync state and current IMU values.

    @param frame_idx - Current video frame index
    @param sample_idx - Mapped IMU sample index
    @param total_frames - Total frames in video
    @param fps - Video FPS
    @param gyro_xyz - Current gyro values (3,)
    @param accel_xyz - Current accel values (3,)
    @param paused - Whether playback is paused
    @param speed - Playback speed multiplier
    @param yolo_enabled - Whether YOLO overlay is active
    @param det_count - Number of detections in current frame
    @returns BGR image (30, STRIP_WIDTH, 3)
    """
    bar = np.zeros((30, STRIP_WIDTH, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    video_t = frame_idx / fps
    text_left = (
        f"Frame {frame_idx}/{total_frames}  "
        f"t={video_t:.1f}s  "
        f"IMU#{sample_idx}  "
        f"G=({gyro_xyz[0]:.1f},{gyro_xyz[1]:.1f},{gyro_xyz[2]:.1f})  "
        f"A=({accel_xyz[0]:.1f},{accel_xyz[1]:.1f},{accel_xyz[2]:.1f})"
    )
    cv2.putText(bar, text_left, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    status_parts = []
    if paused:
        status_parts.append("PAUSED")
    if speed != 1.0:
        status_parts.append(f"{speed:.1f}x")
    if yolo_enabled:
        status_parts.append(f"YOLO({det_count})")
    else:
        status_parts.append("YOLO OFF")

    status = "  ".join(status_parts)
    cv2.putText(bar, status, (STRIP_WIDTH - 280, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 200, 255), 1)

    return bar


def _find_gyro_peaks(gyro_mag: np.ndarray) -> np.ndarray:
    """Find prominent gyro peaks for G-key jump navigation.

    @param gyro_mag - Gyroscope magnitude array
    @returns Sorted array of peak sample indices
    """
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(gyro_mag, distance=100, prominence=PEAK_JUMP_MIN_PROMINENCE)
    return peaks


def _build_composite(
    frame: np.ndarray,
    info_bar: np.ndarray,
    gyro_strip_marked: np.ndarray,
    accel_strip_marked: np.ndarray,
) -> np.ndarray:
    """Stack video + info + gyro strip + accel strip vertically.

    @param frame - Resized video frame (VIDEO_HEIGHT, STRIP_WIDTH, 3)
    @param info_bar - Info bar (30, STRIP_WIDTH, 3)
    @param gyro_strip_marked - Gyro strip with marker (STRIP_HEIGHT, STRIP_WIDTH, 3)
    @param accel_strip_marked - Accel strip with marker (STRIP_HEIGHT, STRIP_WIDTH, 3)
    @returns Composite BGR image
    """
    return np.vstack([frame, info_bar, gyro_strip_marked, accel_strip_marked])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EX-5600 Alignment Viewer: video + YOLO + IMU 3D")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed (default: 1.0)")
    parser.add_argument("--no-yolo", action="store_true", help="Start with YOLO disabled")
    return parser.parse_args()


def run_alignment_viewer() -> None:
    setup_logging()
    args = parse_args()

    logger.info("Loading video...")
    cap = load_left_video()
    meta = get_video_metadata(cap)
    total_frames = meta["frame_count"]
    fps = meta["fps"]

    logger.info("Loading IMU data...")
    imu_raw = load_imu_data()
    accel_raw = imu_raw[:, 1:4]
    gyro_raw = imu_raw[:, 4:7]
    gyro_mag = np.linalg.norm(gyro_raw, axis=1)
    total_samples = len(imu_raw)

    alignment = AlignmentResult(
        offset_samples=DEFAULT_OFFSET_SAMPLES,
        offset_sec=DEFAULT_OFFSET_SAMPLES / fps,
        correlation_score=0.0,
        method="default",
        segment_offsets=[],
    )

    logger.info("Loading YOLO model...")
    detector: Optional[ShovelDetector] = None
    try:
        detector = ShovelDetector()
        detector.load()
        logger.info("YOLO model loaded.")
    except (FileNotFoundError, Exception) as exc:
        logger.warning("YOLO model not available: %s. Running without detections.", exc)
        detector = None

    logger.info("Pre-rendering IMU strips (6 axes)...")
    gyro_strip, gyro_plot_left, gyro_plot_right = _prerender_imu_strip(
        gyro_raw, total_samples, GYRO_COLORS, "Gyro",
    )
    accel_strip, accel_plot_left, accel_plot_right = _prerender_imu_strip(
        accel_raw, total_samples, ACCEL_COLORS, "Accel",
    )

    gyro_peaks = _find_gyro_peaks(gyro_mag)
    logger.info("Found %d gyro peaks for jump nav (G key)", len(gyro_peaks))

    speed = args.speed
    paused = False
    yolo_enabled = not args.no_yolo and detector is not None
    frame_idx = 0

    total_height = VIDEO_HEIGHT + 30 + STRIP_HEIGHT * 2
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, STRIP_WIDTH, total_height)

    logger.info(
        "Viewer ready. SPACE=pause Q=quit +/-=speed F=step G=peak D=toggle_yolo R=restart"
    )
    target_frame_time = 1.0 / fps

    def _render_frame(frame: np.ndarray, fidx: int) -> np.ndarray:
        frame_resized = cv2.resize(frame, (STRIP_WIDTH, VIDEO_HEIGHT))

        det_count = 0
        if yolo_enabled and detector is not None:
            detections = detector.detect(frame, fidx)
            _draw_detections(frame_resized, detections)
            det_count = len(detections.detections)

        sample_idx = frame_to_imu_sample(fidx, alignment)
        sample_idx = max(0, min(sample_idx, total_samples - 1))

        gyro_marked = _draw_time_marker(gyro_strip, sample_idx, total_samples, gyro_plot_left, gyro_plot_right)
        accel_marked = _draw_time_marker(accel_strip, sample_idx, total_samples, accel_plot_left, accel_plot_right)

        info = _draw_info_bar(
            fidx, sample_idx, total_frames, fps,
            gyro_raw[sample_idx], accel_raw[sample_idx],
            paused, speed, yolo_enabled, det_count,
        )

        return _build_composite(frame_resized, info, gyro_marked, accel_marked)

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of video.")
                break

            t0 = time.monotonic()
            composite = _render_frame(frame, frame_idx)
            cv2.imshow(WINDOW_NAME, composite)
            frame_idx += 1

            elapsed = time.monotonic() - t0
            wait_ms = max(1, int((target_frame_time / speed - elapsed) * 1000))
        else:
            wait_ms = 30

        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            speed = min(speed * 2, 16.0)
            logger.info("Speed: %.1fx", speed)
        elif key == ord("-"):
            speed = max(speed / 2, 0.125)
            logger.info("Speed: %.1fx", speed)
        elif key in (ord("d"), ord("D")):
            if detector is not None:
                yolo_enabled = not yolo_enabled
                logger.info("YOLO: %s", "ON" if yolo_enabled else "OFF")
            else:
                logger.info("YOLO model not loaded — cannot toggle.")
        elif key in (ord("f"), ord("F")) and paused:
            ret, frame = cap.read()
            if ret and frame is not None:
                composite = _render_frame(frame, frame_idx)
                cv2.imshow(WINDOW_NAME, composite)
                frame_idx += 1
        elif key in (ord("g"), ord("G")):
            current_sample = frame_to_imu_sample(frame_idx, alignment)
            future_peaks = gyro_peaks[gyro_peaks > current_sample + 5]
            if len(future_peaks) > 0:
                target_sample = future_peaks[0]
                target_frame = imu_sample_to_frame(target_sample, alignment)
                target_frame = max(0, min(target_frame, total_frames - 1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_idx = target_frame
                logger.info("Jumped to peak at sample %d (frame %d)", target_sample, target_frame)
            else:
                logger.info("No more peaks ahead.")
        elif key in (ord("r"), ord("R")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            logger.info("Restarted.")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Viewer closed at frame %d.", frame_idx)


if __name__ == "__main__":
    run_alignment_viewer()
