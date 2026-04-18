"""
Standalone realtime video viewer with YOLO detection overlays.

Uses bucket Y-position + truck visibility for phase classification.

Usage:
    python -m solution.viewer [--video left|right] [--speed MULTIPLIER]

Controls:
    SPACE  — Pause / resume
    Q/ESC  — Quit
    +/=    — Speed up (2×, 4×, 8×)
    -      — Slow down
    D      — Toggle detection overlay
    F      — Step forward one frame (while paused)
    R      — Restart from beginning
"""

import argparse
import logging
import time

import cv2
import numpy as np

from solution.config import setup_logging, COLORS
from solution.data.loader import load_left_video, load_right_video, get_video_metadata
from solution.detection.detector import ShovelDetector, FrameDetections
from solution.kinematics.bucket_phase import (
    BucketPhaseDetector,
    BucketPhase,
    PHASE_DISPLAY,
    PHASE_COLORS_BGR,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "EX-5600 Visor en Tiempo Real"

CLASS_DISPLAY_NAMES = {
    "bucket": "balde",
    "arm_joint": "articulacion",
    "boom": "pluma",
    "truck": "camion",
}

CLASS_COLOR_MAP = {
    "bucket": COLORS.bucket,
    "arm_joint": COLORS.arm_joint,
    "boom": COLORS.boom,
    "truck": COLORS.truck,
}


def draw_detections(frame: np.ndarray, detections: FrameDetections) -> None:
    """
    Draw bounding boxes and class labels on the frame in-place.

    @param frame - BGR image to annotate (modified in-place)
    @param detections - YOLO detections for this frame
    """
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


def draw_hud(
    frame: np.ndarray,
    phase: BucketPhase,
    bucket_y: float,
    truck_visible: bool,
    x_velocity: float,
    idle_alert: bool,
    idle_duration: float,
    cycle_count: int,
    frame_idx: int,
    fps: float,
    paused: bool,
    speed: float,
    show_detections: bool,
) -> None:
    """
    Draw the heads-up display panel on the frame.

    @param frame - BGR image to annotate (modified in-place)
    @param phase - Current bucket phase
    @param bucket_y - Current smoothed bucket Y position
    @param truck_visible - Whether truck is detected
    @param x_velocity - Smoothed bucket X-axis velocity (px/frame)
    @param idle_alert - Whether idle alert is active (>= 4s)
    @param idle_duration - Current idle duration in seconds
    @param cycle_count - Running cycle counter
    @param frame_idx - Current frame index
    @param fps - Video frames per second
    @param paused - Whether playback is paused
    @param speed - Current playback speed multiplier
    @param show_detections - Whether detection overlay is enabled
    """
    h, w = frame.shape[:2]

    hud_w, hud_h = 320, 270
    x0, y0 = w - hud_w - 10, 10
    x1, y1 = w - 10, y0 + hud_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (80, 80, 80), 1)

    tx, ty = x0 + 14, y0 + 28

    time_sec = frame_idx / fps
    cv2.putText(
        frame, f"T: {time_sec:.1f}s  Cuadro: {frame_idx}", (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1,
    )
    ty += 28

    display_cycle = cycle_count if cycle_count > 0 else "-"
    cv2.putText(
        frame, f"CICLO {display_cycle}", (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS.text, 2,
    )
    ty += 32

    phase_color = PHASE_COLORS_BGR.get(phase, (128, 128, 128))
    phase_label = PHASE_DISPLAY.get(phase, "DESCONOCIDO")
    cv2.circle(frame, (tx + 8, ty - 6), 8, phase_color, -1)
    cv2.putText(
        frame, phase_label, (tx + 24, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2,
    )
    ty += 28

    cv2.putText(
        frame, f"Balde Y: {bucket_y:.0f}px", (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
    )
    ty += 20

    vel_color = (0, 200, 200) if x_velocity > 3.0 else (160, 160, 160)
    cv2.putText(
        frame, f"Vel X: {x_velocity:.1f} px/f", (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, vel_color, 1,
    )
    ty += 20

    truck_text = "Camion: SI" if truck_visible else "Camion: NO"
    truck_color = (0, 200, 0) if truck_visible else (0, 0, 200)
    cv2.putText(
        frame, truck_text, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, truck_color, 1,
    )
    ty += 24

    if idle_alert:
        alert_text = f"ALERTA INACTIVO: {idle_duration:.1f}s"
        cv2.putText(
            frame, alert_text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
        )
        ty += 24

        banner_h = 40
        overlay_banner = frame.copy()
        cv2.rectangle(overlay_banner, (0, h // 2 - banner_h // 2), (w, h // 2 + banner_h // 2), (0, 0, 180), -1)
        cv2.addWeighted(overlay_banner, 0.5, frame, 0.5, 0, frame)
        cv2.putText(
            frame, f"INACTIVO {idle_duration:.1f}s", (w // 2 - 120, h // 2 + 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    status_parts = []
    if paused:
        status_parts.append("PAUSADO")
    if speed != 1.0:
        status_parts.append(f"{speed:.1f}x")
    if not show_detections:
        status_parts.append("DET DESACT")
    if status_parts:
        status_text = " | ".join(status_parts)
        cv2.putText(
            frame, status_text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1,
        )


def draw_controls_hint(frame: np.ndarray) -> None:
    """
    Draw a small control legend at the bottom of the frame.

    @param frame - BGR image to annotate (modified in-place)
    """
    h, w = frame.shape[:2]
    hint = "ESPACIO:Pausa  Q:Salir  +/-:Velocidad  D:Detecciones  F:Avanzar  R:Reiniciar"
    cv2.rectangle(frame, (0, h - 28), (w, h), (0, 0, 0), -1)
    cv2.putText(
        frame, hint, (10, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1,
    )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the viewer.

    @returns Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Realtime video viewer with YOLO detection overlays"
    )
    parser.add_argument(
        "--video", choices=["left", "right"], default="left",
        help="Which stereo camera to view (default: left)",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Initial playback speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--no-detect", action="store_true",
        help="Start with detection overlay disabled",
    )
    return parser.parse_args()


def _process_frame(
    frame: np.ndarray,
    frame_idx: int,
    detector: ShovelDetector,
    phase_detector: BucketPhaseDetector,
) -> tuple:
    detections = detector.detect(frame, frame_idx)
    phase = phase_detector.update(detections)
    cycle_count = phase_detector.cycle_count
    x_velocity = phase_detector.x_velocity
    idle_alert = phase_detector.is_idle_alert
    idle_duration = phase_detector.idle_duration_sec

    bucket = detections.get_by_class("bucket")
    bucket_y = bucket.center[1] if bucket else 0.0
    truck_visible = detections.get_by_class("truck") is not None

    return detections, phase, cycle_count, bucket_y, truck_visible, x_velocity, idle_alert, idle_duration


def run_viewer() -> None:
    """Main viewer loop: load video + model, display frames with annotations."""
    setup_logging()
    args = parse_args()

    logger.info("Loading video (%s)...", args.video)
    cap = load_left_video() if args.video == "left" else load_right_video()
    meta = get_video_metadata(cap)
    total_frames = meta["frame_count"]
    fps = meta["fps"]

    logger.info(
        "Video: %d frames, %.1f fps, %.1f sec",
        total_frames, fps, meta["duration_sec"],
    )

    logger.info("Loading YOLO model...")
    detector = ShovelDetector()
    detector.load()

    phase_detector = BucketPhaseDetector()

    speed = args.speed
    paused = False
    show_detections = not args.no_detect
    frame_idx = 0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, meta["width"], meta["height"])

    logger.info("Viewer ready. Press SPACE to pause, Q to quit.")

    target_frame_time = 1.0 / fps

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.info("End of video reached.")
                break

            t0 = time.monotonic()

            detections, phase, cycle_count, bucket_y, truck_visible, x_velocity, idle_alert, idle_duration = _process_frame(
                frame, frame_idx, detector, phase_detector,
            )

            display = frame.copy()

            if show_detections:
                draw_detections(display, detections)

            draw_hud(
                display, phase, bucket_y, truck_visible, x_velocity,
                idle_alert, idle_duration,
                cycle_count, frame_idx, fps, paused, speed, show_detections,
            )
            draw_controls_hint(display)

            cv2.imshow(WINDOW_NAME, display)
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
            logger.info("Paused" if paused else "Resumed")
        elif key in (ord("+"), ord("=")):
            speed = min(speed * 2, 16.0)
            logger.info("Speed: %.1fx", speed)
        elif key == ord("-"):
            speed = max(speed / 2, 0.125)
            logger.info("Speed: %.1fx", speed)
        elif key in (ord("d"), ord("D")):
            show_detections = not show_detections
            logger.info("Detections: %s", "ON" if show_detections else "OFF")
        elif key in (ord("f"), ord("F")) and paused:
            ret, frame = cap.read()
            if ret and frame is not None:
                detections, phase, cycle_count, bucket_y, truck_visible, x_velocity, idle_alert, idle_duration = _process_frame(
                    frame, frame_idx, detector, phase_detector,
                )

                display = frame.copy()
                if show_detections:
                    draw_detections(display, detections)
                draw_hud(
                    display, phase, bucket_y, truck_visible, x_velocity,
                    idle_alert, idle_duration,
                    cycle_count, frame_idx, fps, paused, speed, show_detections,
                )
                draw_controls_hint(display)
                cv2.imshow(WINDOW_NAME, display)
                frame_idx += 1
        elif key in (ord("r"), ord("R")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            phase_detector = BucketPhaseDetector()
            logger.info("Restarted from beginning")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Viewer closed. Processed %d frames.", frame_idx)


if __name__ == "__main__":
    run_viewer()
