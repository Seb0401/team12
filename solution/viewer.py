"""
Standalone realtime video viewer with YOLO detection overlays.

Displays the raw video with bounding boxes, joint angles, and cycle phase
annotations in a live OpenCV window. No full pipeline preprocessing required —
frames are processed and displayed on the fly.

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
from typing import Optional

import cv2
import numpy as np

from solution.config import setup_logging, COLORS
from solution.data.loader import load_left_video, load_right_video, get_video_metadata
from solution.detection.detector import ShovelDetector, FrameDetections
from solution.kinematics.joint_angles import compute_joint_angles, JointAngles
from solution.kinematics.cycle_detector import CycleFSM, Phase

logger = logging.getLogger(__name__)

WINDOW_NAME = "EX-5600 Visor en Tiempo Real"

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
    phase: Phase,
    angles: Optional[JointAngles],
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
    @param phase - Current cycle phase
    @param angles - Current joint angles
    @param cycle_count - Running cycle counter
    @param frame_idx - Current frame index
    @param fps - Video frames per second
    @param paused - Whether playback is paused
    @param speed - Current playback speed multiplier
    @param show_detections - Whether detection overlay is enabled
    """
    h, w = frame.shape[:2]

    hud_w, hud_h = 300, 200
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

    phase_color = PHASE_COLORS.get(phase, (128, 128, 128))
    phase_label = PHASE_LABELS.get(phase, "UNKNOWN")
    cv2.circle(frame, (tx + 8, ty - 6), 8, phase_color, -1)
    cv2.putText(
        frame, phase_label, (tx + 24, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2,
    )
    ty += 28

    if angles and angles.valid:
        for name, val in [("Brazo", angles.arm_angle_deg), ("Balde", angles.bucket_angle_deg)]:
            if val is not None:
                cv2.putText(
                    frame, f"{name}: {val:.1f} grados", (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1,
                )
                ty += 20

    ty += 4
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
    h = frame.shape[0]
    hint = "ESPACIO:Pausa  Q:Salir  +/-:Velocidad  D:Detecciones  F:Avanzar  R:Reiniciar"
    cv2.rectangle(frame, (0, h - 28), (frame.shape[1], h), (0, 0, 0), -1)
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


def run_viewer() -> None:
    """
    Main viewer loop: load video + model, display frames with annotations.

    Handles keyboard controls for pause, speed, detection toggle, etc.
    """
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

    fsm = CycleFSM()

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

            detections = detector.detect(frame, frame_idx)
            angles = compute_joint_angles(detections)
            truck_visible = detections.get_by_class("truck") is not None
            phase = fsm.update(angles, truck_visible=truck_visible)
            cycle_count = len(fsm.cycles)

            display = frame.copy()

            if show_detections:
                draw_detections(display, detections)

            draw_hud(
                display, phase, angles, cycle_count, frame_idx,
                fps, paused, speed, show_detections,
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
                detections = detector.detect(frame, frame_idx)
                angles = compute_joint_angles(detections)
                truck_visible = detections.get_by_class("truck") is not None
                phase = fsm.update(angles, truck_visible=truck_visible)
                cycle_count = len(fsm.cycles)

                display = frame.copy()
                if show_detections:
                    draw_detections(display, detections)
                draw_hud(
                    display, phase, angles, cycle_count, frame_idx,
                    fps, paused, speed, show_detections,
                )
                draw_controls_hint(display)
                cv2.imshow(WINDOW_NAME, display)
                frame_idx += 1
        elif key in (ord("r"), ord("R")):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = 0
            fsm = CycleFSM()
            logger.info("Restarted from beginning")

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Viewer closed. Processed %d frames.", frame_idx)


if __name__ == "__main__":
    run_viewer()
