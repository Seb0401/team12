"""Main pipeline orchestrator for EX-5600 shovel productivity analysis."""

import logging
import time
from typing import List, Optional

import cv2
import numpy as np

from solution.config import (
    setup_logging,
    FPS,
    STEREO_SAMPLE_RATE,
    OUTPUTS_DIR,
)
from solution.data.loader import (
    load_left_video,
    load_right_video,
    load_imu_data,
    get_video_metadata,
    validate_inputs,
)
from solution.detection.detector import ShovelDetector, FrameDetections
from solution.kinematics.joint_angles import compute_joint_angles, JointAngles
from solution.kinematics.cycle_detector import CycleFSM, Phase
from solution.imu.processor import process_imu
from solution.imu.cycle_detector import detect_swing_peaks, pair_swing_events
from solution.imu.fusion import fuse_cycles
from solution.stereo.depth import StereoDepthEstimator
from solution.stereo.volume import estimate_bucket_fill
from solution.stereo.kinematics_3d import estimate_3d_kinematics
from solution.productivity.metrics import compute_productivity
from solution.productivity.recommendations import generate_recommendations
from solution.output.annotator import VideoAnnotator
from solution.output.report import write_json_report
from solution.utils.time_sync import align_imu_to_video

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the full analysis pipeline: load → detect → analyze → output."""
    setup_logging()
    start_time = time.time()

    logger.info("=== EX-5600 Shovel Productivity Analysis ===")

    if not validate_inputs():
        raise RuntimeError("Input validation failed. Check files in ./inputs/")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    logger.info("Loading data...")
    cap_left = load_left_video()
    cap_right = load_right_video()
    imu_raw = load_imu_data()

    meta = get_video_metadata(cap_left)
    total_frames = meta["frame_count"]
    fps = meta["fps"]
    duration_sec = meta["duration_sec"]

    logger.info("Video: %d frames, %.1f fps, %.1f sec", total_frames, fps, duration_sec)

    # ── Process IMU ──
    logger.info("Processing IMU...")
    processed_imu = process_imu(imu_raw)
    imu_swing_events = detect_swing_peaks(processed_imu)
    imu_cycles = pair_swing_events(imu_swing_events)

    # ── Align IMU to video ──
    imu_alignment = align_imu_to_video(imu_raw[:, 0], fps, total_frames)

    # ── Initialize components ──
    detector = ShovelDetector()
    detector.load()

    stereo = StereoDepthEstimator()
    fsm = CycleFSM()
    annotator = VideoAnnotator(fps=fps, resolution=(meta["width"], meta["height"]))
    annotator.open()

    # ── Frame processing loop ──
    logger.info("Processing %d frames...", total_frames)
    fill_volumes_per_cycle: List[Optional[float]] = []
    current_cycle_fills: List[float] = []
    prev_cycle_count = 0

    for frame_idx in range(total_frames):
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or frame_left is None:
            break

        detections = detector.detect(frame_left, frame_idx)
        angles = compute_joint_angles(detections)
        truck_visible = detections.get_by_class("truck") is not None
        phase = fsm.update(angles, truck_visible=truck_visible)

        bucket_fill: Optional[float] = None
        if frame_idx % STEREO_SAMPLE_RATE == 0 and ret_right and frame_right is not None:
            _, depth_map = stereo.compute_depth(frame_left, frame_right)
            bucket_det = detections.get_by_class("bucket")
            bucket_fill = estimate_bucket_fill(depth_map, bucket_det)
            if bucket_fill is not None:
                current_cycle_fills.append(bucket_fill)

        cycle_count = len(fsm.cycles)
        if cycle_count > prev_cycle_count:
            avg_fill = (
                sum(current_cycle_fills) / len(current_cycle_fills)
                if current_cycle_fills
                else None
            )
            fill_volumes_per_cycle.append(avg_fill)
            current_cycle_fills = []
            prev_cycle_count = cycle_count

        annotator.annotate_frame(
            frame=frame_left,
            detections=detections,
            phase=phase,
            angles=angles,
            cycle_count=cycle_count,
            frame_idx=frame_idx,
        )

        if frame_idx % 500 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Frame %d/%d (%.0f%%) — %.1fs elapsed",
                frame_idx, total_frames,
                frame_idx / total_frames * 100,
                elapsed,
            )

    # ── Finalize ──
    fsm.finalize(total_frames - 1)
    annotator.close()
    cap_left.release()
    cap_right.release()

    # ── Fuse cycles ──
    video_cycles = fsm.cycles
    fused = fuse_cycles(video_cycles, imu_cycles)

    # ── Compute productivity ──
    report = compute_productivity(fused, fill_volumes_per_cycle, duration_sec)
    recommendations = generate_recommendations(report)
    report.recommendations = recommendations

    # ── Write output ──
    write_json_report(report, recommendations)

    elapsed = time.time() - start_time
    logger.info("=== Pipeline complete in %.1fs ===", elapsed)
    logger.info(
        "Results: %d cycles, %.0f t/hr, %.0f%% utilization",
        report.summary.total_cycles,
        report.summary.estimated_productivity_tph,
        report.summary.utilization_pct,
    )


if __name__ == "__main__":
    run_pipeline()
