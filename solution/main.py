"""Main pipeline orchestrator for EX-5600 shovel productivity analysis.

Uses BucketPhaseDetector (vision) refined by Gyro-Y (IMU) for accurate
cycle detection and phase timing. Outputs annotated video + JSON metrics.
"""

import logging
import time
from typing import List, Optional

from solution.config import (
    setup_logging,
    STEREO_SAMPLE_RATE,
    OUTPUTS_DIR,
)
from solution.data.loader import (
    load_left_video,
    load_right_video,
    load_imu_data,
    find_imu_path,
    get_video_metadata,
    validate_inputs,
)
from solution.detection.detector import ShovelDetector
from solution.kinematics.bucket_phase import BucketPhase
from solution.kinematics.fused_phase_detector import FusedPhaseDetector
from solution.stereo.depth import StereoDepthEstimator
from solution.stereo.volume import estimate_bucket_fill
from solution.productivity.metrics import compute_refined_productivity
from solution.productivity.recommendations import generate_recommendations
from solution.output.annotator import VideoAnnotator
from solution.output.csv_exporter import DetectionCSVWriter
from solution.imu.dashboard_analysis import run_imu_dashboard_analysis
from solution.output.report import write_json_report
from solution.utils.time_sync import (
    frame_to_imu_sample,
    AlignmentResult,
    DEFAULT_OFFSET_SAMPLES,
)

logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Execute the full analysis pipeline: load -> detect -> fuse -> output."""
    setup_logging()
    start_time = time.time()

    logger.info("=== EX-5600 Shovel Productivity Analysis (Gyro-Refined) ===")

    if not validate_inputs():
        raise RuntimeError("Input validation failed. Check files in ./inputs/")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data...")
    cap_left = load_left_video()
    cap_right = load_right_video()
    imu_raw = load_imu_data()

    meta = get_video_metadata(cap_left)
    total_frames = meta["frame_count"]
    fps = meta["fps"]
    duration_sec = meta["duration_sec"]

    logger.info("Video: %d frames, %.1f fps, %.1f sec", total_frames, fps, duration_sec)

    gyro_y = imu_raw[:, 5]
    total_imu_samples = len(imu_raw)

    alignment = AlignmentResult(
        offset_samples=DEFAULT_OFFSET_SAMPLES,
        offset_sec=DEFAULT_OFFSET_SAMPLES / fps,
        correlation_score=0.0,
        method="default",
        segment_offsets=[],
    )

    detector = ShovelDetector()
    detector.load()

    stereo = StereoDepthEstimator()
    fused_detector = FusedPhaseDetector()
    annotator = VideoAnnotator(fps=fps, resolution=(meta["width"], meta["height"]))
    annotator.open()

    csv_writer = DetectionCSVWriter()
    csv_writer.open()

    logger.info("Processing %d frames with gyro-Y refinement...", total_frames)
    phase_counts: dict[BucketPhase, int] = {p: 0 for p in BucketPhase}
    fill_volumes_per_cycle: List[Optional[float]] = []
    current_cycle_fills: List[float] = []
    prev_cycle_count = 0

    for frame_idx in range(total_frames):
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()

        if not ret_left or frame_left is None:
            break

        detections = detector.detect(frame_left, frame_idx)
        csv_writer.write_frame(detections, fps)

        sample_idx = frame_to_imu_sample(frame_idx, alignment)
        sample_idx = max(0, min(sample_idx, total_imu_samples - 1))
        gy_value = float(gyro_y[sample_idx])

        phase = fused_detector.update(detections, gy_value)
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

        bucket_fill: Optional[float] = None
        if frame_idx % STEREO_SAMPLE_RATE == 0 and ret_right and frame_right is not None:
            _, depth_map = stereo.compute_depth(frame_left, frame_right)
            bucket_det = detections.get_by_class("bucket")
            bucket_fill = estimate_bucket_fill(depth_map, bucket_det)
            if bucket_fill is not None:
                current_cycle_fills.append(bucket_fill)

        cycle_count = fused_detector.cycle_count
        if cycle_count > prev_cycle_count:
            avg_fill = (
                sum(current_cycle_fills) / len(current_cycle_fills)
                if current_cycle_fills
                else None
            )
            fill_volumes_per_cycle.append(avg_fill)
            current_cycle_fills = []
            prev_cycle_count = cycle_count

        _phase_for_annotator = _bucket_phase_to_fsm_phase(phase)
        annotator.annotate_frame(
            frame=frame_left,
            detections=detections,
            phase=_phase_for_annotator,
            angles=None,
            cycle_count=cycle_count,
            frame_idx=frame_idx,
        )

        if frame_idx % 500 == 0:
            elapsed = time.time() - start_time
            logger.info(
                "Frame %d/%d (%.0f%%) — %.1fs elapsed — phase=%s gyro_y=%.1f",
                frame_idx, total_frames,
                frame_idx / total_frames * 100,
                elapsed,
                phase.name,
                gy_value,
            )

    fused_detector.finalize(total_frames - 1)
    annotator.close()
    csv_writer.close()
    cap_left.release()
    cap_right.release()

    refined_cycles = fused_detector.cycles
    report = compute_refined_productivity(
        refined_cycles,
        phase_counts,
        total_video_duration_sec=duration_sec,
        fill_volumes=fill_volumes_per_cycle,
    )
    recommendations = generate_recommendations(report)
    report.recommendations = recommendations

    imu_analysis = None
    try:
        imu_path = find_imu_path()
        logger.info("Running IMU dashboard analysis from %s", imu_path)
        imu_analysis = run_imu_dashboard_analysis(str(imu_path))
        logger.info("IMU dashboard analysis: %d cycles", len(imu_analysis.get("all_cycles", [])))
    except Exception:
        logger.warning("IMU dashboard analysis failed, skipping", exc_info=True)

    write_json_report(report, recommendations, imu_analysis=imu_analysis)

    elapsed = time.time() - start_time
    logger.info("=== Pipeline complete in %.1fs ===", elapsed)
    logger.info(
        "Results: %d cycles, %.0f t/hr, %.0f%% utilization",
        report.summary.total_cycles,
        report.summary.estimated_productivity_tph,
        report.summary.utilization_pct,
    )

    for p, count in phase_counts.items():
        logger.info("  %s: %.1fs (%.1f%%)", p.name, count / fps, count / total_frames * 100)


def _bucket_phase_to_fsm_phase(bp: BucketPhase):
    """Map BucketPhase to CycleFSM Phase for annotator compatibility."""
    from solution.kinematics.cycle_detector import Phase
    mapping = {
        BucketPhase.EXCAVANDO: Phase.DIG,
        BucketPhase.TRANSPORTE: Phase.SWING_TO_DUMP,
        BucketPhase.BOTANDO_CARGA: Phase.DUMP,
        BucketPhase.INACTIVO: Phase.IDLE,
    }
    return mapping.get(bp, Phase.IDLE)


if __name__ == "__main__":
    run_pipeline()
