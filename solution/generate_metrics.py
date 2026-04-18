"""
Generate two separate metrics.json files:
  - outputs/metrics_imu.json   — cycles and phases from IMU data only
  - outputs/metrics_video.json — cycles and phases from video (bucket Y + truck)

Usage:
    python -m solution.generate_metrics
"""

import json
import logging
import time
import glob
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from solution.config import (
    setup_logging,
    INPUTS_DIR,
    OUTPUTS_DIR,
    IMU_GLOB,
    BUCKET_VOLUME_M3,
    MATERIAL_DENSITY_TPM3,
    FPS,
)

logger = logging.getLogger(__name__)


def _generate_imu_metrics() -> dict:
    """
    Produce cycle metrics from IMU data only (gyro-Z swing detection + pitch).

    @returns Dict ready for JSON serialization
    """
    pattern = str(INPUTS_DIR / IMU_GLOB)
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No IMU file matching '{IMU_GLOB}' in {INPUTS_DIR}")

    data = np.load(matches[0])
    timestamps_ns = data[:, 0]
    timestamps_sec = (timestamps_ns - timestamps_ns[0]) / 1e9
    total_duration = timestamps_sec[-1]

    ax, ay, az = data[:, 1], data[:, 2], data[:, 3]
    gx, gy, gz = data[:, 4], data[:, 5], data[:, 6]
    qw, qx, qy, qz = data[:, 7], data[:, 8], data[:, 9], data[:, 10]

    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1, 1))

    window = 50
    gz_smooth = np.convolve(gz, np.ones(window) / window, mode="same")

    threshold = 0.4
    min_segment_time = 2.5

    moving_left = gz_smooth > threshold

    segments = []
    in_seg = False
    seg_start = 0
    for i in range(len(moving_left)):
        if moving_left[i] and not in_seg:
            seg_start = i
            in_seg = True
        elif not moving_left[i] and in_seg:
            segments.append((seg_start, i))
            in_seg = False

    valid_segments = [
        (s, e) for s, e in segments
        if timestamps_sec[e] - timestamps_sec[s] > min_segment_time
    ]

    cycles = []
    for i in range(len(valid_segments) - 1):
        start = valid_segments[i][0]
        end = valid_segments[i + 1][0]

        cycle_gz = gz_smooth[start:end]
        has_return = np.sum(cycle_gz < -threshold) > 20
        if not has_return:
            continue

        cycle_t = timestamps_sec[start:end]
        cycle_acc = acc_mag[start:end]
        cycle_pitch = pitch[start:end]

        duration = cycle_t[-1] - cycle_t[0]
        effort = float(np.sum(cycle_acc))
        pitch_range = float(cycle_pitch.max() - cycle_pitch.min())
        smoothness = float(cycle_acc.std())

        dt = np.mean(np.diff(cycle_t)) if len(cycle_t) > 1 else 1.0
        lifting_mask = np.diff(cycle_pitch) > 0
        lifting_time = float(np.sum(lifting_mask) * dt)

        swing_left_time = float(np.sum(cycle_gz > threshold) * dt)
        swing_right_time = float(np.sum(cycle_gz < -threshold) * dt)
        idle_time = float(np.sum(np.abs(cycle_gz) < 0.1) * dt)
        active_time = duration - idle_time

        cycles.append({
            "cycle_id": len(cycles) + 1,
            "start_sec": round(float(cycle_t[0]), 2),
            "end_sec": round(float(cycle_t[-1]), 2),
            "duration_sec": round(duration, 2),
            "swing_left_sec": round(swing_left_time, 2),
            "swing_right_sec": round(swing_right_time, 2),
            "idle_sec": round(idle_time, 2),
            "active_sec": round(active_time, 2),
            "lifting_time_sec": round(lifting_time, 2),
            "effort": round(effort, 2),
            "pitch_range_rad": round(pitch_range, 4),
            "smoothness": round(smoothness, 4),
        })

    total_idle = sum(c["idle_sec"] for c in cycles)
    total_active = sum(c["active_sec"] for c in cycles)
    durations = [c["duration_sec"] for c in cycles]
    avg_cycle = sum(durations) / len(durations) if durations else 0
    cycles_per_hour = 3600 / avg_cycle if avg_cycle > 0 else 0
    utilization = (total_active / total_duration * 100) if total_duration > 0 else 0

    for c in cycles:
        efficiency = (c["pitch_range_rad"] * c["effort"]) / c["duration_sec"] if c["duration_sec"] > 0 else 0
        c["efficiency"] = round(efficiency, 2)

    max_eff = max((c["efficiency"] for c in cycles), default=1)
    for c in cycles:
        c["efficiency_normalized"] = round(c["efficiency"] / max_eff, 4) if max_eff > 0 else 0

    return {
        "source": "imu_only",
        "summary": {
            "total_duration_sec": round(total_duration, 2),
            "total_cycles": len(cycles),
            "avg_cycle_time_sec": round(avg_cycle, 2),
            "cycles_per_hour": round(cycles_per_hour, 2),
            "utilization_pct": round(utilization, 2),
            "total_idle_time_sec": round(total_idle, 2),
            "imu_sample_rate_hz": round(1.0 / np.median(np.diff(timestamps_sec)), 1),
            "best_cycle_time_sec": round(min(durations), 2) if durations else 0,
            "worst_cycle_time_sec": round(max(durations), 2) if durations else 0,
        },
        "cycles": cycles,
    }


def _generate_video_metrics() -> dict:
    """
    Produce cycle metrics from video analysis (bucket Y + truck presence).

    @returns Dict ready for JSON serialization
    """
    from solution.data.loader import load_left_video, get_video_metadata
    from solution.detection.detector import ShovelDetector
    from solution.kinematics.bucket_phase import BucketPhaseDetector, BucketPhase

    cap = load_left_video()
    meta = get_video_metadata(cap)
    total_frames = meta["frame_count"]
    fps = meta["fps"]
    duration_sec = meta["duration_sec"]

    detector = ShovelDetector()
    detector.load()
    phase_detector = BucketPhaseDetector()

    phase_counts = {p: 0 for p in BucketPhase}

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        detections = detector.detect(frame, frame_idx)
        phase = phase_detector.update(detections)
        phase_counts[phase] += 1

        if frame_idx % 500 == 0:
            logger.info("Video metrics: frame %d/%d (%.0f%%)", frame_idx, total_frames, frame_idx / total_frames * 100)

    phase_detector.finalize(total_frames - 1)
    cap.release()

    cycles = phase_detector.cycles
    idle_events = phase_detector.idle_events

    phase_time = {}
    for p in BucketPhase:
        phase_time[p.name] = round(phase_counts[p] / fps, 2)

    total_idle = phase_time.get("INACTIVO", 0)
    total_excavando = phase_time.get("EXCAVANDO", 0)
    total_transporte = phase_time.get("TRANSPORTE", 0)
    total_botando = phase_time.get("BOTANDO_CARGA", 0)

    cycle_data = []
    for c in cycles:
        breakdown = c.get("phase_breakdown", {})
        cycle_data.append({
            "cycle_id": c["cycle_id"],
            "start_sec": round(c["start_sec"], 2),
            "end_sec": round(c["end_sec"], 2),
            "duration_sec": round(c["duration_sec"], 2),
            "phase_breakdown": breakdown,
        })

    durations = [c["duration_sec"] for c in cycle_data]
    avg_cycle = sum(durations) / len(durations) if durations else 0
    cycles_per_hour = 3600 / avg_cycle if avg_cycle > 0 else 0
    active_time = duration_sec - total_idle
    utilization = (active_time / duration_sec * 100) if duration_sec > 0 else 0

    if durations:
        best = min(durations)
        for c in cycle_data:
            c["efficiency_vs_best"] = round(best / c["duration_sec"], 4) if c["duration_sec"] > 0 else 0

    total_idle_events_duration = sum(e["duration_sec"] for e in idle_events)

    return {
        "source": "video_bucket_position",
        "summary": {
            "total_duration_sec": round(duration_sec, 2),
            "total_cycles": len(cycle_data),
            "avg_cycle_time_sec": round(avg_cycle, 2),
            "cycles_per_hour": round(cycles_per_hour, 2),
            "utilization_pct": round(utilization, 2),
            "phase_totals": {
                "excavando_sec": round(total_excavando, 2),
                "transporte_sec": round(total_transporte, 2),
                "botando_carga_sec": round(total_botando, 2),
                "inactivo_sec": round(total_idle, 2),
            },
            "idle_alert_count": len(idle_events),
            "idle_alert_total_sec": round(total_idle_events_duration, 2),
            "best_cycle_time_sec": round(min(durations), 2) if durations else 0,
            "worst_cycle_time_sec": round(max(durations), 2) if durations else 0,
        },
        "cycles": cycle_data,
        "idle_events": idle_events,
    }


def run() -> None:
    setup_logging()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=== Generating IMU metrics ===")
    t0 = time.time()
    imu_report = _generate_imu_metrics()
    imu_path = OUTPUTS_DIR / "metrics_imu.json"
    with open(imu_path, "w") as f:
        json.dump(imu_report, f, indent=2)
    logger.info("IMU metrics written to %s (%.1fs)", imu_path, time.time() - t0)

    logger.info("=== Generating Video metrics ===")
    t0 = time.time()
    video_report = _generate_video_metrics()
    video_path = OUTPUTS_DIR / "metrics_video.json"
    with open(video_path, "w") as f:
        json.dump(video_report, f, indent=2)
    logger.info("Video metrics written to %s (%.1fs)", video_path, time.time() - t0)

    logger.info("=== Done ===")
    logger.info("IMU: %d cycles, %.1f cycles/hr", imu_report["summary"]["total_cycles"], imu_report["summary"]["cycles_per_hour"])
    logger.info("Video: %d cycles, %.1f cycles/hr", video_report["summary"]["total_cycles"], video_report["summary"]["cycles_per_hour"])


if __name__ == "__main__":
    run()
