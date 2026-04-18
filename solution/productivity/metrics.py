"""OEE-based productivity metric computation for mining shovels."""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from solution.config import (
    BUCKET_VOLUME_M3,
    MATERIAL_DENSITY_TPM3,
    FPS,
)
from solution.imu.fusion import FusedCycle
from solution.kinematics.cycle_detector import Phase

logger = logging.getLogger(__name__)


@dataclass
class CycleMetrics:
    """Per-cycle productivity breakdown."""

    cycle_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    dig_sec: float = 0.0
    swing_to_dump_sec: float = 0.0
    dump_sec: float = 0.0
    swing_to_dig_sec: float = 0.0
    idle_sec: float = 0.0
    bucket_fill_m3: Optional[float] = None
    bucket_fill_pct: Optional[float] = None
    efficiency_vs_best: float = 0.0


@dataclass
class SummaryMetrics:
    """Aggregate productivity metrics for entire video."""

    total_duration_sec: float = 0.0
    total_cycles: int = 0
    avg_cycle_time_sec: float = 0.0
    cycles_per_hour: float = 0.0
    utilization_pct: float = 0.0
    avg_bucket_fill_m3: Optional[float] = None
    estimated_payload_tonnes: float = 0.0
    estimated_productivity_tph: float = 0.0
    total_idle_time_sec: float = 0.0
    total_dig_time_sec: float = 0.0
    total_swing_time_sec: float = 0.0
    best_cycle_time_sec: float = 0.0
    worst_cycle_time_sec: float = 0.0


@dataclass
class ProductivityReport:
    """Complete productivity analysis results."""

    summary: SummaryMetrics = field(default_factory=SummaryMetrics)
    cycles: List[CycleMetrics] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def compute_productivity(
    fused_cycles: List[FusedCycle],
    fill_volumes: Optional[List[Optional[float]]] = None,
    total_video_duration_sec: float = 0.0,
) -> ProductivityReport:
    """
    Compute OEE-based productivity metrics from fused cycle data.

    Productivity (t/hr) = Payload × Cycles/hr × Utilization

    @param fused_cycles - Merged cycle detections
    @param fill_volumes - Per-cycle bucket fill estimates in m³ (or None)
    @param total_video_duration_sec - Total video length for utilization calc
    @returns Complete ProductivityReport
    """
    if not fused_cycles:
        logger.warning("No cycles detected — cannot compute productivity")
        return ProductivityReport()

    cycle_metrics_list: List[CycleMetrics] = []

    for i, fc in enumerate(fused_cycles):
        cm = CycleMetrics(
            cycle_id=fc.cycle_id,
            start_sec=fc.start_sec,
            end_sec=fc.end_sec,
            duration_sec=fc.duration_sec,
        )

        if fc.video_cycle is not None:
            vc = fc.video_cycle
            cm.dig_sec = vc.phase_duration(Phase.DIG)
            cm.swing_to_dump_sec = vc.phase_duration(Phase.SWING_TO_DUMP)
            cm.dump_sec = vc.phase_duration(Phase.DUMP)
            cm.swing_to_dig_sec = vc.phase_duration(Phase.SWING_TO_DIG)
            cm.idle_sec = vc.phase_duration(Phase.IDLE)

        if fill_volumes and i < len(fill_volumes) and fill_volumes[i] is not None:
            cm.bucket_fill_m3 = fill_volumes[i]
            cm.bucket_fill_pct = (fill_volumes[i] / BUCKET_VOLUME_M3) * 100.0

        cycle_metrics_list.append(cm)

    durations = [cm.duration_sec for cm in cycle_metrics_list]
    best_time = min(durations)

    for cm in cycle_metrics_list:
        cm.efficiency_vs_best = best_time / cm.duration_sec if cm.duration_sec > 0 else 0.0

    total_idle = sum(cm.idle_sec for cm in cycle_metrics_list)
    total_dig = sum(cm.dig_sec for cm in cycle_metrics_list)
    total_swing = sum(
        cm.swing_to_dump_sec + cm.swing_to_dig_sec for cm in cycle_metrics_list
    )

    n_cycles = len(cycle_metrics_list)
    avg_cycle = sum(durations) / n_cycles
    cycles_per_hour = 3600.0 / avg_cycle if avg_cycle > 0 else 0.0

    active_time = total_video_duration_sec - total_idle
    utilization = (active_time / total_video_duration_sec * 100.0) if total_video_duration_sec > 0 else 0.0

    fills = [cm.bucket_fill_m3 for cm in cycle_metrics_list if cm.bucket_fill_m3 is not None]
    avg_fill = sum(fills) / len(fills) if fills else None

    payload = (avg_fill or BUCKET_VOLUME_M3) * MATERIAL_DENSITY_TPM3
    productivity_tph = payload * cycles_per_hour * (utilization / 100.0)

    summary = SummaryMetrics(
        total_duration_sec=total_video_duration_sec,
        total_cycles=n_cycles,
        avg_cycle_time_sec=avg_cycle,
        cycles_per_hour=cycles_per_hour,
        utilization_pct=utilization,
        avg_bucket_fill_m3=avg_fill,
        estimated_payload_tonnes=payload,
        estimated_productivity_tph=productivity_tph,
        total_idle_time_sec=total_idle,
        total_dig_time_sec=total_dig,
        total_swing_time_sec=total_swing,
        best_cycle_time_sec=best_time,
        worst_cycle_time_sec=max(durations),
    )

    logger.info(
        "Productivity: %d cycles, %.1f cycles/hr, %.1f%% util, %.0f t/hr",
        n_cycles, cycles_per_hour, utilization, productivity_tph,
    )

    return ProductivityReport(
        summary=summary,
        cycles=cycle_metrics_list,
    )
