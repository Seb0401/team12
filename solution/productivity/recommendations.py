"""Generate operator recommendations from productivity metrics."""

import logging
from typing import List

from solution.productivity.metrics import ProductivityReport, CycleMetrics

logger = logging.getLogger(__name__)


def generate_recommendations(report: ProductivityReport) -> List[str]:
    """
    Analyze cycle data and produce actionable operator recommendations.

    @param report - Complete productivity report
    @returns List of recommendation strings
    """
    recs: List[str] = []

    if not report.cycles:
        recs.append("Insufficient data to generate recommendations.")
        return recs

    summary = report.summary

    _check_idle_time(report, recs)
    _check_swing_time(report, recs)
    _check_cycle_consistency(report, recs)
    _check_utilization(summary.utilization_pct, recs)
    _check_bucket_fill(report, recs)

    if not recs:
        recs.append(
            f"Good performance: {summary.cycles_per_hour:.0f} cycles/hr, "
            f"{summary.utilization_pct:.0f}% utilization."
        )

    logger.info("Generated %d recommendations", len(recs))
    return recs


def _check_idle_time(report: ProductivityReport, recs: List[str]) -> None:
    for cm in report.cycles:
        if cm.idle_sec > 10.0:
            recs.append(
                f"Cycle {cm.cycle_id}: {cm.idle_sec:.0f}s idle "
                f"({cm.start_sec:.0f}–{cm.end_sec:.0f}s) — "
                "investigate truck positioning or wait time."
            )


def _check_swing_time(report: ProductivityReport, recs: List[str]) -> None:
    swing_times = [
        cm.swing_to_dump_sec + cm.swing_to_dig_sec for cm in report.cycles
    ]
    if not swing_times:
        return
    avg_swing = sum(swing_times) / len(swing_times)
    if avg_swing > 12.0:
        recs.append(
            f"Average total swing time {avg_swing:.1f}s — "
            "optimal is <10s. Consider repositioning closer to trucks."
        )


def _check_cycle_consistency(report: ProductivityReport, recs: List[str]) -> None:
    if len(report.cycles) < 3:
        return
    best = report.summary.best_cycle_time_sec
    worst = report.summary.worst_cycle_time_sec
    if worst > best * 1.5:
        recs.append(
            f"Cycle time variance high: best {best:.0f}s vs worst {worst:.0f}s. "
            "Investigate outlier cycles for delays."
        )


def _check_utilization(utilization_pct: float, recs: List[str]) -> None:
    if utilization_pct < 75.0:
        recs.append(
            f"Utilization {utilization_pct:.0f}% is below target 75%. "
            "Reduce standby/idle periods."
        )


def _check_bucket_fill(report: ProductivityReport, recs: List[str]) -> None:
    fills = [cm.bucket_fill_pct for cm in report.cycles if cm.bucket_fill_pct is not None]
    if not fills:
        return
    avg_fill = sum(fills) / len(fills)
    if avg_fill < 80.0:
        recs.append(
            f"Average bucket fill {avg_fill:.0f}% — "
            "target ≥85%. Adjust dig depth or approach angle."
        )
