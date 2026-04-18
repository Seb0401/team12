"""Generate JSON metrics report."""

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from solution.config import OUTPUTS_DIR, METRICS_JSON_FILENAME
from solution.productivity.metrics import ProductivityReport

logger = logging.getLogger(__name__)


def write_json_report(
    report: ProductivityReport,
    recommendations: List[str],
    imu_analysis: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Serialize productivity report to JSON, optionally merging IMU dashboard analysis.

    Solution.main metrics have priority. IMU analysis is nested under 'imu_analysis'.

    @param report - Complete productivity report
    @param recommendations - Operator recommendation strings
    @param imu_analysis - Optional IMU dashboard analysis dict from run_imu_dashboard_analysis
    @returns Path to written JSON file
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUTS_DIR / METRICS_JSON_FILENAME

    data: Dict[str, Any] = {
        "summary": asdict(report.summary),
        "cycles": [asdict(c) for c in report.cycles],
        "recommendations": recommendations,
    }

    if imu_analysis is not None:
        data["imu_analysis"] = imu_analysis

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)

    logger.info("Metrics report written: %s", output_path)
    return output_path


def _json_default(obj):
    """Handle non-serializable types (numpy, etc)."""
    import numpy as np

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")
