"""
IMU-based dashboard analysis for EX-5600 shovel productivity.

Extracts cycle metrics, productivity KPIs, optimal cycle analysis,
and optimization scenarios from raw IMU data (gyro-Z swing detection + pitch).
Ported from export_dashboard.py as a library function (no I/O, no argparse).
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (matching export_dashboard.py calibration)
# ---------------------------------------------------------------------------

GZ_THRESHOLD = 0.4
MIN_SEGMENT_TIME = 2.5
GZ_SMOOTH_WINDOW = 50
IDLE_THRESHOLD = 0.3
BUCKET_TONNES = 52.0
HORAS_OPERACION = 20
CICLOS_POR_LADO = 7
TARGET_UTILIZATION_PCT = 85.0
TARGET_MAX_GAP_SEC = 8.0
IDEAL_DURATION_PERCENTILE = 10
IDEAL_WEIGHT_PERCENTILE = 90


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive kinematic signals from raw IMU columns.

    @param df - DataFrame with columns: time, ax, ay, az, gx, gy, gz, qw, qx, qy, qz
    @returns DataFrame with added columns: acc_mag, acc_dynamic, gz_smooth, pitch, yaw,
             moving_left, moving_right, idle
    """
    df = df.copy()
    df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    df["acc_dynamic"] = np.maximum(df["acc_mag"] - 9.81, 0.0)
    df["gz_smooth"] = df["gz"].rolling(window=GZ_SMOOTH_WINDOW, min_periods=1).mean()
    df["pitch"] = np.arcsin(
        np.clip(2.0 * (df["qw"] * df["qy"] - df["qz"] * df["qx"]), -1.0, 1.0)
    )
    df["yaw"] = np.arctan2(
        2.0 * (df["qw"] * df["qz"] + df["qx"] * df["qy"]),
        1.0 - 2.0 * (df["qy"] ** 2 + df["qz"] ** 2),
    )
    df["moving_left"] = df["gz_smooth"] > GZ_THRESHOLD
    df["moving_right"] = df["gz_smooth"] < -GZ_THRESHOLD
    df["idle"] = df["acc_mag"] < IDLE_THRESHOLD
    return df


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def _detect_cycles(
    df: pd.DataFrame,
) -> Tuple[List[pd.DataFrame], List[Tuple[int, int]]]:
    """
    Detect excavation cycles from gz_smooth left-swing segments.

    @param df - Preprocessed IMU DataFrame
    @returns Tuple of (list of cycle DataFrames, list of valid segment (start, end) tuples)
    """
    segments: List[Tuple[int, int]] = []
    in_seg = False
    start = 0

    for i in range(len(df)):
        v = df["moving_left"].iloc[i]
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False

    valid = [
        (s, e)
        for s, e in segments
        if df["time"].iloc[e] - df["time"].iloc[s] > MIN_SEGMENT_TIME
    ]

    cycles: List[pd.DataFrame] = []
    for i in range(len(valid) - 1):
        s = valid[i][0]
        e = valid[i + 1][0]
        cyc = df.iloc[s:e].copy()
        has_right = (cyc["gz_smooth"] < -GZ_THRESHOLD).sum() > 20
        has_left = (cyc["gz_smooth"] > GZ_THRESHOLD).sum() > 20
        if has_right or has_left:
            cycles.append(cyc)

    return cycles, valid


# ---------------------------------------------------------------------------
# Per-cycle metrics
# ---------------------------------------------------------------------------


def _compute_cycle_metrics(cycles: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Compute detailed metrics for each detected cycle.

    @param cycles - List of cycle DataFrames from _detect_cycles
    @returns DataFrame with one row per valid cycle and columns: cycle, t_start, t_end,
             duration, effort, pitch_range, smoothness, lifting_time, weight_proxy,
             side, efficiency, cost, w_norm, t_norm, eff_norm, score, tonnes,
             cycle_rate, productivity_cycle
    """
    rows: List[Dict[str, Any]] = []
    cycle_count = 0

    for cyc in cycles:
        t_start = float(cyc["time"].iloc[0])
        t_end = float(cyc["time"].iloc[-1])
        dur = t_end - t_start
        if dur < 5.0:
            continue

        dt_arr = cyc["time"].diff().fillna(0).values
        effort = float(cyc["acc_mag"].sum())
        pitch_range = float(cyc["pitch"].max() - cyc["pitch"].min())
        smoothness = float(cyc["acc_mag"].std())
        dt_mean = float(cyc["time"].diff().mean())
        lifting_mask = (cyc["pitch"].diff() > 0).values
        lifting_time = float(
            lifting_mask.sum() * dt_mean if not np.isnan(dt_mean) else 0.0
        )
        acc_dyn = cyc["acc_dynamic"].values
        weight_proxy = float(np.sum(acc_dyn[lifting_mask] * dt_arr[lifting_mask]))
        side = (
            "izquierda" if (cycle_count // CICLOS_POR_LADO) % 2 == 0 else "derecha"
        )
        eff = (pitch_range * effort) / dur if dur > 0 else 0.0
        cost = dur * effort

        rows.append(
            {
                "cycle": cycle_count,
                "t_start": t_start,
                "t_end": t_end,
                "duration": dur,
                "effort": effort,
                "pitch_range": pitch_range,
                "smoothness": smoothness,
                "lifting_time": lifting_time,
                "weight_proxy": weight_proxy,
                "side": side,
                "efficiency": eff,
                "cost": cost,
            }
        )
        cycle_count += 1

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    res = res.reset_index(drop=True)
    res["cycle"] = res.index

    w_max = res["weight_proxy"].max() or 1.0
    t_max = res["duration"].max() or 1.0
    eff_max = res["efficiency"].max() or 1.0

    res["w_norm"] = res["weight_proxy"] / w_max
    res["t_norm"] = res["duration"] / t_max
    res["eff_norm"] = res["efficiency"] / eff_max

    t_inv = 1.0 / res["t_norm"]
    t_inv_norm = t_inv / t_inv.max()
    res["score"] = 0.5 * res["w_norm"] + 0.3 * t_inv_norm + 0.2 * res["eff_norm"]

    res["tonnes"] = res["weight_proxy"] * (BUCKET_TONNES / w_max)
    res["cycle_rate"] = 3600.0 / res["duration"]
    res["productivity_cycle"] = res["cycle_rate"] * res["tonnes"]

    return res


# ---------------------------------------------------------------------------
# Global productivity
# ---------------------------------------------------------------------------


def _compute_productivity(
    df: pd.DataFrame,
    cycles: List[pd.DataFrame],
    res: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute aggregate productivity KPIs from cycle metrics.

    @param df - Preprocessed IMU DataFrame
    @param cycles - List of cycle DataFrames
    @param res - Cycle metrics DataFrame from _compute_cycle_metrics
    @returns Dict of productivity KPIs including gaps_arr (numpy array)
    """
    total_t = float(df["time"].iloc[-1] - df["time"].iloc[0])
    total_h = total_t / 3600.0 if total_t > 0 else 1e-9
    cph = len(res) / total_h

    dt = float(df["time"].diff().mean())
    idle_t = float(df["idle"].sum()) * (dt if not np.isnan(dt) else 0.0)
    idle_r = idle_t / total_t if total_t > 0 else 0.0
    util = max(0.0, 1.0 - idle_r)

    valid_cycles = [
        cycles[int(r["cycle"])]
        for _, r in res.iterrows()
        if int(r["cycle"]) < len(cycles)
    ]
    gaps = [
        float(valid_cycles[i + 1]["time"].iloc[0] - valid_cycles[i]["time"].iloc[-1])
        for i in range(len(valid_cycles) - 1)
    ]
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])

    cv = (
        res["duration"].std() / res["duration"].mean()
        if len(res) > 1 and res["duration"].mean() > 0
        else 0.0
    )

    avg_tonnes = float(res["tonnes"].mean())
    tph = cph * avg_tonnes * util
    total_cost = float(res["cost"].sum())
    total_tonnes = float(res["tonnes"].sum()) or 1.0
    cost_per_tonne = total_cost / total_tonnes

    valid_dur = res["duration"][res["duration"] >= 10.0]
    best_cycle_sec = (
        float(valid_dur.min()) if len(valid_dur) > 0 else float(res["duration"].min())
    )

    return {
        "total_time_sec": total_t,
        "total_cycles": len(res),
        "cycles_per_hour": cph,
        "utilization_pct": util * 100,
        "idle_ratio_pct": idle_r * 100,
        "avg_effort": float(res["effort"].mean()),
        "consistency_cv": float(cv),
        "avg_cycle_sec": float(res["duration"].mean()),
        "best_cycle_sec": best_cycle_sec,
        "worst_cycle_sec": float(res["duration"].max()),
        "median_gap_sec": float(np.median(gaps_arr)),
        "p90_gap_sec": float(np.percentile(gaps_arr, 90)),
        "gaps_arr": gaps_arr,
        "avg_tonnes": avg_tonnes,
        "tonnes_per_hour": tph,
        "total_cost": total_cost,
        "cost_per_tonne": cost_per_tonne,
        "total_tonnes": total_tonnes,
    }


# ---------------------------------------------------------------------------
# Optimal cycles
# ---------------------------------------------------------------------------


def _compute_optimal_cycles(res: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify best real cycle and compute ideal cycle parameters.

    @param res - Cycle metrics DataFrame
    @returns Dict with best_real (Series) and ideal (dict of ideal KPIs)
    """
    best_real_idx = res["score"].idxmax()
    best_real = res.loc[best_real_idx]

    res_valid = res[res["duration"] >= 10.0]
    if len(res_valid) == 0:
        res_valid = res

    t_ideal = float(
        np.percentile(res_valid["duration"], IDEAL_DURATION_PERCENTILE)
    )
    w_ideal = float(
        np.percentile(res_valid["weight_proxy"], IDEAL_WEIGHT_PERCENTILE)
    )
    eff_ideal = float(
        np.percentile(res_valid["efficiency"], IDEAL_WEIGHT_PERCENTILE)
    )

    w_max = float(res["weight_proxy"].max()) or 1.0
    tonnes_ideal = w_ideal * (BUCKET_TONNES / w_max)
    cph_ideal = 3600.0 / t_ideal if t_ideal > 0 else 0.0
    tph_ideal = cph_ideal * tonnes_ideal

    return {
        "best_real": best_real,
        "ideal": {
            "weight_proxy": w_ideal,
            "duration": t_ideal,
            "efficiency": eff_ideal,
            "tonnes": tonnes_ideal,
            "cph": cph_ideal,
            "tph": tph_ideal,
            "note": f"p{IDEAL_DURATION_PERCENTILE} dur, p{IDEAL_WEIGHT_PERCENTILE} peso",
        },
    }


# ---------------------------------------------------------------------------
# Optimization scenario
# ---------------------------------------------------------------------------


def _compute_optimized(
    real: Dict[str, Any],
    res: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Simulate optimized scenario by reducing inter-cycle gaps.

    @param real - Productivity KPIs from _compute_productivity
    @param res - Cycle metrics DataFrame
    @returns Dict of optimized scenario KPIs
    """
    gaps_arr = real["gaps_arr"]
    med_gap = real["median_gap_sec"]
    opt_gaps = np.clip(gaps_arr, None, med_gap)
    gap_red = float(gaps_arr.mean() - opt_gaps.mean())

    avg_cyc = real["avg_cycle_sec"] if real["avg_cycle_sec"] > 0 else 1.0
    extra_cph = gap_red * real["cycles_per_hour"] / avg_cyc

    opt_util = max(real["utilization_pct"], TARGET_UTILIZATION_PCT)
    opt_cph = real["cycles_per_hour"] + extra_cph
    opt_cph += opt_cph * (opt_util - real["utilization_pct"]) / 100.0

    res_valid = res[res["duration"] >= 10.0]
    if len(res_valid) >= 4:
        opt_avg_cycle = float(np.percentile(res_valid["duration"], 25))
    else:
        opt_avg_cycle = real["best_cycle_sec"] * 1.05

    opt_cph_from_cycle = 3600.0 / opt_avg_cycle if opt_avg_cycle > 0 else opt_cph
    opt_cph_final = min(opt_cph, opt_cph_from_cycle)

    real_tph = real["tonnes_per_hour"]
    opt_tph = opt_cph_final * real["avg_tonnes"] * (opt_util / 100.0)
    delta_tph = opt_tph - real_tph

    return {
        "cycles_per_hour": opt_cph_final,
        "utilization_pct": opt_util,
        "avg_cycle_sec": opt_avg_cycle,
        "gap_reduction_sec": gap_red,
        "extra_cycles_per_hour": extra_cph,
        "real_tph": real_tph,
        "opt_tph": opt_tph,
        "delta_tph": delta_tph,
        "delta_tpd": delta_tph * HORAS_OPERACION,
        "delta_pct": (delta_tph / real_tph * 100) if real_tph > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------


def _generate_insights(
    res: pd.DataFrame,
    real: Dict[str, Any],
    opt: Dict[str, Any],
    optimos: Dict[str, Any],
) -> List[str]:
    """
    Generate actionable operator insights from IMU analysis.

    @param res - Cycle metrics DataFrame
    @param real - Productivity KPIs
    @param opt - Optimized scenario KPIs
    @param optimos - Optimal cycle analysis
    @returns List of insight/recommendation strings
    """
    ins: List[str] = []

    if real["consistency_cv"] > 0.15:
        ins.append(
            f"Alta variabilidad de ciclos (CV={real['consistency_cv']:.2f}) "
            "— estandarizar tecnica."
        )
    if real["idle_ratio_pct"] > 15:
        ins.append(
            f"{real['idle_ratio_pct']:.0f}% tiempo muerto "
            "— mejorar coordinacion con camiones."
        )
    if real["utilization_pct"] < TARGET_UTILIZATION_PCT:
        ins.append(
            f"Utilizacion {real['utilization_pct']:.0f}% vs "
            f"objetivo {TARGET_UTILIZATION_PCT:.0f}%."
        )
    if real["median_gap_sec"] > TARGET_MAX_GAP_SEC:
        ins.append(
            f"Gap mediano {real['median_gap_sec']:.1f}s > "
            f"objetivo {TARGET_MAX_GAP_SEC}s."
        )
    if real["p90_gap_sec"] > real["median_gap_sec"] * 2:
        ins.append(
            f"P90 gaps ({real['p90_gap_sec']:.1f}s) >> mediana "
            "— esperas prolongadas recurrentes."
        )

    ideal = optimos["ideal"]
    eff_real_avg = float(res["efficiency"].mean())
    eff_ideal_v = (
        (ideal["weight_proxy"] * ideal["efficiency"]) / ideal["duration"]
        if ideal["duration"] > 0
        else 0
    )
    improvement = (eff_ideal_v / eff_real_avg - 1) * 100 if eff_real_avg > 0 else 0

    ins.append(
        "El operador ya demostro capacidad maxima en ciclos separados. "
        f"Estandarizar mejores practicas podria liberar +{improvement:.0f}% de eficiencia."
    )
    ins.append(
        f"Optimizando logistica: +{opt['delta_tph']:.0f} t/hr "
        f"(+{opt['delta_pct']:.0f}%) "
        f"= +{opt['delta_tpd']:.0f} t/dia ({HORAS_OPERACION}h)."
    )
    return ins


# ---------------------------------------------------------------------------
# Signal data for dashboard
# ---------------------------------------------------------------------------


def _build_signal_data(
    df_proc: pd.DataFrame,
    res: pd.DataFrame,
    cycles: List[pd.DataFrame],
    real: Dict[str, Any],
    opt: Dict[str, Any],
    optimos: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Build all signal/chart data needed by the dashboard frontend.

    @param df_proc - Preprocessed IMU DataFrame
    @param res - Cycle metrics DataFrame
    @param cycles - List of cycle DataFrames
    @param real - Productivity KPIs
    @param opt - Optimized scenario KPIs
    @param optimos - Optimal cycle analysis
    @returns Dict with signals, best_cycle_signal, ang_per_cycle, gaps, prod_acum, stats
    """
    n_total = len(df_proc)
    step = max(1, n_total // 800)
    df_sub = df_proc.iloc[::step].copy()

    t_arr = df_proc["time"].values
    gz_arr = df_proc["gz_smooth"].values
    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    gz_cum = np.cumsum(gz_arr * dt_arr)
    gz_cum_sub = gz_cum[::step]

    signals = {
        "time": [round(float(v), 3) for v in df_sub["time"].values],
        "gz_smooth": [round(float(v), 4) for v in df_sub["gz_smooth"].values],
        "pitch": [round(float(v), 4) for v in df_sub["pitch"].values],
        "acc_dynamic": [round(float(v), 4) for v in df_sub["acc_dynamic"].values],
        "acc_mag": [round(float(v), 4) for v in df_sub["acc_mag"].values],
        "gz_cumsum": [round(float(v), 4) for v in gz_cum_sub],
    }

    best_real = optimos["best_real"]
    best_idx = int(best_real["cycle"])
    best_cyc_data = cycles[best_idx] if best_idx < len(cycles) else cycles[-1]
    t_rel = (best_cyc_data["time"] - best_cyc_data["time"].iloc[0]).values
    dyn_max = float(best_cyc_data["acc_dynamic"].max()) or 1.0
    best_cycle_signal = {
        "time": [round(float(v), 3) for v in t_rel],
        "gz_smooth": [round(float(v), 4) for v in best_cyc_data["gz_smooth"].values],
        "pitch": [round(float(v), 4) for v in best_cyc_data["pitch"].values],
        "acc_dynamic_norm": [
            round(float(v) / dyn_max, 4) for v in best_cyc_data["acc_dynamic"].values
        ],
    }

    ang_per_cycle: List[float] = []
    for _, row in res.iterrows():
        t_s = row["t_start"]
        t_e = row["t_end"]
        mask = (t_arr >= t_s) & (t_arr <= t_e)
        gz_c = np.abs(gz_arr[mask])
        dt_c = (
            np.diff(t_arr[mask], prepend=t_arr[mask][0])
            if mask.sum() > 0
            else np.array([0])
        )
        total = float(np.sum(gz_c * dt_c))
        ang_per_cycle.append(round(total, 3))

    cycle_starts = [round(float(row["t_start"]), 3) for _, row in res.iterrows()]
    cycle_sides = [row["side"] for _, row in res.iterrows()]

    gaps_arr = real["gaps_arr"]
    opt_gaps = np.clip(gaps_arr, None, real["median_gap_sec"])

    n = real["total_cycles"]
    t_r = (
        real["tonnes_per_hour"] / real["cycles_per_hour"]
        if real["cycles_per_hour"] > 0
        else 0
    )
    t_o = opt["opt_tph"] / opt["cycles_per_hour"] if opt["cycles_per_hour"] > 0 else 0
    t_i = (
        optimos["ideal"]["tph"] / optimos["ideal"]["cph"]
        if optimos["ideal"]["cph"] > 0
        else 0
    )
    x_cyc = list(range(1, n + 1))
    prod_acum = {
        "cycles": x_cyc,
        "real": [round(i * t_r, 2) for i in x_cyc],
        "opt": [round(i * t_o, 2) for i in x_cyc],
        "ideal": [round(i * t_i, 2) for i in x_cyc],
    }

    stats_cols = [
        "duration",
        "effort",
        "pitch_range",
        "smoothness",
        "lifting_time",
        "tonnes",
        "efficiency",
        "cost",
        "score",
    ]
    stats_df = res[stats_cols].describe().loc[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]
    ]
    stats_dict: Dict[str, Dict[str, float]] = {}
    for col in stats_cols:
        stats_dict[col] = {
            stat: round(float(stats_df.loc[stat, col]), 4) for stat in stats_df.index
        }

    return {
        "signals": signals,
        "best_cycle_signal": best_cycle_signal,
        "ang_per_cycle": ang_per_cycle,
        "cycle_starts": cycle_starts,
        "cycle_sides": cycle_sides,
        "gaps_real": [round(float(v), 4) for v in gaps_arr],
        "gaps_opt": [round(float(v), 4) for v in opt_gaps],
        "prod_acum": prod_acum,
        "stats": stats_dict,
    }


# ---------------------------------------------------------------------------
# JSON-safe cleaning
# ---------------------------------------------------------------------------


def _clean_value(v: Any) -> Any:
    """
    Convert numpy/pandas scalars to JSON-safe Python types.

    @param v - Value to clean
    @returns JSON-serializable Python value
    """
    if isinstance(v, (np.floating, float)):
        return round(float(v), 6)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, bool):
        return bool(v)
    return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_imu_dashboard_analysis(imu_npy_path: str) -> Dict[str, Any]:
    """
    Run full IMU-based dashboard analysis and return a JSON-serializable dict.

    This is the main entry point. It loads the .npy file, preprocesses signals,
    detects cycles, computes all metrics/KPIs, and returns a dict ready for
    merging into metrics.json under an 'imu_analysis' key.

    @param imu_npy_path - Path to the IMU .npy file
    @returns Dict with keys: meta, signals, best_cycle_signal, ang_per_cycle,
             cycle_starts, cycle_sides, gaps_real, gaps_opt, prod_acum, stats,
             real, opt, ideal, ideal_note, best_real, all_cycles, recommendations
    """
    data = np.load(str(imu_npy_path))
    df = pd.DataFrame(
        data,
        columns=[
            "time", "ax", "ay", "az",
            "gx", "gy", "gz",
            "qw", "qx", "qy", "qz",
        ],
    )
    df["time"] = (df["time"] - df["time"].iloc[0]) / 1e9

    logger.info("IMU dashboard analysis: %d samples, %.1fs", len(df), df["time"].iloc[-1])

    df_proc = _preprocess(df)
    cycles, _valid_segments = _detect_cycles(df_proc)
    res = _compute_cycle_metrics(cycles)

    if len(res) == 0:
        logger.warning("No valid IMU cycles detected")
        return {"meta": {"total_samples": len(df)}, "all_cycles": [], "recommendations": []}

    real = _compute_productivity(df_proc, cycles, res)
    opt = _compute_optimized(real, res)
    optimos = _compute_optimal_cycles(res)
    insights = _generate_insights(res, real, opt, optimos)

    signal_data = _build_signal_data(df_proc, res, cycles, real, opt, optimos)

    best_real = optimos["best_real"]
    ideal = optimos["ideal"]

    cycles_data: List[Dict[str, Any]] = []
    for _, row in res.iterrows():
        d = {}
        for k, v in row.items():
            if k in ("t_start", "t_end"):
                continue
            d[k] = _clean_value(v)
        cycles_data.append(d)

    best_real_dict = {
        k: _clean_value(v) for k, v in best_real.items() if k not in ("t_start", "t_end")
    }

    result: Dict[str, Any] = {
        "meta": {
            "total_samples": int(len(df_proc)),
            "duration_sec": round(float(real["total_time_sec"]), 2),
            "duration_min": round(float(real["total_time_sec"]) / 60, 2),
            "sample_freq_hz": round(float(1 / df_proc["time"].diff().mean()), 2),
            "horas_operacion": HORAS_OPERACION,
            "bucket_tonnes": BUCKET_TONNES,
        },
        "real": {k: _clean_value(v) for k, v in real.items() if k != "gaps_arr"},
        "opt": {k: _clean_value(v) for k, v in opt.items()},
        "ideal": {k: _clean_value(v) for k, v in ideal.items() if k != "note"},
        "ideal_note": ideal.get("note", ""),
        "best_real": best_real_dict,
        "all_cycles": cycles_data,
        "recommendations": insights,
    }

    result.update(signal_data)

    logger.info(
        "IMU dashboard analysis complete: %d cycles, %.0f t/hr",
        real["total_cycles"],
        real["tonnes_per_hour"],
    )

    return result
