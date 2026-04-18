"""
export_dashboard.py
===================
Genera outputs/dashboard_data.json con todos los datos necesarios
para el dashboard HTML (dashboard.html).

Uso:
    python export_dashboard.py
    python export_dashboard.py --imu ruta/al/archivo.npy
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion (igual que dashboard_shovel.py)
# ---------------------------------------------------------------------------

INPUTS_DIR   = Path("./inputs")
OUTPUTS_DIR  = Path("./outputs")
IMU_GLOB     = "*imu*.npy"
FALLBACK_IMU = Path("C:/Hackathon/Archive/40343737_20260313_110600_to_112100_imu.npy")

GZ_THRESHOLD              = 0.4
MIN_SEGMENT_TIME          = 2.5
GZ_SMOOTH_WINDOW          = 50
IDLE_THRESHOLD            = 0.3
BUCKET_TONNES             = 52.0
HORAS_OPERACION           = 20
CICLOS_POR_LADO           = 7
TARGET_UTILIZATION_PCT    = 85.0
TARGET_MAX_GAP_SEC        = 8.0
IDEAL_DURATION_PERCENTILE = 10
IDEAL_WEIGHT_PERCENTILE   = 90


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def find_imu_file(explicit=None):
    if explicit:
        return Path(explicit)
    matches = glob.glob(str(INPUTS_DIR / IMU_GLOB))
    if matches:
        return Path(matches[0])
    if FALLBACK_IMU.exists():
        return FALLBACK_IMU
    raise FileNotFoundError(f"No se encontro archivo IMU en {INPUTS_DIR}")


def load_imu(path):
    data = np.load(str(path))
    df = pd.DataFrame(data, columns=[
        "time", "ax", "ay", "az",
        "gx", "gy", "gz",
        "qw", "qx", "qy", "qz",
    ])
    df["time"] = (df["time"] - df["time"].iloc[0]) / 1e9
    return df


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def preprocess(df):
    df = df.copy()
    df["acc_mag"]     = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["acc_dynamic"] = np.maximum(df["acc_mag"] - 9.81, 0.0)
    df["gz_smooth"]   = df["gz"].rolling(window=GZ_SMOOTH_WINDOW, min_periods=1).mean()
    df["pitch"]       = np.arcsin(
        np.clip(2.0 * (df["qw"] * df["qy"] - df["qz"] * df["qx"]), -1.0, 1.0)
    )
    df["yaw"] = np.arctan2(
        2.0 * (df["qw"] * df["qz"] + df["qx"] * df["qy"]),
        1.0 - 2.0 * (df["qy"]**2 + df["qz"]**2)
    )
    df["moving_left"]  = df["gz_smooth"] > GZ_THRESHOLD
    df["moving_right"] = df["gz_smooth"] < -GZ_THRESHOLD
    df["idle"]         = df["acc_mag"] < IDLE_THRESHOLD
    return df


# ---------------------------------------------------------------------------
# Deteccion de ciclos
# ---------------------------------------------------------------------------

def detectar_ciclos(df):
    segments, in_seg, start = [], False, 0
    for i in range(len(df)):
        v = df["moving_left"].iloc[i]
        if v and not in_seg:
            start, in_seg = i, True
        elif not v and in_seg:
            segments.append((start, i))
            in_seg = False

    valid = [(s, e) for s, e in segments
             if df["time"].iloc[e] - df["time"].iloc[s] > MIN_SEGMENT_TIME]

    cycles = []
    for i in range(len(valid) - 1):
        s = valid[i][0]
        e = valid[i + 1][0]
        cyc = df.iloc[s:e].copy()
        has_right = (cyc["gz_smooth"] < -GZ_THRESHOLD).sum() > 20
        has_left  = (cyc["gz_smooth"] > GZ_THRESHOLD).sum() > 20
        if has_right or has_left:
            cycles.append(cyc)

    return cycles, valid


# ---------------------------------------------------------------------------
# Metricas por ciclo
# ---------------------------------------------------------------------------

def calcular_metricas(cycles):
    rows = []
    cycle_count = 0
    for i, cyc in enumerate(cycles):
        t_start = float(cyc["time"].iloc[0])
        t_end   = float(cyc["time"].iloc[-1])
        dur     = t_end - t_start
        if dur < 5.0:
            continue

        dt_arr       = cyc["time"].diff().fillna(0).values
        effort       = float(cyc["acc_mag"].sum())
        pitch_range  = float(cyc["pitch"].max() - cyc["pitch"].min())
        smoothness   = float(cyc["acc_mag"].std())
        dt_mean      = float(cyc["time"].diff().mean())
        lifting_mask = (cyc["pitch"].diff() > 0).values
        lifting_time = float(lifting_mask.sum() * dt_mean
                             if not np.isnan(dt_mean) else 0.0)
        acc_dyn      = cyc["acc_dynamic"].values
        weight_proxy = float(np.sum(acc_dyn[lifting_mask] * dt_arr[lifting_mask]))
        side = "izquierda" if (cycle_count // CICLOS_POR_LADO) % 2 == 0 else "derecha"
        eff  = (pitch_range * effort) / dur if dur > 0 else 0.0
        cost = dur * effort

        rows.append({
            "cycle": cycle_count, "t_start": t_start, "t_end": t_end,
            "duration": dur, "effort": effort, "pitch_range": pitch_range,
            "smoothness": smoothness, "lifting_time": lifting_time,
            "weight_proxy": weight_proxy, "side": side,
            "efficiency": eff, "cost": cost,
        })
        cycle_count += 1

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    res = res.reset_index(drop=True)
    res["cycle"] = res.index

    w_max   = res["weight_proxy"].max() or 1.0
    t_max   = res["duration"].max()     or 1.0
    eff_max = res["efficiency"].max()   or 1.0

    res["w_norm"]   = res["weight_proxy"] / w_max
    res["t_norm"]   = res["duration"]     / t_max
    res["eff_norm"] = res["efficiency"]   / eff_max

    t_inv      = 1.0 / res["t_norm"]
    t_inv_norm = t_inv / t_inv.max()
    res["score"] = 0.5 * res["w_norm"] + 0.3 * t_inv_norm + 0.2 * res["eff_norm"]

    res["tonnes"]             = res["weight_proxy"] * (BUCKET_TONNES / w_max)
    res["cycle_rate"]         = 3600.0 / res["duration"]
    res["productivity_cycle"] = res["cycle_rate"] * res["tonnes"]

    return res


# ---------------------------------------------------------------------------
# Productividad global
# ---------------------------------------------------------------------------

def calcular_productividad(df, cycles, res):
    total_t = float(df["time"].iloc[-1] - df["time"].iloc[0])
    total_h = total_t / 3600.0 if total_t > 0 else 1e-9
    cph     = len(res) / total_h

    dt     = float(df["time"].diff().mean())
    idle_t = float(df["idle"].sum()) * (dt if not np.isnan(dt) else 0.0)
    idle_r = idle_t / total_t if total_t > 0 else 0.0
    util   = max(0.0, 1.0 - idle_r)

    valid_cycles = [cycles[int(r["cycle"])] for _, r in res.iterrows()
                    if int(r["cycle"]) < len(cycles)]
    gaps = [float(valid_cycles[i+1]["time"].iloc[0] - valid_cycles[i]["time"].iloc[-1])
            for i in range(len(valid_cycles) - 1)]
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])

    cv = (res["duration"].std() / res["duration"].mean()
          if len(res) > 1 and res["duration"].mean() > 0 else 0.0)

    avg_tonnes     = float(res["tonnes"].mean())
    tph            = cph * avg_tonnes * util
    total_cost     = float(res["cost"].sum())
    total_tonnes   = float(res["tonnes"].sum()) or 1.0
    cost_per_tonne = total_cost / total_tonnes

    valid_dur = res["duration"][res["duration"] >= 10.0]
    best_cycle_sec = float(valid_dur.min()) if len(valid_dur) > 0 else float(res["duration"].min())

    return {
        "total_time_sec":   total_t,
        "total_cycles":     len(res),
        "cycles_per_hour":  cph,
        "utilization_pct":  util * 100,
        "idle_ratio_pct":   idle_r * 100,
        "avg_effort":       float(res["effort"].mean()),
        "consistency_cv":   float(cv),
        "avg_cycle_sec":    float(res["duration"].mean()),
        "best_cycle_sec":   best_cycle_sec,
        "worst_cycle_sec":  float(res["duration"].max()),
        "median_gap_sec":   float(np.median(gaps_arr)),
        "p90_gap_sec":      float(np.percentile(gaps_arr, 90)),
        "gaps_arr":         gaps_arr,
        "avg_tonnes":       avg_tonnes,
        "tonnes_per_hour":  tph,
        "total_cost":       total_cost,
        "cost_per_tonne":   cost_per_tonne,
        "total_tonnes":     total_tonnes,
    }


# ---------------------------------------------------------------------------
# Ciclos optimos
# ---------------------------------------------------------------------------

def calcular_ciclos_optimos(res):
    best_real_idx = res["score"].idxmax()
    best_real     = res.loc[best_real_idx]

    res_valid = res[res["duration"] >= 10.0]
    if len(res_valid) == 0:
        res_valid = res

    t_ideal   = float(np.percentile(res_valid["duration"],     IDEAL_DURATION_PERCENTILE))
    w_ideal   = float(np.percentile(res_valid["weight_proxy"], IDEAL_WEIGHT_PERCENTILE))
    eff_ideal = float(np.percentile(res_valid["efficiency"],   IDEAL_WEIGHT_PERCENTILE))

    w_max        = float(res["weight_proxy"].max()) or 1.0
    tonnes_ideal = w_ideal * (BUCKET_TONNES / w_max)
    cph_ideal    = 3600.0 / t_ideal if t_ideal > 0 else 0.0
    tph_ideal    = cph_ideal * tonnes_ideal

    return {
        "best_real": best_real,
        "ideal": {
            "weight_proxy": w_ideal,
            "duration":     t_ideal,
            "efficiency":   eff_ideal,
            "tonnes":       tonnes_ideal,
            "cph":          cph_ideal,
            "tph":          tph_ideal,
            "note": f"p{IDEAL_DURATION_PERCENTILE} dur, p{IDEAL_WEIGHT_PERCENTILE} peso",
        }
    }


# ---------------------------------------------------------------------------
# Escenario optimizado
# ---------------------------------------------------------------------------

def calcular_optimizado(real, res):
    gaps_arr  = real["gaps_arr"]
    med_gap   = real["median_gap_sec"]
    opt_gaps  = np.clip(gaps_arr, None, med_gap)
    gap_red   = float(gaps_arr.mean() - opt_gaps.mean())

    avg_cyc   = real["avg_cycle_sec"] if real["avg_cycle_sec"] > 0 else 1.0
    extra_cph = gap_red * real["cycles_per_hour"] / avg_cyc

    opt_util  = max(real["utilization_pct"], TARGET_UTILIZATION_PCT)
    opt_cph   = real["cycles_per_hour"] + extra_cph
    opt_cph  += opt_cph * (opt_util - real["utilization_pct"]) / 100.0

    res_valid = res[res["duration"] >= 10.0]
    if len(res_valid) >= 4:
        opt_avg_cycle = float(np.percentile(res_valid["duration"], 25))
    else:
        opt_avg_cycle = real["best_cycle_sec"] * 1.05

    opt_cph_from_cycle = 3600.0 / opt_avg_cycle if opt_avg_cycle > 0 else opt_cph
    opt_cph_final      = min(opt_cph, opt_cph_from_cycle)

    real_tph  = real["tonnes_per_hour"]
    opt_tph   = opt_cph_final * real["avg_tonnes"] * (opt_util / 100.0)
    delta_tph = opt_tph - real_tph

    return {
        "cycles_per_hour":       opt_cph_final,
        "utilization_pct":       opt_util,
        "avg_cycle_sec":         opt_avg_cycle,
        "gap_reduction_sec":     gap_red,
        "extra_cycles_per_hour": extra_cph,
        "real_tph":              real_tph,
        "opt_tph":               opt_tph,
        "delta_tph":             delta_tph,
        "delta_tpd":             delta_tph * HORAS_OPERACION,
        "delta_pct":             (delta_tph / real_tph * 100) if real_tph > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Recomendaciones
# ---------------------------------------------------------------------------

def generar_insights(res, real, opt, optimos):
    ins = []
    if real["consistency_cv"] > 0.15:
        ins.append(f"Alta variabilidad de ciclos (CV={real['consistency_cv']:.2f}) — estandarizar tecnica.")
    if real["idle_ratio_pct"] > 15:
        ins.append(f"{real['idle_ratio_pct']:.0f}% tiempo muerto — mejorar coordinacion con camiones.")
    if real["utilization_pct"] < TARGET_UTILIZATION_PCT:
        ins.append(f"Utilizacion {real['utilization_pct']:.0f}% vs objetivo {TARGET_UTILIZATION_PCT:.0f}%.")
    if real["median_gap_sec"] > TARGET_MAX_GAP_SEC:
        ins.append(f"Gap mediano {real['median_gap_sec']:.1f}s > objetivo {TARGET_MAX_GAP_SEC}s.")
    if real["p90_gap_sec"] > real["median_gap_sec"] * 2:
        ins.append(f"P90 gaps ({real['p90_gap_sec']:.1f}s) >> mediana — esperas prolongadas recurrentes.")

    ideal        = optimos["ideal"]
    eff_real_avg = float(res["efficiency"].mean())
    eff_ideal_v  = (ideal["weight_proxy"] * ideal["efficiency"]) / ideal["duration"] \
                   if ideal["duration"] > 0 else 0
    improvement  = (eff_ideal_v / eff_real_avg - 1) * 100 if eff_real_avg > 0 else 0

    ins.append(
        f"El operador ya demostro capacidad maxima en ciclos separados. "
        f"Estandarizar mejores practicas podria liberar +{improvement:.0f}% de eficiencia."
    )
    ins.append(
        f"Optimizando logistica: +{opt['delta_tph']:.0f} t/hr (+{opt['delta_pct']:.0f}%) "
        f"= +{opt['delta_tpd']:.0f} t/dia ({HORAS_OPERACION}h)."
    )
    return ins


# ---------------------------------------------------------------------------
# Exportar JSON para dashboard HTML
# ---------------------------------------------------------------------------

def exportar_dashboard_json(df_proc, cycles, res, real, opt, optimos, insights):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "dashboard_data.json"

    def _clean(v):
        if isinstance(v, (np.floating, float)):
            return round(float(v), 6)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, bool):
            return bool(v)
        return v

    # Submuestrear señales IMU a ~800 puntos
    n_total = len(df_proc)
    step = max(1, n_total // 800)
    df_sub = df_proc.iloc[::step].copy()

    t_arr  = df_proc["time"].values
    gz_arr = df_proc["gz_smooth"].values
    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    gz_cum = np.cumsum(gz_arr * dt_arr)
    gz_cum_sub = gz_cum[::step]

    signals = {
        "time":        [round(float(v), 3) for v in df_sub["time"].values],
        "gz_smooth":   [round(float(v), 4) for v in df_sub["gz_smooth"].values],
        "pitch":       [round(float(v), 4) for v in df_sub["pitch"].values],
        "acc_dynamic": [round(float(v), 4) for v in df_sub["acc_dynamic"].values],
        "acc_mag":     [round(float(v), 4) for v in df_sub["acc_mag"].values],
        "gz_cumsum":   [round(float(v), 4) for v in gz_cum_sub],
    }

    # Señal del mejor ciclo
    best_real = optimos["best_real"]
    best_idx  = int(best_real["cycle"])
    best_cyc_data = cycles[best_idx] if best_idx < len(cycles) else cycles[-1]
    t_rel = (best_cyc_data["time"] - best_cyc_data["time"].iloc[0]).values
    dyn_max = float(best_cyc_data["acc_dynamic"].max()) or 1.0
    best_cycle_signal = {
        "time":        [round(float(v), 3) for v in t_rel],
        "gz_smooth":   [round(float(v), 4) for v in best_cyc_data["gz_smooth"].values],
        "pitch":       [round(float(v), 4) for v in best_cyc_data["pitch"].values],
        "acc_dynamic_norm": [round(float(v) / dyn_max, 4) for v in best_cyc_data["acc_dynamic"].values],
    }

    # Angulo total girado por ciclo
    ang_per_cycle = []
    for _, row in res.iterrows():
        t_s = row["t_start"]
        t_e = row["t_end"]
        mask = (t_arr >= t_s) & (t_arr <= t_e)
        gz_c  = np.abs(gz_arr[mask])
        dt_c  = np.diff(t_arr[mask], prepend=t_arr[mask][0]) if mask.sum() > 0 else np.array([0])
        total = float(np.sum(gz_c * dt_c))
        ang_per_cycle.append(round(total, 3))

    cycle_starts = [round(float(row["t_start"]), 3) for _, row in res.iterrows()]
    cycle_sides  = [row["side"] for _, row in res.iterrows()]

    # Gaps
    gaps_arr = real["gaps_arr"]
    opt_gaps = np.clip(gaps_arr, None, real["median_gap_sec"])

    # Produccion acumulada
    n = real["total_cycles"]
    t_r = real["tonnes_per_hour"] / real["cycles_per_hour"] if real["cycles_per_hour"] > 0 else 0
    t_o = opt["opt_tph"]  / opt["cycles_per_hour"]          if opt["cycles_per_hour"]  > 0 else 0
    t_i = optimos["ideal"]["tph"] / optimos["ideal"]["cph"] if optimos["ideal"]["cph"] > 0 else 0
    x_cyc = list(range(1, n + 1))
    prod_acum = {
        "cycles": x_cyc,
        "real":   [round(i * t_r, 2) for i in x_cyc],
        "opt":    [round(i * t_o, 2) for i in x_cyc],
        "ideal":  [round(i * t_i, 2) for i in x_cyc],
    }

    # Estadisticas descriptivas
    stats_cols = ["duration","effort","pitch_range","smoothness",
                  "lifting_time","tonnes","efficiency","cost","score"]
    stats_df = res[stats_cols].describe().loc[["mean","std","min","25%","50%","75%","max"]]
    stats_dict = {}
    for col in stats_cols:
        stats_dict[col] = {stat: round(float(stats_df.loc[stat, col]), 4)
                           for stat in stats_df.index}

    # Todos los ciclos
    cycles_data = []
    for _, row in res.iterrows():
        d = {}
        for k, v in row.items():
            if k in ("t_start", "t_end"):
                continue
            d[k] = _clean(v)
        cycles_data.append(d)

    ideal = optimos["ideal"]
    best_real_dict = {k: _clean(v) for k, v in best_real.items()
                      if k not in ("t_start", "t_end")}

    data = {
        "meta": {
            "generated_at":    str(pd.Timestamp.now()),
            "total_samples":   int(len(df_proc)),
            "duration_sec":    round(float(real["total_time_sec"]), 2),
            "duration_min":    round(float(real["total_time_sec"]) / 60, 2),
            "sample_freq_hz":  round(float(1 / df_proc["time"].diff().mean()), 2),
            "horas_operacion": HORAS_OPERACION,
            "bucket_tonnes":   BUCKET_TONNES,
        },
        "signals":            signals,
        "best_cycle_signal":  best_cycle_signal,
        "ang_per_cycle":      ang_per_cycle,
        "cycle_starts":       cycle_starts,
        "cycle_sides":        cycle_sides,
        "gaps_real":          [round(float(v), 4) for v in gaps_arr],
        "gaps_opt":           [round(float(v), 4) for v in opt_gaps],
        "prod_acum":          prod_acum,
        "stats":              stats_dict,
        "real":               {k: _clean(v) for k, v in real.items() if k != "gaps_arr"},
        "opt":                {k: _clean(v) for k, v in opt.items()},
        "ideal":              {k: _clean(v) for k, v in ideal.items() if k != "note"},
        "ideal_note":         ideal.get("note", ""),
        "best_real":          best_real_dict,
        "all_cycles":         cycles_data,
        "recommendations":    insights,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] Dashboard JSON guardado en: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exporta dashboard_data.json para el dashboard HTML"
    )
    parser.add_argument("--imu", type=str, default=None,
                        help="Ruta explicita al archivo .npy (opcional)")
    args = parser.parse_args()

    imu_path = find_imu_file(args.imu)
    print(f"[INFO] Cargando IMU: {imu_path}")
    df = load_imu(imu_path)
    print(f"[INFO] {len(df)} muestras, {df['time'].iloc[-1]:.1f}s de duracion")

    df_proc = preprocess(df)
    cycles, valid_segments = detectar_ciclos(df_proc)
    res = calcular_metricas(cycles)

    if len(res) == 0:
        print("No se detectaron ciclos validos.")
        return

    real     = calcular_productividad(df_proc, cycles, res)
    opt      = calcular_optimizado(real, res)
    optimos  = calcular_ciclos_optimos(res)
    insights = generar_insights(res, real, opt, optimos)

    exportar_dashboard_json(df_proc, cycles, res, real, opt, optimos, insights)
    print("[INFO] Ahora abre dashboard.html en tu navegador.")


if __name__ == "__main__":
    main()
