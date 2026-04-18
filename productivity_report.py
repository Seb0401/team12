"""
EX-5600 Shovel — Dashboard Interactivo de Productividad
========================================================
5 vistas navegables con botones:

  [1] Metricas por Ciclo   — tabla detallada + explicacion de columnas
  [2] Relacion Ciclos      — estadisticas descriptivas + gaps entre ciclos
  [3] Productividad Global — KPIs OEE, toneladas, costo, idle
  [4] Ciclo Optimo         — mejor real vs ideal realista + frontera
  [5] Gap de Mejora        — escenario optimizado logistico

Uso:
    python productivity_report.py
    python productivity_report.py --imu ruta/al/archivo.npy
    python productivity_report.py --no-gui
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

INPUTS_DIR   = Path("./inputs")
OUTPUTS_DIR  = Path("./outputs")
IMU_GLOB     = "*imu*.npy"
FALLBACK_IMU = Path("C:/Hackathon/Archive/40343737_20260313_110600_to_112100_imu.npy")

GZ_THRESHOLD     = 0.4
MIN_SEGMENT_TIME = 2.5
GZ_SMOOTH_WINDOW = 50
IDLE_THRESHOLD   = 0.3
BUCKET_TONNES    = 52.0
HORAS_OPERACION  = 20

TARGET_UTILIZATION_PCT = 85.0
TARGET_MAX_GAP_SEC     = 8.0

# Percentil para ciclo ideal realista (evita outliers de duracion minima)
IDEAL_DURATION_PERCENTILE = 10   # p10 de duracion = rapido pero alcanzable
IDEAL_WEIGHT_PERCENTILE   = 90   # p90 de peso = carga alta pero alcanzable

DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
BTN_BG   = "#0f3460"
BTN_ACT  = "#c0392b"
ACCENT   = "#00d4ff"
GREEN    = "#00ff88"
RED      = "#ff6b6b"
YELLOW   = "#ffd700"
ORANGE   = "#ff9f43"
PURPLE   = "#a29bfe"
WHITE    = "#ffffff"
GRAY     = "#aaaaaa"


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
        s, e = valid[i][0], valid[i + 1][0]
        cyc = df.iloc[s:e].copy()
        if (cyc["gz_smooth"] < -GZ_THRESHOLD).sum() > 20 or \
           (cyc["gz_smooth"] > GZ_THRESHOLD).sum() > 20:
            cycles.append(cyc)

    return cycles, valid


# ---------------------------------------------------------------------------
# Metricas por ciclo
# ---------------------------------------------------------------------------

def calcular_metricas(cycles):
    rows = []
    for i, cyc in enumerate(cycles):
        t_start = cyc["time"].iloc[0]
        t_end   = cyc["time"].iloc[-1]
        dur     = t_end - t_start
        dt_arr  = cyc["time"].diff().fillna(0).values

        effort       = float(cyc["acc_mag"].sum())
        pitch_range  = float(cyc["pitch"].max() - cyc["pitch"].min())
        smoothness   = float(cyc["acc_mag"].std())
        dt_mean      = cyc["time"].diff().mean()
        lifting_mask = cyc["pitch"].diff() > 0
        lifting_time = float(lifting_mask.sum() * dt_mean
                             if not np.isnan(dt_mean) else 0.0)

        acc_dyn      = cyc["acc_dynamic"].values
        weight_proxy = float(np.sum(acc_dyn[lifting_mask.values] *
                                    dt_arr[lifting_mask.values]))

        side = "izquierda" if cyc["gz_smooth"].mean() > 0 else "derecha"
        eff  = (pitch_range * effort) / dur if dur > 0 else 0.0
        cost = dur * effort

        rows.append({
            "cycle":        i,
            "t_start":      t_start,
            "t_end":        t_end,
            "duration":     dur,
            "effort":       effort,
            "pitch_range":  pitch_range,
            "smoothness":   smoothness,
            "lifting_time": lifting_time,
            "weight_proxy": weight_proxy,
            "side":         side,
            "efficiency":   eff,
            "cost":         cost,
        })

    res = pd.DataFrame(rows)
    if len(res) == 0:
        return res

    w_max   = res["weight_proxy"].max() or 1.0
    t_max   = res["duration"].max()     or 1.0
    eff_max = res["efficiency"].max()   or 1.0

    res["w_norm"]   = res["weight_proxy"] / w_max
    res["t_norm"]   = res["duration"]     / t_max
    res["eff_norm"] = res["efficiency"]   / eff_max

    t_inv_norm   = (1.0 / res["t_norm"]) / (1.0 / res["t_norm"]).max()
    res["score"] = (0.5 * res["w_norm"] +
                    0.3 * t_inv_norm +
                    0.2 * res["eff_norm"])

    res["tonnes"]             = res["weight_proxy"] * (BUCKET_TONNES / w_max)
    res["cycle_rate"]         = 3600.0 / res["duration"].replace(0, np.nan)
    res["productivity_cycle"] = res["cycle_rate"] * res["tonnes"]

    return res


# ---------------------------------------------------------------------------
# Productividad global
# ---------------------------------------------------------------------------

def calcular_productividad(df, cycles, res):
    total_t = df["time"].iloc[-1] - df["time"].iloc[0]
    total_h = total_t / 3600.0 if total_t > 0 else 1e-9
    cph     = len(cycles) / total_h

    dt     = df["time"].diff().mean()
    idle_t = df["idle"].sum() * (dt if not np.isnan(dt) else 0.0)
    idle_r = idle_t / total_t if total_t > 0 else 0.0
    util   = max(0.0, 1.0 - idle_r)

    gaps = [cycles[i+1]["time"].iloc[0] - cycles[i]["time"].iloc[-1]
            for i in range(len(cycles) - 1)]
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])

    cv = (res["duration"].std() / res["duration"].mean()
          if len(res) > 1 and res["duration"].mean() > 0 else 0.0)

    avg_tonnes     = res["tonnes"].mean() if len(res) > 0 else 0.0
    tph            = cph * avg_tonnes * util
    total_cost     = res["cost"].sum()
    total_tonnes   = res["tonnes"].sum() or 1.0
    cost_per_tonne = total_cost / total_tonnes
    avg_w_proxy    = res["weight_proxy"].mean() if len(res) > 0 else 0.0
    prod_proxy     = avg_w_proxy * cph * util

    return {
        "total_time_sec":   total_t,
        "total_cycles":     len(cycles),
        "cycles_per_hour":  cph,
        "utilization_pct":  util * 100,
        "idle_ratio_pct":   idle_r * 100,
        "avg_effort":       res["effort"].mean() if len(res) > 0 else 0.0,
        "consistency_cv":   cv,
        "avg_cycle_sec":    res["duration"].mean() if len(res) > 0 else 0.0,
        "best_cycle_sec":   res["duration"].min()  if len(res) > 0 else 0.0,
        "worst_cycle_sec":  res["duration"].max()  if len(res) > 0 else 0.0,
        "median_gap_sec":   float(np.median(gaps_arr)),
        "p90_gap_sec":      float(np.percentile(gaps_arr, 90)),
        "gaps_arr":         gaps_arr,
        "avg_tonnes":       avg_tonnes,
        "tonnes_per_hour":  tph,
        "total_cost":       total_cost,
        "cost_per_tonne":   cost_per_tonne,
        "prod_proxy":       prod_proxy,
        "total_tonnes":     total_tonnes,
    }


# ---------------------------------------------------------------------------
# Ciclos optimos — ideal REALISTA (percentiles, no extremos absolutos)
# ---------------------------------------------------------------------------

def calcular_ciclos_optimos(res):
    """
    Mejor ciclo real: argmax(score)

    Ciclo ideal realista:
      - Duracion: percentil 10 (rapido pero alcanzable, no el minimo absoluto)
      - Peso:     percentil 90 (carga alta pero alcanzable)
      - Eficiencia: percentil 90
    """
    best_real_idx = res["score"].idxmax()
    best_real     = res.loc[best_real_idx]

    # Ideal realista con percentiles
    t_ideal   = float(np.percentile(res["duration"],     IDEAL_DURATION_PERCENTILE))
    w_ideal   = float(np.percentile(res["weight_proxy"], IDEAL_WEIGHT_PERCENTILE))
    eff_ideal = float(np.percentile(res["efficiency"],   IDEAL_WEIGHT_PERCENTILE))

    # Toneladas proporcionales al peso ideal
    w_max         = res["weight_proxy"].max() or 1.0
    tonnes_ideal  = w_ideal * (BUCKET_TONNES / w_max)
    cph_ideal     = 3600.0 / t_ideal if t_ideal > 0 else 0.0
    tph_ideal     = cph_ideal * tonnes_ideal

    return {
        "best_real": best_real,
        "ideal": {
            "weight_proxy": w_ideal,
            "duration":     t_ideal,
            "efficiency":   eff_ideal,
            "tonnes":       tonnes_ideal,
            "cph":          cph_ideal,
            "tph":          tph_ideal,
            "note": (f"p{IDEAL_DURATION_PERCENTILE} duracion, "
                     f"p{IDEAL_WEIGHT_PERCENTILE} peso/eficiencia"),
        }
    }


# ---------------------------------------------------------------------------
# Escenario optimizado logistico
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

    # Ciclo objetivo: percentil 25 (conservador)
    if len(res) >= 4:
        opt_avg_cycle = float(np.percentile(res["duration"], 25))
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
    eff_real_avg = res["efficiency"].mean()
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
# Consola
# ---------------------------------------------------------------------------

SEP  = "=" * 72
SEP2 = "-" * 72

def _h(t): print(f"\n{SEP}\n  {t}\n{SEP}")
def _s(t): print(f"\n{SEP2}\n  {t}\n{SEP2}")

def imprimir_consola(df_proc, cycles, res, real, opt, optimos, insights):
    ideal     = optimos["ideal"]
    best_real = optimos["best_real"]

    _h("1. DATOS IMU — RESUMEN")
    print(f"  Muestras:      {len(df_proc)}")
    print(f"  Duracion:      {real['total_time_sec']:.1f} s  ({real['total_time_sec']/60:.1f} min)")
    print(f"  Frecuencia:    {1/df_proc['time'].diff().mean():.1f} Hz")
    for col, lbl in [("acc_mag","acc_mag"), ("acc_dynamic","acc_dynamic"),
                     ("gz_smooth","gz_smooth"), ("pitch","pitch"), ("yaw","yaw")]:
        s = df_proc[col]
        print(f"  {lbl:<12} media={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")
    print(f"  idle:          {df_proc['idle'].mean()*100:.1f}% del tiempo")

    _h("2. DETECCION DE CICLOS")
    print(f"  Ciclos detectados: {real['total_cycles']}")

    _h("3. METRICAS POR CICLO")
    print("""
  Columnas:
    Dur(s)     = t_fin - t_inicio
    Esfuerzo   = sum(acc_mag)
    Pitch      = max(pitch) - min(pitch)
    Suavidad   = std(acc_mag)
    T.Elev(s)  = tiempo con diff(pitch) > 0
    Peso(t)    = sum(acc_dynamic * dt) en fase lifting, escalado a toneladas
    Eficiencia = (Pitch * Esfuerzo) / Duracion
    Costo      = Duracion * Esfuerzo
    Score      = 0.5*W_norm + 0.3*(1/T_norm) + 0.2*Eff_norm
    Prod(t/hr) = (3600/Dur) * Peso(t)
""")
    fmt = ("{:>4}  {:>7}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  "
           "{:>7}  {:>8}  {:>6}  {:>10}")
    hdr = fmt.format("#","Dur(s)","Esfuerzo","Pitch","Suavid",
                     "T.Elev","Peso(t)","Efic","Costo","Score","Prod(t/hr)")
    print(f"  {hdr}")
    print(f"  {'-'*len(hdr)}")
    for _, row in res.iterrows():
        print("  " + fmt.format(
            int(row["cycle"]),
            f"{row['duration']:.1f}",
            f"{row['effort']:.0f}",
            f"{row['pitch_range']:.3f}",
            f"{row['smoothness']:.3f}",
            f"{row['lifting_time']:.1f}",
            f"{row['tonnes']:.1f}",
            f"{row['efficiency']:.2f}",
            f"{row['cost']:.0f}",
            f"{row['score']:.3f}",
            f"{row['productivity_cycle']:.0f}",
        ))

    _s("Estadisticas de ciclos")
    print(res[["duration","effort","pitch_range","smoothness",
               "lifting_time","tonnes","efficiency","cost","score"]]
          .describe().loc[["mean","std","min","25%","50%","75%","max"]]
          .to_string(float_format=lambda x: f"{x:.3f}"))

    _s("Gaps entre ciclos (s)")
    g = real["gaps_arr"]
    print(f"  N={len(g)}  media={g.mean():.2f}  mediana={np.median(g):.2f}  "
          f"P90={np.percentile(g,90):.2f}  max={g.max():.2f}  min={g.min():.2f}")
    for i, v in enumerate(g):
        flag = "  <-- ALTO" if v > TARGET_MAX_GAP_SEC else ""
        print(f"    Intervalo {i+1:>3}: {v:6.2f} s{flag}")

    _h("4. PRODUCTIVIDAD GLOBAL")
    print(f"  Ciclos/hora:          {real['cycles_per_hour']:.2f}")
    print(f"  Utilizacion:          {real['utilization_pct']:.1f}%")
    print(f"  Tiempo muerto:        {real['idle_ratio_pct']:.1f}%")
    print(f"  Duracion media:       {real['avg_cycle_sec']:.2f} s")
    print(f"  Mejor ciclo:          {real['best_cycle_sec']:.2f} s")
    print(f"  Peor ciclo:           {real['worst_cycle_sec']:.2f} s")
    print(f"  Consistencia (CV):    {real['consistency_cv']:.3f}")
    print(f"  Gap mediano:          {real['median_gap_sec']:.2f} s")
    print(f"  Gap P90:              {real['p90_gap_sec']:.2f} s")
    print(f"  Toneladas/ciclo avg:  {real['avg_tonnes']:.2f} t")
    print(f"  Toneladas totales:    {real['total_tonnes']:.1f} t")
    print(f"  Toneladas/hora:       {real['tonnes_per_hour']:.2f} t/hr")
    print(f"  Toneladas/dia ({HORAS_OPERACION}h):   {real['tonnes_per_hour']*HORAS_OPERACION:.0f} t")
    print(f"  Costo total:          {real['total_cost']:.0f}")
    print(f"  Costo/tonelada:       {real['cost_per_tonne']:.4f}")

    _h("5. MEJOR CICLO REAL")
    print(f"  Ciclo #:              {int(best_real['cycle'])}")
    print(f"  Score:                {best_real['score']:.4f}")
    print(f"  Duracion:             {best_real['duration']:.2f} s")
    print(f"  Toneladas:            {best_real['tonnes']:.2f} t")
    print(f"  Eficiencia:           {best_real['efficiency']:.4f}")
    print(f"  Peso proxy:           {best_real['weight_proxy']:.4f}")
    print(f"  Ciclos/hr (ciclo):    {3600/best_real['duration']:.1f}")
    print(f"  Prod. ciclo (t/hr):   {best_real['productivity_cycle']:.1f}")

    _h("6. CICLO IDEAL REALISTA")
    print(f"  Nota: {ideal['note']}")
    print(f"  Duracion (p{IDEAL_DURATION_PERCENTILE}):       {ideal['duration']:.2f} s")
    print(f"  Peso proxy (p{IDEAL_WEIGHT_PERCENTILE}):      {ideal['weight_proxy']:.4f}")
    print(f"  Eficiencia (p{IDEAL_WEIGHT_PERCENTILE}):      {ideal['efficiency']:.4f}")
    print(f"  Toneladas:            {ideal['tonnes']:.2f} t")
    print(f"  Ciclos/hr ideal:      {ideal['cph']:.2f}")
    print(f"  Toneladas/hr ideal:   {ideal['tph']:.2f} t/hr")
    print(f"  Toneladas/dia ideal:  {ideal['tph']*HORAS_OPERACION:.0f} t")

    _h("7. GAP DE MEJORA Y ESCENARIO OPTIMIZADO")
    print(f"  --- Logistico (reduccion gaps + utilizacion) ---")
    print(f"  Reduccion gaps:       {opt['gap_reduction_sec']:.2f} s/ciclo")
    print(f"  Ciclos extra/hr:      +{opt['extra_cycles_per_hour']:.2f}")
    print(f"  Ciclos/hr opt.:       {opt['cycles_per_hour']:.2f}")
    print(f"  Utilizacion obj.:     {opt['utilization_pct']:.1f}%")
    print(f"  Ciclo prom. obj.:     {opt['avg_cycle_sec']:.2f} s")
    print(f"  t/hr opt.:            {opt['opt_tph']:.2f}")
    print(f"  Delta t/hr:           +{opt['delta_tph']:.2f} (+{opt['delta_pct']:.1f}%)")
    print(f"  Delta t/dia:          +{opt['delta_tpd']:.0f} t")
    print(f"\n  --- Ideal casuistico ---")
    delta_tph_ideal = ideal["tph"] - real["tonnes_per_hour"]
    print(f"  Delta t/hr vs real:   +{delta_tph_ideal:.2f} t/hr")
    print(f"  Delta t/dia vs real:  +{delta_tph_ideal*HORAS_OPERACION:.0f} t")

    _h("8. RECOMENDACIONES")
    for i, ins in enumerate(insights, 1):
        print(f"  {i}. {ins}")
    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Helpers GUI
# ---------------------------------------------------------------------------

def _style_ax(ax, title="", xlabel="", ylabel="", ts=10):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    if title:
        ax.set_title(title, fontsize=ts, color=WHITE, pad=5, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color=GRAY)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color=GRAY)


def _text_panel(ax, title, tc=YELLOW, bc=ACCENT):
    ax.set_facecolor(BTN_BG)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(bc); sp.set_linewidth(1.5)
    ax.set_title(title, fontsize=10, color=tc, fontweight="bold", pad=5)


# ---------------------------------------------------------------------------
# VISTA 1 — Metricas por Ciclo
# ---------------------------------------------------------------------------

def vista_metricas_ciclo(fig, cr, res):
    gs = gridspec.GridSpec(2, 2, left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3], hspace=0.45, wspace=0.35)
    axes = []

    # ── Tabla de ciclos ────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, :])
    axes.append(ax_tbl)
    _text_panel(ax_tbl, "Metricas por Ciclo — Todos los Ciclos", YELLOW, YELLOW)

    headers = ["#", "Dur(s)", "Esfuerzo", "Pitch", "Suavidad",
               "T.Elev(s)", "Peso(t)", "Eficiencia", "Costo", "Score", "Lado"]
    col_x   = [0.00, 0.07, 0.16, 0.26, 0.35, 0.44, 0.53, 0.62, 0.72, 0.82, 0.92]

    for hdr, cx in zip(headers, col_x):
        ax_tbl.text(cx, 0.95, hdr, transform=ax_tbl.transAxes,
                    fontsize=7, color=YELLOW, fontweight="bold", va="top")
    ax_tbl.axhline(0.90, color="#555", linewidth=0.8, transform=ax_tbl.transAxes)

    n_show = min(len(res), 12)
    row_h  = 0.86 / max(n_show, 1)
    for rank, (_, row) in enumerate(res.head(n_show).iterrows()):
        y = 0.88 - rank * row_h
        bg = "#1e2d4a" if rank % 2 == 0 else BTN_BG
        ax_tbl.axhspan(y - row_h * 0.85, y + row_h * 0.1,
                       xmin=0, xmax=1, color=bg, transform=ax_tbl.transAxes)
        sc_color = GREEN if row["score"] >= res["score"].mean() else ORANGE
        vals = [
            (f"{int(row['cycle'])}",      ACCENT),
            (f"{row['duration']:.1f}",    WHITE),
            (f"{row['effort']:.0f}",      WHITE),
            (f"{row['pitch_range']:.3f}", WHITE),
            (f"{row['smoothness']:.3f}",  WHITE),
            (f"{row['lifting_time']:.1f}",WHITE),
            (f"{row['tonnes']:.1f}",      GREEN),
            (f"{row['efficiency']:.2f}",  WHITE),
            (f"{row['cost']:.0f}",        PURPLE),
            (f"{row['score']:.3f}",       sc_color),
            (row["side"],                 ORANGE),
        ]
        for (val, color), cx in zip(vals, col_x):
            ax_tbl.text(cx, y, val, transform=ax_tbl.transAxes,
                        fontsize=7, color=color, va="top", fontfamily="monospace")

    # ── Explicacion de columnas ────────────────────────────────────────────
    ax_exp = fig.add_subplot(gs[1, 0])
    axes.append(ax_exp)
    _text_panel(ax_exp, "Como se calcula cada columna", ACCENT, ACCENT)
    formulas = [
        ("Dur(s)",     "t_fin - t_inicio"),
        ("Esfuerzo",   "sum(acc_mag)"),
        ("Pitch",      "max(pitch) - min(pitch)"),
        ("Suavidad",   "std(acc_mag)"),
        ("T.Elev(s)",  "sum(dt) donde diff(pitch)>0"),
        ("Peso(t)",    "sum(acc_dyn*dt) en lifting, escalado"),
        ("Eficiencia", "(Pitch * Esfuerzo) / Duracion"),
        ("Costo",      "Duracion * Esfuerzo"),
        ("Score",      "0.5*W_norm + 0.3*(1/T_norm) + 0.2*Eff_norm"),
        ("Prod(t/hr)", "(3600/Dur) * Peso(t)"),
    ]
    y = 0.93
    for col, formula in formulas:
        ax_exp.text(0.03, y, f"{col:<12}", transform=ax_exp.transAxes,
                    fontsize=7.5, color=YELLOW, va="top", fontfamily="monospace")
        ax_exp.text(0.38, y, formula, transform=ax_exp.transAxes,
                    fontsize=7.5, color=WHITE, va="top")
        y -= 0.088

    # ── Score por ciclo ────────────────────────────────────────────────────
    ax_sc = fig.add_subplot(gs[1, 1])
    axes.append(ax_sc)
    _style_ax(ax_sc, "Score por ciclo", "Ciclo", "Score")
    colors_sc = [GREEN if s >= res["score"].mean() else RED for s in res["score"]]
    ax_sc.bar(res["cycle"], res["score"], color=colors_sc, width=0.7, alpha=0.9)
    ax_sc.axhline(res["score"].mean(), color=YELLOW, linestyle="--",
                  linewidth=1.2, label=f"media {res['score'].mean():.3f}")
    ax_sc.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# VISTA 2 — Relacion entre Ciclos
# ---------------------------------------------------------------------------

def vista_relacion_ciclos(fig, cr, res, real):
    gs = gridspec.GridSpec(2, 2, left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3], hspace=0.45, wspace=0.35)
    axes = []

    # ── Estadisticas descriptivas ──────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[0, 0])
    axes.append(ax_stats)
    _text_panel(ax_stats, "Estadisticas de Ciclos", YELLOW, ACCENT)

    stats_cols = ["duration", "effort", "pitch_range", "smoothness",
                  "lifting_time", "tonnes", "efficiency", "cost", "score"]
    stats = res[stats_cols].describe().loc[["mean","std","min","25%","50%","75%","max"]]

    col_labels = ["Dur", "Esf", "Pitch", "Suav", "T.El", "Ton", "Efic", "Cost", "Score"]
    row_labels = ["media", "std", "min", "p25", "p50", "p75", "max"]

    col_xs = [0.02 + i * 0.107 for i in range(len(col_labels))]
    row_ys = [0.90 - i * 0.12  for i in range(len(row_labels) + 1)]

    # Cabecera
    for lbl, cx in zip(col_labels, col_xs):
        ax_stats.text(cx, row_ys[0], lbl, transform=ax_stats.transAxes,
                      fontsize=6.5, color=YELLOW, fontweight="bold", va="top")
    ax_stats.axhline(row_ys[0] - 0.04, color="#555", linewidth=0.6,
                     transform=ax_stats.transAxes)

    for ri, (stat_name, row_y) in enumerate(zip(row_labels, row_ys[1:])):
        ax_stats.text(0.00, row_y, stat_name, transform=ax_stats.transAxes,
                      fontsize=6, color=GRAY, va="top")
        for ci, (col, cx) in enumerate(zip(stats_cols, col_xs)):
            val = stats.loc[stats.index[ri], col]
            ax_stats.text(cx, row_y, f"{val:.2f}", transform=ax_stats.transAxes,
                          fontsize=6, color=WHITE, va="top", fontfamily="monospace")

    # ── Gaps entre ciclos ──────────────────────────────────────────────────
    ax_gaps = fig.add_subplot(gs[0, 1])
    axes.append(ax_gaps)
    _style_ax(ax_gaps, "Gaps entre ciclos (s)", "Intervalo", "Segundos")
    gaps_arr = real["gaps_arr"]
    if len(gaps_arr) > 0:
        colors_g = [GREEN if g <= real["median_gap_sec"] else RED for g in gaps_arr]
        ax_gaps.bar(range(len(gaps_arr)), gaps_arr, color=colors_g, width=0.7, alpha=0.9)
        ax_gaps.axhline(real["median_gap_sec"], color=YELLOW, linestyle="--",
                        linewidth=1.2, label=f"mediana {real['median_gap_sec']:.1f}s")
        ax_gaps.axhline(TARGET_MAX_GAP_SEC, color=GREEN, linestyle=":",
                        linewidth=1.5, label=f"objetivo {TARGET_MAX_GAP_SEC}s")
        ax_gaps.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Duracion por ciclo ─────────────────────────────────────────────────
    ax_dur = fig.add_subplot(gs[1, 0])
    axes.append(ax_dur)
    _style_ax(ax_dur, "Duracion por ciclo", "Ciclo", "Segundos")
    colors_d = [GREEN if d <= res["duration"].quantile(0.5) else RED
                for d in res["duration"]]
    ax_dur.bar(res["cycle"], res["duration"], color=colors_d, width=0.7, alpha=0.9)
    ax_dur.axhline(res["duration"].mean(), color=YELLOW, linestyle="--",
                   linewidth=1.2, label=f"media {res['duration'].mean():.1f}s")
    ax_dur.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Histograma duracion ────────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 1])
    axes.append(ax_hist)
    _style_ax(ax_hist, "Distribucion de duracion", "Segundos", "Frecuencia")
    if len(res) >= 2:
        ax_hist.hist(res["duration"], bins=min(15, len(res)),
                     color=PURPLE, alpha=0.8, edgecolor="#333")
    ax_hist.axvline(res["duration"].mean(), color=YELLOW, linestyle="--",
                    linewidth=1.5, label=f"media {res['duration'].mean():.1f}s")
    ax_hist.axvline(res["duration"].median(), color=ACCENT, linestyle=":",
                    linewidth=1.5, label=f"mediana {res['duration'].median():.1f}s")
    ax_hist.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# VISTA 3 — Productividad Global
# ---------------------------------------------------------------------------

def vista_productividad_global(fig, cr, res, real):
    gs = gridspec.GridSpec(2, 3, left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3], hspace=0.45, wspace=0.35)
    axes = []

    # ── Panel KPIs ─────────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :2])
    axes.append(ax_kpi)
    _text_panel(ax_kpi, "Productividad Global — KPIs OEE", YELLOW, ACCENT)

    kpis = [
        ("Ciclos totales",       f"{real['total_cycles']}",                WHITE),
        ("Ciclos / hora",        f"{real['cycles_per_hour']:.2f}",         ACCENT),
        ("Utilizacion",          f"{real['utilization_pct']:.1f}%",
         GREEN if real["utilization_pct"] >= 75 else RED),
        ("Tiempo muerto",        f"{real['idle_ratio_pct']:.1f}%",
         GREEN if real["idle_ratio_pct"] <= 15 else RED),
        ("Duracion media",       f"{real['avg_cycle_sec']:.1f} s",         WHITE),
        ("Mejor ciclo",          f"{real['best_cycle_sec']:.1f} s",        GREEN),
        ("Peor ciclo",           f"{real['worst_cycle_sec']:.1f} s",       RED),
        ("Consistencia (CV)",    f"{real['consistency_cv']:.3f}",
         GREEN if real["consistency_cv"] <= 0.15 else ORANGE),
        ("Gap mediano",          f"{real['median_gap_sec']:.1f} s",
         GREEN if real["median_gap_sec"] <= TARGET_MAX_GAP_SEC else RED),
        ("Gap P90",              f"{real['p90_gap_sec']:.1f} s",           ORANGE),
        ("Toneladas / hora",     f"{real['tonnes_per_hour']:.1f} t",       YELLOW),
        ("Costo / tonelada",     f"{real['cost_per_tonne']:.3f}",          PURPLE),
    ]
    cols = 3
    for idx, (label, value, color) in enumerate(kpis):
        col = idx % cols
        row = idx // cols
        x = 0.03 + col * 0.34
        y = 0.88 - row * 0.22
        ax_kpi.text(x, y,        label, transform=ax_kpi.transAxes,
                    fontsize=7.5, color=GRAY, va="top")
        ax_kpi.text(x, y - 0.10, value, transform=ax_kpi.transAxes,
                    fontsize=13, color=color, va="top", fontweight="bold",
                    fontfamily="monospace")

    # ── Panel toneladas/dia ────────────────────────────────────────────────
    ax_tpd = fig.add_subplot(gs[0, 2])
    axes.append(ax_tpd)
    ax_tpd.set_facecolor("#0a1628")
    ax_tpd.set_xticks([]); ax_tpd.set_yticks([])
    for sp in ax_tpd.spines.values():
        sp.set_edgecolor(YELLOW); sp.set_linewidth(2)
    ax_tpd.set_title("Produccion", fontsize=10, color=YELLOW, fontweight="bold", pad=5)

    prod_lines = [
        ("Toneladas / hora",                  f"{real['tonnes_per_hour']:.1f} t",  GREEN,  14),
        (f"Toneladas / dia ({HORAS_OPERACION}h)", f"{real['tonnes_per_hour']*HORAS_OPERACION:.0f} t", ORANGE, 16),
        ("Toneladas totales",                 f"{real['total_tonnes']:.0f} t",     WHITE,  11),
        ("Costo total",                       f"{real['total_cost']:.0f}",         PURPLE, 10),
        ("Costo / tonelada",                  f"{real['cost_per_tonne']:.3f}",     PURPLE, 10),
        ("Productividad proxy",               f"{real['prod_proxy']:.1f}",         ACCENT, 10),
    ]
    y = 0.90
    for label, val, color, size in prod_lines:
        ax_tpd.text(0.05, y, label, transform=ax_tpd.transAxes,
                    fontsize=7.5, color=GRAY, va="top")
        ax_tpd.text(0.05, y - 0.09, val, transform=ax_tpd.transAxes,
                    fontsize=size, color=color, va="top", fontweight="bold",
                    fontfamily="monospace")
        y -= 0.17

    # ── Productividad por ciclo ────────────────────────────────────────────
    ax_prod = fig.add_subplot(gs[1, 0])
    axes.append(ax_prod)
    _style_ax(ax_prod, "Productividad por ciclo (t/hr)", "Ciclo", "t/hr")
    ax_prod.plot(res["cycle"], res["productivity_cycle"],
                 color=ORANGE, marker="o", markersize=4, linewidth=1.5)
    ax_prod.fill_between(res["cycle"], res["productivity_cycle"], alpha=0.2, color=ORANGE)
    ax_prod.axhline(res["productivity_cycle"].mean(), color=YELLOW, linestyle="--",
                    linewidth=1.2, label=f"media {res['productivity_cycle'].mean():.0f}")
    ax_prod.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Toneladas por ciclo ────────────────────────────────────────────────
    ax_t = fig.add_subplot(gs[1, 1])
    axes.append(ax_t)
    _style_ax(ax_t, "Toneladas estimadas por ciclo", "Ciclo", "Toneladas")
    colors_t = [GREEN if t >= res["tonnes"].mean() else RED for t in res["tonnes"]]
    ax_t.bar(res["cycle"], res["tonnes"], color=colors_t, width=0.7, alpha=0.9)
    ax_t.axhline(res["tonnes"].mean(), color=YELLOW, linestyle="--",
                 linewidth=1.2, label=f"media {res['tonnes'].mean():.1f} t")
    ax_t.axhline(BUCKET_TONNES, color=GREEN, linestyle=":", linewidth=1.5,
                 label=f"nominal {BUCKET_TONNES} t")
    ax_t.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Costo por ciclo ────────────────────────────────────────────────────
    ax_cost = fig.add_subplot(gs[1, 2])
    axes.append(ax_cost)
    _style_ax(ax_cost, "Costo por ciclo", "Ciclo", "Costo")
    colors_c = [GREEN if c <= res["cost"].mean() else RED for c in res["cost"]]
    ax_cost.bar(res["cycle"], res["cost"], color=colors_c, width=0.7, alpha=0.9)
    ax_cost.axhline(res["cost"].mean(), color=YELLOW, linestyle="--",
                    linewidth=1.2, label=f"media {res['cost'].mean():.0f}")
    ax_cost.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# VISTA 4 — Ciclo Optimo: Mejor Real vs Ideal Realista
# ---------------------------------------------------------------------------

def vista_ciclo_optimo(fig, cr, res, cycles, real, optimos):
    gs = gridspec.GridSpec(2, 3, left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3], hspace=0.50, wspace=0.38)
    axes = []

    best_real = optimos["best_real"]
    ideal     = optimos["ideal"]
    best_cyc_data = cycles[int(best_real["cycle"])]

    # ── Senal del mejor ciclo real ─────────────────────────────────────────
    ax_sig = fig.add_subplot(gs[0, :2])
    axes.append(ax_sig)
    _style_ax(ax_sig,
              f"Ciclo #{int(best_real['cycle'])} — Mejor Real "
              f"(score={best_real['score']:.3f})",
              "Tiempo relativo (s)", "Valor")
    t_rel = best_cyc_data["time"] - best_cyc_data["time"].iloc[0]
    ax_sig.plot(t_rel, best_cyc_data["gz_smooth"], color=ACCENT,
                linewidth=1.5, label="gz suavizado")
    ax_sig.plot(t_rel, best_cyc_data["pitch"], color=ORANGE,
                linewidth=1.5, label="pitch (rad)")
    dyn_max = best_cyc_data["acc_dynamic"].max() or 1
    ax_sig.plot(t_rel, best_cyc_data["acc_dynamic"] / dyn_max,
                color=PURPLE, linewidth=1, alpha=0.7, label="acc_dynamic (norm)")
    ax_sig.axhline(0, color="#555", linewidth=0.5)
    ax_sig.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Tabla comparativa ─────────────────────────────────────────────────
    ax_char = fig.add_subplot(gs[0, 2])
    axes.append(ax_char)
    ax_char.set_facecolor("#0a1628")
    ax_char.set_xticks([]); ax_char.set_yticks([])
    for sp in ax_char.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2)
    ax_char.set_title(f"Mejor Real vs Ideal\n({ideal['note']})",
                      fontsize=8, color=GREEN, fontweight="bold", pad=4)

    rows_char = [
        ("Metrica",         "Real",                                   "Ideal",                  YELLOW),
        ("Duracion (s)",    f"{best_real['duration']:.1f}",           f"{ideal['duration']:.1f}",   ACCENT),
        ("Toneladas",       f"{best_real['tonnes']:.1f}",             f"{ideal['tonnes']:.1f}",     GREEN),
        ("Eficiencia",      f"{best_real['efficiency']:.2f}",         f"{ideal['efficiency']:.2f}", WHITE),
        ("Peso proxy",      f"{best_real['weight_proxy']:.2f}",       f"{ideal['weight_proxy']:.2f}", WHITE),
        ("Score",           f"{best_real['score']:.3f}",              "—",                          ORANGE),
        ("Ciclos/hr",       f"{3600/best_real['duration']:.0f}",      f"{ideal['cph']:.0f}",        WHITE),
        ("t/hr (ciclo)",    f"{best_real['productivity_cycle']:.0f}", f"{ideal['tph']:.0f}",        PURPLE),
    ]
    y = 0.95
    for label, val_r, val_i, color in rows_char:
        ax_char.text(0.03, y, label,  transform=ax_char.transAxes,
                     fontsize=7.5, color=GRAY, va="top")
        ax_char.text(0.52, y, val_r,  transform=ax_char.transAxes,
                     fontsize=7.5, color=color, va="top", fontweight="bold",
                     fontfamily="monospace")
        ax_char.text(0.78, y, val_i,  transform=ax_char.transAxes,
                     fontsize=7.5, color=GREEN, va="top", fontweight="bold",
                     fontfamily="monospace")
        y -= 0.11

    # ── Frontera de eficiencia ─────────────────────────────────────────────
    ax_front = fig.add_subplot(gs[1, :2])
    axes.append(ax_front)
    _style_ax(ax_front, "Frontera de Eficiencia: Duracion vs Toneladas",
              "Duracion (s)", "Toneladas estimadas")
    sc = ax_front.scatter(res["duration"], res["tonnes"],
                          c=res["score"], cmap="RdYlGn", s=60, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax_front, label="Score").ax.yaxis.label.set_color("white")
    ax_front.scatter(best_real["duration"], best_real["tonnes"],
                     color=GREEN, s=150, marker="*", zorder=5,
                     label=f"Mejor real (#{int(best_real['cycle'])})")
    ax_front.scatter(ideal["duration"], ideal["tonnes"],
                     color=YELLOW, s=200, marker="D", zorder=5,
                     label=f"Ideal realista ({ideal['note']})")
    ax_front.axvline(ideal["duration"], color=YELLOW, linestyle=":", linewidth=1, alpha=0.5)
    ax_front.axhline(ideal["tonnes"],   color=YELLOW, linestyle=":", linewidth=1, alpha=0.5)
    ax_front.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Panel mejora eficiencia ────────────────────────────────────────────
    ax_eff = fig.add_subplot(gs[1, 2])
    axes.append(ax_eff)
    _text_panel(ax_eff, "Mejora de Eficiencia", YELLOW, ORANGE)

    eff_real_avg = res["efficiency"].mean()
    eff_ideal_v  = (ideal["weight_proxy"] * ideal["efficiency"]) / ideal["duration"] \
                   if ideal["duration"] > 0 else 0
    improvement  = (eff_ideal_v / eff_real_avg - 1) * 100 if eff_real_avg > 0 else 0

    eff_lines = [
        ("Eficiencia real avg",  f"{eff_real_avg:.3f}",   WHITE,  9),
        ("Eficiencia ideal",     f"{eff_ideal_v:.3f}",    GREEN,  9),
        ("Mejora",               f"+{improvement:.0f}%",  GREEN,  20),
        ("",                     "",                      WHITE,  4),
        ("Ciclos/hr real",       f"{real['cycles_per_hour']:.1f}",  WHITE, 9),
        ("Ciclos/hr ideal",      f"{ideal['cph']:.1f}",             GREEN, 9),
        ("",                     "",                      WHITE,  4),
        ("t/hr real",            f"{real['tonnes_per_hour']:.1f} t", WHITE, 9),
        ("t/hr ideal",           f"{ideal['tph']:.1f} t",            GREEN, 9),
        ("Delta t/hr",           f"+{ideal['tph']-real['tonnes_per_hour']:.1f} t", ORANGE, 11),
        ("",                     "",                      WHITE,  4),
        ("t/dia real",           f"{real['tonnes_per_hour']*HORAS_OPERACION:.0f} t", WHITE, 9),
        ("t/dia ideal",          f"{ideal['tph']*HORAS_OPERACION:.0f} t",            GREEN, 9),
    ]
    y = 0.96
    for label, val, color, size in eff_lines:
        if label:
            ax_eff.text(0.05, y, label, transform=ax_eff.transAxes,
                        fontsize=7, color=GRAY, va="top")
        if val:
            ax_eff.text(0.60, y, val, transform=ax_eff.transAxes,
                        fontsize=size, color=color, va="top", fontweight="bold",
                        fontfamily="monospace")
        y -= 0.065 if size >= 9 else 0.03

    return axes


# ---------------------------------------------------------------------------
# VISTA 5 — Gap de Mejora y Escenario Optimizado
# ---------------------------------------------------------------------------

def vista_gap_mejora(fig, cr, res, real, opt, optimos):
    gs = gridspec.GridSpec(2, 3, left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3], hspace=0.50, wspace=0.38)
    axes = []

    ideal = optimos["ideal"]

    # ── Barras comparativas: 3 escenarios ─────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    axes.append(ax_bar)
    _style_ax(ax_bar, "Comparativa: Real vs Opt. Logistico vs Ideal Realista", "", "")

    metricas = [
        ("Ciclos/hr",       real["cycles_per_hour"],  opt["cycles_per_hour"],  ideal["cph"],         ""),
        ("Utilizacion %",   real["utilization_pct"],  opt["utilization_pct"],  TARGET_UTILIZATION_PCT,"%"),
        ("t/hr",            real["tonnes_per_hour"],  opt["opt_tph"],          ideal["tph"],         "t"),
        ("Ciclo prom (s)",  real["avg_cycle_sec"],     opt["avg_cycle_sec"],    ideal["duration"],    "s"),
        ("Gap mediano (s)", real["median_gap_sec"],    TARGET_MAX_GAP_SEC,      TARGET_MAX_GAP_SEC,   "s"),
    ]
    labels    = [m[0] for m in metricas]
    vals_real = [m[1] for m in metricas]
    vals_opt  = [m[2] for m in metricas]
    vals_ideal= [m[3] for m in metricas]
    units     = [m[4] for m in metricas]

    x = np.arange(len(labels))
    w = 0.25
    bars_r = ax_bar.bar(x - w,  vals_real,  w, label="Real",        color=RED,    alpha=0.85)
    bars_o = ax_bar.bar(x,      vals_opt,   w, label="Opt. Logist.", color=ORANGE, alpha=0.85)
    bars_i = ax_bar.bar(x + w,  vals_ideal, w, label="Ideal",        color=GREEN,  alpha=0.85)

    for bars, vals in [(bars_r, vals_real), (bars_o, vals_opt), (bars_i, vals_ideal)]:
        for bar, val, unit in zip(bars, vals, units):
            ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        f"{val:.0f}{unit}", ha="center", va="bottom", fontsize=7, color="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, color="white")
    ax_bar.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")
    ax_bar.text(0.5, 1.04,
                f"Opt. Logistico: +{opt['delta_tph']:.0f} t/hr (+{opt['delta_pct']:.0f}%)  |  "
                f"Ideal: +{ideal['tph']-real['tonnes_per_hour']:.0f} t/hr",
                transform=ax_bar.transAxes, ha="center", fontsize=9,
                color=YELLOW, fontweight="bold")

    # ── Produccion acumulada ───────────────────────────────────────────────
    ax_line = fig.add_subplot(gs[1, 0])
    axes.append(ax_line)
    _style_ax(ax_line, "Produccion acumulada (t)", "Ciclo", "Toneladas")
    n = real["total_cycles"]
    x_cyc = np.arange(1, n + 1)
    t_r = real["tonnes_per_hour"] / real["cycles_per_hour"] if real["cycles_per_hour"] > 0 else 0
    t_o = opt["opt_tph"]  / opt["cycles_per_hour"]          if opt["cycles_per_hour"]  > 0 else 0
    t_i = ideal["tph"]    / ideal["cph"]                    if ideal["cph"]            > 0 else 0
    ax_line.plot(x_cyc, x_cyc * t_r, color=RED,    linewidth=2,
                 label=f"Real ({real['tonnes_per_hour']:.0f} t/hr)")
    ax_line.plot(x_cyc, x_cyc * t_o, color=ORANGE, linewidth=2, linestyle="--",
                 label=f"Opt. ({opt['opt_tph']:.0f} t/hr)")
    ax_line.plot(x_cyc, x_cyc * t_i, color=GREEN,  linewidth=2, linestyle=":",
                 label=f"Ideal ({ideal['tph']:.0f} t/hr)")
    ax_line.fill_between(x_cyc, x_cyc * t_r, x_cyc * t_i, alpha=0.1, color=GREEN)
    ax_line.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Gaps: real vs objetivo ─────────────────────────────────────────────
    ax_gaps = fig.add_subplot(gs[1, 1])
    axes.append(ax_gaps)
    _style_ax(ax_gaps, "Gaps: Real vs Objetivo", "Intervalo", "Segundos")
    gaps_arr = real["gaps_arr"]
    if len(gaps_arr) > 0:
        x_g = np.arange(len(gaps_arr))
        ax_gaps.bar(x_g - 0.2, gaps_arr, 0.4, color=RED, alpha=0.8, label="Real")
        opt_gaps = np.clip(gaps_arr, None, real["median_gap_sec"])
        ax_gaps.bar(x_g + 0.2, opt_gaps, 0.4, color=GREEN, alpha=0.8, label="Optimizado")
        ax_gaps.axhline(TARGET_MAX_GAP_SEC, color=YELLOW, linestyle="--",
                        linewidth=1.2, label=f"objetivo {TARGET_MAX_GAP_SEC}s")
        ax_gaps.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Panel resumen del gap ──────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 2])
    axes.append(ax_sum)
    _text_panel(ax_sum, "Resumen del Gap de Mejora", YELLOW, GREEN)

    delta_tph_ideal = ideal["tph"] - real["tonnes_per_hour"]
    sum_lines = [
        ("--- LOGISTICO ---",                  YELLOW, 9,  "bold"),
        (f"Reduccion gaps: {opt['gap_reduction_sec']:.1f}s/ciclo", WHITE, 8, "normal"),
        (f"Ciclos extra/hr: +{opt['extra_cycles_per_hour']:.1f}",  WHITE, 8, "normal"),
        (f"Delta t/hr: +{opt['delta_tph']:.0f} t (+{opt['delta_pct']:.0f}%)", GREEN, 10, "bold"),
        (f"Delta t/dia: +{opt['delta_tpd']:.0f} t",                ORANGE, 10, "bold"),
        ("",                                   WHITE, 4,  "normal"),
        ("--- IDEAL REALISTA ---",             YELLOW, 9,  "bold"),
        (f"Duracion obj: {ideal['duration']:.1f}s (p{IDEAL_DURATION_PERCENTILE})", WHITE, 8, "normal"),
        (f"Peso obj: p{IDEAL_WEIGHT_PERCENTILE} de ciclos",         WHITE, 8, "normal"),
        (f"Delta t/hr: +{delta_tph_ideal:.0f} t",                  GREEN, 10, "bold"),
        (f"Delta t/dia: +{delta_tph_ideal*HORAS_OPERACION:.0f} t", ORANGE, 10, "bold"),
        ("",                                   WHITE, 4,  "normal"),
        ("--- MAXIMO POSIBLE ---",             YELLOW, 9,  "bold"),
        (f"t/hr ideal: {ideal['tph']:.0f} t",  GREEN, 11, "bold"),
        (f"t/dia ideal: {ideal['tph']*HORAS_OPERACION:.0f} t", GREEN, 11, "bold"),
    ]
    y = 0.97
    for text, color, size, weight in sum_lines:
        ax_sum.text(0.05, y, text, transform=ax_sum.transAxes,
                    fontsize=size, color=color, fontweight=weight, va="top")
        y -= 0.062 if size >= 9 else 0.035

    return axes


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    VIEWS = [
        ("Metricas Ciclo",   "metricas"),
        ("Relacion Ciclos",  "relacion"),
        ("Productividad",    "productividad"),
        ("Ciclo Optimo",     "optimo"),
        ("Gap de Mejora",    "gap"),
    ]

    def __init__(self, df_proc, cycles, res, real, opt, valid_segments, insights, optimos):
        self.df_proc        = df_proc
        self.cycles         = cycles
        self.res            = res
        self.real           = real
        self.opt            = opt
        self.valid_segments = valid_segments
        self.insights       = insights
        self.optimos        = optimos
        self.content_axes   = []
        self._build_figure()
        self._show_view("metricas")

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(DARK_BG)
        self.fig.text(0.5, 0.975,
                      "EX-5600 — Dashboard de Productividad",
                      ha="center", fontsize=14, color=WHITE, fontweight="bold")

        n   = len(self.VIEWS)
        bw  = 0.155
        bh  = 0.042
        by  = 0.920
        gap = (1.0 - n * bw) / (n + 1)

        self.buttons  = {}
        self.btn_axes = {}
        for i, (label, key) in enumerate(self.VIEWS):
            bx = gap + i * (bw + gap)
            ax_btn = self.fig.add_axes([bx, by, bw, bh])
            btn = Button(ax_btn, label, color=BTN_BG, hovercolor="#1e3a5f")
            btn.label.set_color(WHITE)
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold")
            btn.on_clicked(lambda event, k=key: self._show_view(k))
            self.buttons[key]  = btn
            self.btn_axes[key] = ax_btn

        sep = self.fig.add_axes([0.02, 0.912, 0.96, 0.001])
        sep.set_facecolor("#555")
        sep.set_xticks([]); sep.set_yticks([])

        self.content_rect = [0.03, 0.03, 0.97, 0.905]

    def _show_view(self, view_key):
        for ax in self.content_axes:
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        self.content_axes = []

        for key, ax_btn in self.btn_axes.items():
            ax_btn.set_facecolor(BTN_ACT if key == view_key else BTN_BG)

        cr = self.content_rect
        if view_key == "metricas":
            self.content_axes = vista_metricas_ciclo(self.fig, cr, self.res)
        elif view_key == "relacion":
            self.content_axes = vista_relacion_ciclos(self.fig, cr, self.res, self.real)
        elif view_key == "productividad":
            self.content_axes = vista_productividad_global(self.fig, cr, self.res, self.real)
        elif view_key == "optimo":
            self.content_axes = vista_ciclo_optimo(
                self.fig, cr, self.res, self.cycles, self.real, self.optimos)
        elif view_key == "gap":
            self.content_axes = vista_gap_mejora(
                self.fig, cr, self.res, self.real, self.opt, self.optimos)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
# Guardar JSON
# ---------------------------------------------------------------------------

def guardar_reporte(res, real, opt, optimos, insights):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "productivity_report.json"

    def _clean(v):
        if isinstance(v, (np.floating, float)):
            return round(float(v), 4)
        if isinstance(v, (np.integer, int)):
            return int(v)
        return v

    cycles_data = [
        {k: _clean(v) for k, v in row.items() if k not in ("t_start", "t_end")}
        for _, row in res.iterrows()
    ]

    report = {
        "real":            {k: _clean(v) for k, v in real.items() if k != "gaps_arr"},
        "gaps_arr":        real["gaps_arr"].tolist(),
        "optimizado":      {k: _clean(v) for k, v in opt.items()},
        "ideal":           {k: _clean(v) for k, v in optimos["ideal"].items()
                            if k != "note"},
        "best_real_cycle": {k: _clean(v) for k, v in optimos["best_real"].items()
                            if k not in ("t_start", "t_end")},
        "top10_cycles":    sorted(cycles_data, key=lambda x: x.get("score", 0),
                                  reverse=True)[:10],
        "all_cycles":      cycles_data,
        "recommendations": insights,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Reporte JSON guardado en: {out}")
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EX-5600 — Dashboard interactivo de productividad"
    )
    parser.add_argument("--imu",    type=str,  default=None,
                        help="Ruta explicita al archivo .npy (opcional)")
    parser.add_argument("--no-gui", action="store_true",
                        help="Solo consola, sin ventana grafica")
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

    # Salida completa en consola
    imprimir_consola(df_proc, cycles, res, real, opt, optimos, insights)

    guardar_reporte(res, real, opt, optimos, insights)

    if not args.no_gui:
        dash = Dashboard(df_proc, cycles, res, real, opt, valid_segments, insights, optimos)
        dash.show()


if __name__ == "__main__":
    main()
    