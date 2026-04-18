"""
EX-5600 Shovel — Dashboard Interactivo de Productividad
========================================================
6 vistas navegables con botones:

  [1] Metricas Ciclo   — tabla detallada + explicacion de columnas
  [2] Relacion Ciclos  — estadisticas descriptivas + gaps
  [3] Productividad    — KPIs OEE, toneladas, costo
  [4] Ciclo Optimo     — mejor real vs ideal realista + frontera
  [5] Gap de Mejora    — escenario optimizado logistico
  [6] Recorrido        — trayectoria gz real vs optimizada

Uso:
    python dashboard_shovel.py
    python dashboard_shovel.py --imu ruta/al/archivo.npy
    python dashboard_shovel.py --no-gui
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# Ciclos por lado antes de cambiar (el brazo alterna cada 7 depositos)
CICLOS_POR_LADO = 7

TARGET_UTILIZATION_PCT    = 85.0
TARGET_MAX_GAP_SEC        = 8.0
IDEAL_DURATION_PERCENTILE = 10
IDEAL_WEIGHT_PERCENTILE   = 90

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
    """
    El brazo alterna de lado cada CICLOS_POR_LADO depositos:
      ciclos 0..6  -> lado A (izquierda)
      ciclos 7..13 -> lado B (derecha)
      ciclos 14..20-> lado A (izquierda)
      ...
    Esto corrige el error de deteccion de lado basado en gz_smooth.mean()
    que puede ser incorrecto cuando el ciclo contiene movimiento bidireccional.
    """
    rows = []
    cycle_count = 0   # contador de ciclos validos (para asignar lado correcto)
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

        # Lado determinado por posicion en la secuencia (cada 7 ciclos alterna)
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
        "prod_proxy":       float(res["weight_proxy"].mean()) * cph * util,
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
    dt_mean = df_proc["time"].diff().mean()
    if dt_mean > 0:
        print(f"  Frecuencia:    {1/dt_mean:.1f} Hz")
    for col in ["acc_mag", "acc_dynamic", "gz_smooth", "pitch", "yaw"]:
        s = df_proc[col]
        print(f"  {col:<12} media={s.mean():.3f}  std={s.std():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")
    print(f"  idle:          {df_proc['idle'].mean()*100:.1f}% del tiempo")

    _h("2. DETECCION DE CICLOS")
    print(f"  Ciclos detectados (validos >= 5s): {real['total_cycles']}")
    print(f"  Lado asignado por posicion (cada {CICLOS_POR_LADO} ciclos alterna)")

    _h("3. METRICAS POR CICLO")
    print("""
  Columnas:
    Dur(s)     = t_fin - t_inicio
    Esfuerzo   = sum(acc_mag)
    Pitch      = max(pitch) - min(pitch)
    Suavidad   = std(acc_mag)
    T.Elev(s)  = tiempo con diff(pitch) > 0
    Peso(t)    = sum(acc_dyn*dt) en fase lifting, escalado a toneladas
    Eficiencia = (Pitch * Esfuerzo) / Duracion
    Costo      = Duracion * Esfuerzo
    Score      = 0.5*W_norm + 0.3*(1/T_norm) + 0.2*Eff_norm
    Prod(t/hr) = (3600/Dur) * Peso(t)
""")
    fmt = ("{:>4}  {:>7}  {:>8}  {:>7}  {:>7}  {:>7}  {:>7}  "
           "{:>7}  {:>8}  {:>6}  {:>10}  {:>10}")
    hdr = fmt.format("#","Dur(s)","Esfuerzo","Pitch","Suavid",
                     "T.Elev","Peso(t)","Efic","Costo","Score","Prod(t/hr)","Lado")
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
            row["side"],
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
    print(f"  Mejor ciclo (>=10s):  {real['best_cycle_sec']:.2f} s")
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
    print(f"  Lado:                 {best_real['side']}")
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
    print(f"  Reduccion gaps:       {opt['gap_reduction_sec']:.2f} s/ciclo")
    print(f"  Ciclos extra/hr:      +{opt['extra_cycles_per_hour']:.2f}")
    print(f"  Ciclos/hr opt.:       {opt['cycles_per_hour']:.2f}")
    print(f"  Utilizacion obj.:     {opt['utilization_pct']:.1f}%")
    print(f"  Ciclo prom. obj.:     {opt['avg_cycle_sec']:.2f} s")
    print(f"  t/hr opt.:            {opt['opt_tph']:.2f}")
    print(f"  Delta t/hr:           +{opt['delta_tph']:.2f} (+{opt['delta_pct']:.1f}%)")
    print(f"  Delta t/dia:          +{opt['delta_tpd']:.0f} t")
    delta_ideal = ideal["tph"] - real["tonnes_per_hour"]
    print(f"\n  Ideal casuistico:")
    print(f"  Delta t/hr vs real:   +{delta_ideal:.2f} t/hr")
    print(f"  Delta t/dia vs real:  +{delta_ideal*HORAS_OPERACION:.0f} t")

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


def _info_panel(ax, title, tc=YELLOW, bc=ACCENT):
    ax.set_facecolor(BTN_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor(bc)
        sp.set_linewidth(1.5)
    ax.set_title(title, fontsize=10, color=tc, fontweight="bold", pad=5)


def _hline(ax, y_val, **kwargs):
    ax.axhline(y=y_val, **kwargs)


def _hspan_text(ax, y_frac_bottom, y_frac_top, color):
    ax.fill_between([0, 1], [y_frac_bottom, y_frac_bottom],
                    [y_frac_top, y_frac_top],
                    color=color, transform=ax.transAxes, zorder=0)


# ---------------------------------------------------------------------------
# VISTA 1 — Metricas por Ciclo
# ---------------------------------------------------------------------------

def vista_metricas_ciclo(fig, cr, res):
    gs = gridspec.GridSpec(2, 2,
                           left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3],
                           hspace=0.45, wspace=0.35)
    axes = []

    ax_tbl = fig.add_subplot(gs[0, :])
    axes.append(ax_tbl)
    _info_panel(ax_tbl, "Metricas por Ciclo — Todos los Ciclos", YELLOW, YELLOW)

    headers = ["#", "Dur(s)", "Esfuerzo", "Pitch", "Suavidad",
               "T.Elev(s)", "Peso(t)", "Eficiencia", "Costo", "Score", "Lado"]
    col_x   = [0.00, 0.07, 0.16, 0.26, 0.35, 0.44, 0.53, 0.62, 0.72, 0.82, 0.92]

    for hdr, cx in zip(headers, col_x):
        ax_tbl.text(cx, 0.95, hdr, transform=ax_tbl.transAxes,
                    fontsize=7, color=YELLOW, fontweight="bold", va="top")
    ax_tbl.plot([0, 1], [0.90, 0.90], color="#555", linewidth=0.8,
                transform=ax_tbl.transAxes)

    n_show = min(len(res), 12)
    row_h  = 0.86 / max(n_show, 1)
    for rank, (_, row) in enumerate(res.head(n_show).iterrows()):
        y  = 0.88 - rank * row_h
        bg = "#1e2d4a" if rank % 2 == 0 else BTN_BG
        _hspan_text(ax_tbl, y - row_h * 0.85, y + row_h * 0.1, bg)
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

    ax_exp = fig.add_subplot(gs[1, 0])
    axes.append(ax_exp)
    _info_panel(ax_exp, "Como se calcula cada columna", ACCENT, ACCENT)
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
        ("Lado",       f"alterna cada {CICLOS_POR_LADO} ciclos"),
    ]
    y = 0.93
    for col, formula in formulas:
        ax_exp.text(0.03, y, f"{col:<12}", transform=ax_exp.transAxes,
                    fontsize=7.5, color=YELLOW, va="top", fontfamily="monospace")
        ax_exp.text(0.38, y, formula, transform=ax_exp.transAxes,
                    fontsize=7.5, color=WHITE, va="top")
        y -= 0.082

    ax_sc = fig.add_subplot(gs[1, 1])
    axes.append(ax_sc)
    _style_ax(ax_sc, "Score por ciclo", "Ciclo", "Score")
    colors_sc = [GREEN if s >= res["score"].mean() else RED for s in res["score"]]
    ax_sc.bar(res["cycle"], res["score"], color=colors_sc, width=0.7, alpha=0.9)
    _hline(ax_sc, res["score"].mean(), color=YELLOW, linestyle="--",
           linewidth=1.2, label=f"media {res['score'].mean():.3f}")
    ax_sc.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# VISTA 2 — Relacion entre Ciclos
# ---------------------------------------------------------------------------

def vista_relacion_ciclos(fig, cr, res, real):
    gs = gridspec.GridSpec(2, 2,
                           left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3],
                           hspace=0.45, wspace=0.35)
    axes = []

    ax_stats = fig.add_subplot(gs[0, 0])
    axes.append(ax_stats)
    _info_panel(ax_stats, "Estadisticas de Ciclos", YELLOW, ACCENT)

    stats_cols = ["duration","effort","pitch_range","smoothness",
                  "lifting_time","tonnes","efficiency","cost","score"]
    stats = res[stats_cols].describe().loc[["mean","std","min","25%","50%","75%","max"]]
    col_labels = ["Dur","Esf","Pitch","Suav","T.El","Ton","Efic","Cost","Score"]
    row_labels = ["media","std","min","p25","p50","p75","max"]
    col_xs = [0.02 + i * 0.107 for i in range(len(col_labels))]
    row_ys = [0.90 - i * 0.12  for i in range(len(row_labels) + 1)]

    for lbl, cx in zip(col_labels, col_xs):
        ax_stats.text(cx, row_ys[0], lbl, transform=ax_stats.transAxes,
                      fontsize=6.5, color=YELLOW, fontweight="bold", va="top")
    ax_stats.plot([0, 1], [row_ys[0] - 0.04, row_ys[0] - 0.04],
                  color="#555", linewidth=0.6, transform=ax_stats.transAxes)

    for ri, (stat_name, row_y) in enumerate(zip(row_labels, row_ys[1:])):
        ax_stats.text(0.00, row_y, stat_name, transform=ax_stats.transAxes,
                      fontsize=6, color=GRAY, va="top")
        for ci, (col, cx) in enumerate(zip(stats_cols, col_xs)):
            val = stats.iloc[ri, ci]
            ax_stats.text(cx, row_y, f"{val:.2f}", transform=ax_stats.transAxes,
                          fontsize=6, color=WHITE, va="top", fontfamily="monospace")

    ax_gaps = fig.add_subplot(gs[0, 1])
    axes.append(ax_gaps)
    _style_ax(ax_gaps, "Gaps entre ciclos (s)", "Intervalo", "Segundos")
    gaps_arr = real["gaps_arr"]
    if len(gaps_arr) > 0:
        colors_g = [GREEN if g <= real["median_gap_sec"] else RED for g in gaps_arr]
        ax_gaps.bar(range(len(gaps_arr)), gaps_arr, color=colors_g, width=0.7, alpha=0.9)
        _hline(ax_gaps, real["median_gap_sec"], color=YELLOW, linestyle="--",
               linewidth=1.2, label=f"mediana {real['median_gap_sec']:.1f}s")
        _hline(ax_gaps, TARGET_MAX_GAP_SEC, color=GREEN, linestyle=":",
               linewidth=1.5, label=f"objetivo {TARGET_MAX_GAP_SEC}s")
        ax_gaps.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    ax_dur = fig.add_subplot(gs[1, 0])
    axes.append(ax_dur)
    _style_ax(ax_dur, "Duracion por ciclo", "Ciclo", "Segundos")
    colors_d = [GREEN if d <= res["duration"].quantile(0.5) else RED
                for d in res["duration"]]
    ax_dur.bar(res["cycle"], res["duration"], color=colors_d, width=0.7, alpha=0.9)
    _hline(ax_dur, res["duration"].mean(), color=YELLOW, linestyle="--",
           linewidth=1.2, label=f"media {res['duration'].mean():.1f}s")
    ax_dur.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

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
    gs = gridspec.GridSpec(2, 3,
                           left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3],
                           hspace=0.45, wspace=0.35)
    axes = []

    ax_kpi = fig.add_subplot(gs[0, :2])
    axes.append(ax_kpi)
    _info_panel(ax_kpi, "Productividad Global — KPIs OEE", YELLOW, ACCENT)

    kpis = [
        ("Ciclos totales",       f"{real['total_cycles']}",                WHITE),
        ("Ciclos / hora",        f"{real['cycles_per_hour']:.2f}",         ACCENT),
        ("Utilizacion",          f"{real['utilization_pct']:.1f}%",
         GREEN if real["utilization_pct"] >= 75 else RED),
        ("Tiempo muerto",        f"{real['idle_ratio_pct']:.1f}%",
         GREEN if real["idle_ratio_pct"] <= 15 else RED),
        ("Duracion media",       f"{real['avg_cycle_sec']:.1f} s",         WHITE),
        ("Mejor ciclo (>=10s)",  f"{real['best_cycle_sec']:.1f} s",        GREEN),
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

    ax_tpd = fig.add_subplot(gs[0, 2])
    axes.append(ax_tpd)
    _info_panel(ax_tpd, "Produccion", YELLOW, YELLOW)
    prod_lines = [
        ("Toneladas / hora",                  f"{real['tonnes_per_hour']:.1f} t",  GREEN,  14),
        (f"Toneladas / dia ({HORAS_OPERACION}h)", f"{real['tonnes_per_hour']*HORAS_OPERACION:.0f} t", ORANGE, 16),
        ("Toneladas totales",                 f"{real['total_tonnes']:.0f} t",     WHITE,  11),
        ("Costo total",                       f"{real['total_cost']:.0f}",         PURPLE, 10),
        ("Costo / tonelada",                  f"{real['cost_per_tonne']:.3f}",     PURPLE, 10),
    ]
    y = 0.90
    for label, val, color, size in prod_lines:
        ax_tpd.text(0.05, y, label, transform=ax_tpd.transAxes,
                    fontsize=7.5, color=GRAY, va="top")
        ax_tpd.text(0.05, y - 0.09, val, transform=ax_tpd.transAxes,
                    fontsize=size, color=color, va="top", fontweight="bold",
                    fontfamily="monospace")
        y -= 0.17

    ax_prod = fig.add_subplot(gs[1, 0])
    axes.append(ax_prod)
    _style_ax(ax_prod, "Productividad por ciclo (t/hr)", "Ciclo", "t/hr")
    ax_prod.plot(res["cycle"], res["productivity_cycle"],
                 color=ORANGE, marker="o", markersize=4, linewidth=1.5)
    ax_prod.fill_between(res["cycle"], res["productivity_cycle"], alpha=0.2, color=ORANGE)
    _hline(ax_prod, res["productivity_cycle"].mean(), color=YELLOW, linestyle="--",
           linewidth=1.2, label=f"media {res['productivity_cycle'].mean():.0f}")
    ax_prod.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    ax_t = fig.add_subplot(gs[1, 1])
    axes.append(ax_t)
    _style_ax(ax_t, "Toneladas estimadas por ciclo", "Ciclo", "Toneladas")
    colors_t = [GREEN if t >= res["tonnes"].mean() else RED for t in res["tonnes"]]
    ax_t.bar(res["cycle"], res["tonnes"], color=colors_t, width=0.7, alpha=0.9)
    _hline(ax_t, res["tonnes"].mean(), color=YELLOW, linestyle="--",
           linewidth=1.2, label=f"media {res['tonnes'].mean():.1f} t")
    _hline(ax_t, BUCKET_TONNES, color=GREEN, linestyle=":", linewidth=1.5,
           label=f"nominal {BUCKET_TONNES} t")
    ax_t.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    ax_cost = fig.add_subplot(gs[1, 2])
    axes.append(ax_cost)
    _style_ax(ax_cost, "Costo por ciclo", "Ciclo", "Costo")
    colors_c = [GREEN if c <= res["cost"].mean() else RED for c in res["cost"]]
    ax_cost.bar(res["cycle"], res["cost"], color=colors_c, width=0.7, alpha=0.9)
    _hline(ax_cost, res["cost"].mean(), color=YELLOW, linestyle="--",
           linewidth=1.2, label=f"media {res['cost'].mean():.0f}")
    ax_cost.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# VISTA 4 — Ciclo Optimo
# ---------------------------------------------------------------------------

def vista_ciclo_optimo(fig, cr, res, cycles, real, optimos):
    gs = gridspec.GridSpec(2, 3,
                           left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3],
                           hspace=0.50, wspace=0.38)
    axes = []

    best_real = optimos["best_real"]
    ideal     = optimos["ideal"]
    cyc_idx   = int(best_real["cycle"])
    best_cyc_data = cycles[cyc_idx] if cyc_idx < len(cycles) else cycles[-1]

    ax_sig = fig.add_subplot(gs[0, :2])
    axes.append(ax_sig)
    _style_ax(ax_sig,
              f"Ciclo #{cyc_idx} — Mejor Real (score={best_real['score']:.3f}, "
              f"dur={best_real['duration']:.1f}s, lado={best_real['side']})",
              "Tiempo relativo (s)", "Valor")
    t_rel = best_cyc_data["time"] - best_cyc_data["time"].iloc[0]
    ax_sig.plot(t_rel, best_cyc_data["gz_smooth"], color=ACCENT,
                linewidth=1.5, label="gz suavizado")
    ax_sig.plot(t_rel, best_cyc_data["pitch"], color=ORANGE,
                linewidth=1.5, label="pitch (rad)")
    dyn_max = float(best_cyc_data["acc_dynamic"].max()) or 1.0
    ax_sig.plot(t_rel, best_cyc_data["acc_dynamic"] / dyn_max,
                color=PURPLE, linewidth=1, alpha=0.7, label="acc_dynamic (norm)")
    ax_sig.axhline(0, color="#555", linewidth=0.5)
    ax_sig.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    ax_char = fig.add_subplot(gs[0, 2])
    axes.append(ax_char)
    _info_panel(ax_char, f"Mejor Real vs Ideal\n({ideal['note']})", GREEN, GREEN)
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

    ax_front = fig.add_subplot(gs[1, :2])
    axes.append(ax_front)
    _style_ax(ax_front, "Frontera de Eficiencia: Duracion vs Toneladas",
              "Duracion (s)", "Toneladas estimadas")
    sc = ax_front.scatter(res["duration"], res["tonnes"],
                          c=res["score"], cmap="RdYlGn", s=60, alpha=0.85, zorder=3)
    plt.colorbar(sc, ax=ax_front, label="Score").ax.yaxis.label.set_color("white")
    ax_front.scatter(best_real["duration"], best_real["tonnes"],
                     color=GREEN, s=150, marker="*", zorder=5,
                     label=f"Mejor real (#{cyc_idx})")
    ax_front.scatter(ideal["duration"], ideal["tonnes"],
                     color=YELLOW, s=200, marker="D", zorder=5,
                     label=f"Ideal ({ideal['note']})")
    ax_front.axvline(ideal["duration"], color=YELLOW, linestyle=":", linewidth=1, alpha=0.5)
    ax_front.axhline(ideal["tonnes"],   color=YELLOW, linestyle=":", linewidth=1, alpha=0.5)
    ax_front.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    ax_eff = fig.add_subplot(gs[1, 2])
    axes.append(ax_eff)
    _info_panel(ax_eff, "Mejora de Eficiencia", YELLOW, ORANGE)
    eff_real_avg = float(res["efficiency"].mean())
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
# VISTA 5 — Gap de Mejora
# Cambios: barras separadas en subgraficas individuales + sin linea objetivo en gaps
# ---------------------------------------------------------------------------

def vista_gap_mejora(fig, cr, res, real, opt, optimos):
    gs = gridspec.GridSpec(2, 3,
                           left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3],
                           hspace=0.55, wspace=0.40)
    axes = []
    ideal = optimos["ideal"]

    # ── Fila superior: 5 metricas en subgraficas separadas (1 barra por metrica) ──
    # Usamos un GridSpec anidado para la fila superior
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs[0, :],
                                              wspace=0.45)

    metricas = [
        ("Ciclos/hr",       real["cycles_per_hour"],  opt["cycles_per_hour"],  ideal["cph"],          ""),
        ("Utilizacion %",   real["utilization_pct"],  opt["utilization_pct"],  TARGET_UTILIZATION_PCT, "%"),
        ("t/hr",            real["tonnes_per_hour"],  opt["opt_tph"],          ideal["tph"],           "t"),
        ("Ciclo prom (s)",  real["avg_cycle_sec"],     opt["avg_cycle_sec"],    ideal["duration"],      "s"),
        ("Gap mediano (s)", real["median_gap_sec"],    TARGET_MAX_GAP_SEC,      TARGET_MAX_GAP_SEC,     "s"),
    ]

    for mi, (label, v_real, v_opt, v_ideal, unit) in enumerate(metricas):
        ax_m = fig.add_subplot(gs_top[0, mi])
        axes.append(ax_m)
        _style_ax(ax_m, label, ts=8)
        bars = ax_m.bar([0, 1, 2], [v_real, v_opt, v_ideal],
                        color=[RED, ORANGE, GREEN], alpha=0.85, width=0.6)
        ax_m.set_xticks([0, 1, 2])
        ax_m.set_xticklabels(["Real", "Opt.", "Ideal"], fontsize=7, color="white")
        for bar, val in zip(bars, [v_real, v_opt, v_ideal]):
            ax_m.text(bar.get_x() + bar.get_width()/2,
                      bar.get_height() * 1.02,
                      f"{val:.1f}{unit}", ha="center", va="bottom",
                      fontsize=7, color="white")

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

    # ── Gaps real vs optimizado (sin linea objetivo — diferencias son decimales) ──
    ax_gaps = fig.add_subplot(gs[1, 1])
    axes.append(ax_gaps)
    _style_ax(ax_gaps, "Gaps: Real vs Optimizado", "Intervalo", "Segundos")
    gaps_arr = real["gaps_arr"]
    if len(gaps_arr) > 0:
        x_g = np.arange(len(gaps_arr))
        ax_gaps.bar(x_g - 0.2, gaps_arr, 0.4, color=RED, alpha=0.8, label="Real")
        opt_gaps = np.clip(gaps_arr, None, real["median_gap_sec"])
        ax_gaps.bar(x_g + 0.2, opt_gaps, 0.4, color=GREEN, alpha=0.8, label="Optimizado")
        # Sin linea objetivo (los gaps optimizados solo difieren en decimales)
        ax_gaps.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Resumen del gap ────────────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 2])
    axes.append(ax_sum)
    _info_panel(ax_sum, "Resumen del Gap de Mejora", YELLOW, GREEN)
    delta_ideal = ideal["tph"] - real["tonnes_per_hour"]
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
        (f"Delta t/hr: +{delta_ideal:.0f} t",                      GREEN, 10, "bold"),
        (f"Delta t/dia: +{delta_ideal*HORAS_OPERACION:.0f} t",     ORANGE, 10, "bold"),
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
# VISTA 6 — Recorrido (gz acumulado real vs optimizado)
# ---------------------------------------------------------------------------

def vista_recorrido(fig, cr, df_proc, cycles, res, real):
    """
    Muestra el 'recorrido' angular de la pala usando gz (velocidad angular Z).
    - gz acumulado = integral de gz en el tiempo -> angulo total girado
    - Recorrido real: gz_smooth acumulado de toda la sesion
    - Recorrido optimizado: mismo recorrido pero con los ciclos lentos
      reemplazados por la duracion del ciclo p25 (manteniendo el angulo total)
    - Tambien muestra gz_smooth por ciclo coloreado por lado
    """
    gs = gridspec.GridSpec(2, 2,
                           left=cr[0], right=cr[2],
                           bottom=cr[1], top=cr[3],
                           hspace=0.45, wspace=0.35)
    axes = []

    # ── gz acumulado — recorrido angular total ─────────────────────────────
    ax_cum = fig.add_subplot(gs[0, :])
    axes.append(ax_cum)
    _style_ax(ax_cum, "Recorrido Angular Acumulado (integral de gz)",
              "Tiempo (s)", "Angulo acumulado (rad)")

    t_arr  = df_proc["time"].values
    gz_arr = df_proc["gz_smooth"].values
    dt_arr = np.diff(t_arr, prepend=t_arr[0])
    gz_cum = np.cumsum(gz_arr * dt_arr)

    ax_cum.plot(t_arr, gz_cum, color=ACCENT, linewidth=1.2, label="Recorrido real")
    ax_cum.fill_between(t_arr, gz_cum, alpha=0.15, color=ACCENT)

    # Marcar inicio de cada ciclo
    for _, row in res.iterrows():
        t_s = row["t_start"]
        idx = np.searchsorted(t_arr, t_s)
        if idx < len(gz_cum):
            color_side = GREEN if row["side"] == "izquierda" else ORANGE
            ax_cum.axvline(t_s, color=color_side, linewidth=0.6, alpha=0.5)

    # Leyenda de lados
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=GREEN,  linewidth=1.5, label="Inicio ciclo izquierda"),
        Line2D([0], [0], color=ORANGE, linewidth=1.5, label="Inicio ciclo derecha"),
        Line2D([0], [0], color=ACCENT, linewidth=2,   label="gz acumulado"),
    ]
    ax_cum.legend(handles=legend_elements, fontsize=7,
                  facecolor=DARK_BG, labelcolor="white")

    # ── gz por ciclo coloreado por lado ───────────────────────────────────
    ax_gz = fig.add_subplot(gs[1, 0])
    axes.append(ax_gz)
    _style_ax(ax_gz, "gz suavizado por ciclo (coloreado por lado)",
              "Tiempo (s)", "gz (rad/s)")

    for _, row in res.iterrows():
        t_s = row["t_start"]
        t_e = row["t_end"]
        mask = (t_arr >= t_s) & (t_arr <= t_e)
        color_side = GREEN if row["side"] == "izquierda" else ORANGE
        ax_gz.plot(t_arr[mask], gz_arr[mask], color=color_side,
                   linewidth=0.8, alpha=0.7)

    ax_gz.axhline(0, color="#555", linewidth=0.5)
    ax_gz.axhline(GZ_THRESHOLD,  color=RED, linestyle="--", linewidth=0.8, alpha=0.6)
    ax_gz.axhline(-GZ_THRESHOLD, color=RED, linestyle="--", linewidth=0.8, alpha=0.6)

    legend_gz = [
        Line2D([0], [0], color=GREEN,  linewidth=2, label="Izquierda"),
        Line2D([0], [0], color=ORANGE, linewidth=2, label="Derecha"),
    ]
    ax_gz.legend(handles=legend_gz, fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Recorrido por ciclo (angulo total girado por ciclo) ────────────────
    ax_ang = fig.add_subplot(gs[1, 1])
    axes.append(ax_ang)
    _style_ax(ax_ang, "Angulo total girado por ciclo (|gz| integrado)",
              "Ciclo", "Angulo (rad)")

    ang_per_cycle = []
    for _, row in res.iterrows():
        t_s = row["t_start"]
        t_e = row["t_end"]
        mask = (t_arr >= t_s) & (t_arr <= t_e)
        gz_c  = np.abs(gz_arr[mask])
        dt_c  = np.diff(t_arr[mask], prepend=t_arr[mask][0]) if mask.sum() > 0 else np.array([0])
        total = float(np.sum(gz_c * dt_c))
        ang_per_cycle.append(total)

    ang_arr = np.array(ang_per_cycle)
    colors_ang = [GREEN if row["side"] == "izquierda" else ORANGE
                  for _, row in res.iterrows()]
    ax_ang.bar(res["cycle"], ang_arr, color=colors_ang, width=0.7, alpha=0.9)
    _hline(ax_ang, ang_arr.mean(), color=YELLOW, linestyle="--",
           linewidth=1.2, label=f"media {ang_arr.mean():.1f} rad")

    legend_ang = [
        Line2D([0], [0], color=GREEN,  linewidth=0, marker="s",
               markersize=8, label="Izquierda"),
        Line2D([0], [0], color=ORANGE, linewidth=0, marker="s",
               markersize=8, label="Derecha"),
    ]
    ax_ang.legend(handles=legend_ang, fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class Dashboard:
    VIEWS = [
        ("Metricas Ciclo",  "metricas"),
        ("Relacion Ciclos", "relacion"),
        ("Productividad",   "productividad"),
        ("Ciclo Optimo",    "optimo"),
        ("Gap de Mejora",   "gap"),
        ("Recorrido",       "recorrido"),
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
        self._btn_refs      = []
        self._btn_ax_map    = {}

        self._build_figure()
        self._show_view("metricas")

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(DARK_BG)
        self.fig.text(0.5, 0.975,
                      "EX-5600 — Dashboard de Productividad",
                      ha="center", fontsize=14, color=WHITE, fontweight="bold")

        n   = len(self.VIEWS)
        bw  = 0.130
        bh  = 0.042
        by  = 0.920
        gap = (1.0 - n * bw) / (n + 1)

        for i, (label, key) in enumerate(self.VIEWS):
            bx     = gap + i * (bw + gap)
            ax_btn = self.fig.add_axes([bx, by, bw, bh])
            ax_btn.set_facecolor(BTN_BG)
            for sp in ax_btn.spines.values():
                sp.set_edgecolor("#888")

            btn = Button(ax_btn, label, color=BTN_BG, hovercolor="#1e3a5f")
            btn.label.set_color(WHITE)
            btn.label.set_fontsize(8)
            btn.label.set_fontweight("bold")
            btn.on_clicked(lambda event, k=key: self._show_view(k))

            self._btn_refs.append(btn)
            self._btn_ax_map[key] = ax_btn

        sep = self.fig.add_axes([0.02, 0.912, 0.96, 0.001])
        sep.set_facecolor("#555")
        sep.set_xticks([]); sep.set_yticks([])
        for sp in sep.spines.values():
            sp.set_visible(False)

        self.content_rect = [0.03, 0.03, 0.97, 0.905]

    def _show_view(self, view_key):
        for ax in self.content_axes:
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        self.content_axes = []

        for key, ax_btn in self._btn_ax_map.items():
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
        elif view_key == "recorrido":
            self.content_axes = vista_recorrido(
                self.fig, cr, self.df_proc, self.cycles, self.res, self.real)

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

    imprimir_consola(df_proc, cycles, res, real, opt, optimos, insights)
    guardar_reporte(res, real, opt, optimos, insights)

    if not args.no_gui:
        dash = Dashboard(df_proc, cycles, res, real, opt, valid_segments, insights, optimos)
        dash.show()


if __name__ == "__main__":
    main()