"""
EX-5600 Shovel — Reporte de Productividad con Comparativa de Optimización
=========================================================================
Análisis batch completo del IMU: métricas reales vs escenario optimizado.

Uso:
    python productivity_report.py
    python productivity_report.py --imu ruta/al/archivo.npy
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

INPUTS_DIR = Path("./inputs")
OUTPUTS_DIR = Path("./outputs")
IMU_GLOB = "*imu*.npy"
FALLBACK_IMU = Path("C:/Hackathon/Archive/40343737_20260313_110600_to_112100_imu.npy")

GZ_THRESHOLD = 0.4
MIN_SEGMENT_TIME = 2.5
GZ_SMOOTH_WINDOW = 50
IDLE_THRESHOLD = 0.3
BUCKET_TONNES = 52.0          # toneladas nominales EX-5600

# Metas de referencia industria
TARGET_UTILIZATION_PCT = 85.0
TARGET_CYCLES_PER_HOUR = 100.0
TARGET_CONSISTENCY_CV = 0.10
TARGET_MAX_GAP_SEC = 8.0


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def find_imu_file(explicit: str = None) -> Path:
    if explicit:
        return Path(explicit)
    matches = glob.glob(str(INPUTS_DIR / IMU_GLOB))
    if matches:
        return Path(matches[0])
    if FALLBACK_IMU.exists():
        return FALLBACK_IMU
    raise FileNotFoundError(f"No se encontró archivo IMU en {INPUTS_DIR}")


def load_imu(path: Path) -> pd.DataFrame:
    data = np.load(str(path))
    df = pd.DataFrame(data, columns=[
        "time", "ax", "ay", "az", "gx", "gy", "gz",
        "qw", "qx", "qy", "qz",
    ])
    df["time"] = (df["time"] - df["time"].iloc[0]) / 1e9
    return df


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["acc_mag"] = np.sqrt(df["ax"] ** 2 + df["ay"] ** 2 + df["az"] ** 2)
    df["gz_smooth"] = df["gz"].rolling(window=GZ_SMOOTH_WINDOW, min_periods=1).mean()
    df["pitch"] = np.arcsin(
        np.clip(2 * (df["qw"] * df["qy"] - df["qz"] * df["qx"]), -1.0, 1.0)
    )
    return df


# ---------------------------------------------------------------------------
# Detección de ciclos
# ---------------------------------------------------------------------------

def detectar_ciclos(df: pd.DataFrame):
    df["moving_left"] = df["gz_smooth"] > GZ_THRESHOLD
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
# Métricas por ciclo
# ---------------------------------------------------------------------------

def calcular_metricas(cycles: list) -> pd.DataFrame:
    rows = []
    for i, cyc in enumerate(cycles):
        dur = cyc["time"].iloc[-1] - cyc["time"].iloc[0]
        effort = cyc["acc_mag"].sum()
        pitch_range = cyc["pitch"].max() - cyc["pitch"].min()
        smoothness = cyc["acc_mag"].std()
        dt = cyc["time"].diff().mean()
        lifting_time = (cyc["pitch"].diff() > 0).sum() * (dt if not np.isnan(dt) else 0)
        side = "izquierda" if cyc["gz_smooth"].mean() > 0 else "derecha"
        eff = (pitch_range * effort) / dur if dur > 0 else 0.0
        rows.append({
            "cycle": i, "duration": dur, "effort": effort,
            "pitch_range": pitch_range, "smoothness": smoothness,
            "lifting_time": lifting_time, "side": side, "efficiency": eff,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Modelo de productividad REAL
# ---------------------------------------------------------------------------

def calcular_productividad(df: pd.DataFrame, cycles: list, res: pd.DataFrame) -> dict:
    total_t = df["time"].iloc[-1] - df["time"].iloc[0]
    total_h = total_t / 3600 if total_t > 0 else 1e-9
    cph = len(cycles) / total_h

    dt = df["time"].diff().mean()
    idle_t = (df["acc_mag"] < IDLE_THRESHOLD).sum() * (dt if not np.isnan(dt) else 0)
    idle_r = idle_t / total_t if total_t > 0 else 0
    util = max(0.0, 1.0 - idle_r)

    avg_effort = res["effort"].mean() if len(res) > 0 else 0.0
    productivity = cph * avg_effort * util

    gaps = [cycles[i + 1]["time"].iloc[0] - cycles[i]["time"].iloc[-1]
            for i in range(len(cycles) - 1)]
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])

    med_gap = float(np.median(gaps_arr))
    p90_gap = float(np.percentile(gaps_arr, 90))
    cv = (res["duration"].std() / res["duration"].mean()
          if len(res) > 1 and res["duration"].mean() > 0 else 0.0)

    return {
        "total_time_sec": total_t,
        "total_cycles": len(cycles),
        "cycles_per_hour": cph,
        "utilization_pct": util * 100,
        "idle_ratio_pct": idle_r * 100,
        "avg_effort": avg_effort,
        "productivity_proxy": productivity,
        "consistency_cv": cv,
        "avg_cycle_sec": res["duration"].mean() if len(res) > 0 else 0,
        "best_cycle_sec": res["duration"].min() if len(res) > 0 else 0,
        "worst_cycle_sec": res["duration"].max() if len(res) > 0 else 0,
        "median_gap_sec": med_gap,
        "p90_gap_sec": p90_gap,
        "gaps_arr": gaps_arr,
    }


# ---------------------------------------------------------------------------
# Escenario OPTIMIZADO
# ---------------------------------------------------------------------------

def calcular_optimizado(real: dict, res: pd.DataFrame) -> dict:
    """
    Proyecta un escenario optimizado aplicando mejoras realistas:
    - Gaps recortados al mediano (mejor coordinación con camiones)
    - Utilización elevada al target de industria (85%)
    - Ciclos más consistentes (eliminar outliers lentos)
    """
    gaps_arr = real["gaps_arr"]
    med_gap = real["median_gap_sec"]

    # 1. Reducción de gaps al mediano
    opt_gaps = np.clip(gaps_arr, None, med_gap)
    gap_reduction = float(gaps_arr.mean() - opt_gaps.mean())

    # 2. Ciclos extra por reducción de gaps
    avg_cyc = real["avg_cycle_sec"] if real["avg_cycle_sec"] > 0 else 1.0
    extra_cph = gap_reduction * real["cycles_per_hour"] / avg_cyc

    # 3. Utilización mejorada al target
    opt_util = max(real["utilization_pct"], TARGET_UTILIZATION_PCT)

    # 4. Ciclos/hora optimizados
    opt_cph = real["cycles_per_hour"] + extra_cph
    # Bonus por mejor utilización
    util_bonus = (opt_util - real["utilization_pct"]) / 100.0
    opt_cph += opt_cph * util_bonus

    # 5. Tiempo de ciclo optimizado: usar el percentil 25 como referencia alcanzable
    if len(res) >= 4:
        p25_cycle = float(np.percentile(res["duration"], 25))
        opt_avg_cycle = p25_cycle
    else:
        opt_avg_cycle = real["best_cycle_sec"] * 1.05  # 5% sobre el mejor

    opt_cph_from_cycle = 3600 / opt_avg_cycle if opt_avg_cycle > 0 else opt_cph
    # Tomar el más conservador entre las dos estimaciones
    opt_cph_final = min(opt_cph, opt_cph_from_cycle)

    # 6. Productividad en toneladas
    real_tph = real["cycles_per_hour"] * BUCKET_TONNES * (real["utilization_pct"] / 100)
    opt_tph = opt_cph_final * BUCKET_TONNES * (opt_util / 100)

    delta_tph = opt_tph - real_tph
    delta_tpd = delta_tph * 20  # turno de 20 horas

    return {
        "cycles_per_hour": opt_cph_final,
        "utilization_pct": opt_util,
        "avg_cycle_sec": opt_avg_cycle,
        "gap_reduction_sec": gap_reduction,
        "extra_cycles_per_hour": extra_cph,
        "real_tph": real_tph,
        "opt_tph": opt_tph,
        "delta_tph": delta_tph,
        "delta_tpd": delta_tpd,
        "delta_pct": (delta_tph / real_tph * 100) if real_tph > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Insights / Recomendaciones
# ---------------------------------------------------------------------------

def generar_insights(res: pd.DataFrame, real: dict, opt: dict) -> list:
    ins = []

    if real["consistency_cv"] > 0.15:
        ins.append(
            f"⚠️  Alta variabilidad de ciclos (CV={real['consistency_cv']:.2f}) — "
            "estandarizar técnica de excavación."
        )

    if real["idle_ratio_pct"] > 15:
        ins.append(
            f"⚠️  {real['idle_ratio_pct']:.0f}% tiempo muerto — "
            "mejorar coordinación con camiones y reducir esperas."
        )

    if real["utilization_pct"] < TARGET_UTILIZATION_PCT:
        gap = TARGET_UTILIZATION_PCT - real["utilization_pct"]
        ins.append(
            f"⚠️  Utilización {real['utilization_pct']:.0f}% vs objetivo {TARGET_UTILIZATION_PCT:.0f}% "
            f"(brecha de {gap:.0f}pp)."
        )

    if real["median_gap_sec"] > TARGET_MAX_GAP_SEC:
        ins.append(
            f"⚠️  Gap mediano entre ciclos {real['median_gap_sec']:.1f}s > objetivo {TARGET_MAX_GAP_SEC}s — "
            "revisar posicionamiento de camiones."
        )

    if real["p90_gap_sec"] > real["median_gap_sec"] * 2:
        ins.append(
            f"⚠️  P90 de gaps ({real['p90_gap_sec']:.1f}s) muy superior a la mediana — "
            "hay eventos de espera prolongada recurrentes."
        )

    if len(res) > 1 and res["efficiency"].max() > res["efficiency"].mean() * 1.3:
        best_idx = res["efficiency"].idxmax()
        ins.append(
            f"💡 Ciclo {int(res.loc[best_idx, 'cycle'])} es el más eficiente "
            f"(eficiencia={res.loc[best_idx, 'efficiency']:.1f}) — "
            "replicar su técnica en todos los ciclos."
        )

    ins.append(
        f"💡 Optimizando gaps y utilización: "
        f"+{opt['delta_tph']:.0f} t/hr (+{opt['delta_pct']:.0f}%) = "
        f"+{opt['delta_tpd']:.0f} t/día (turno 20h)."
    )

    return ins


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

DARK_BG = "#1a1a2e"
PANEL_BG = "#16213e"
ACCENT = "#00d4ff"
GREEN = "#00ff88"
RED = "#ff6b6b"
YELLOW = "#ffd700"
ORANGE = "#ff9f43"
PURPLE = "#a29bfe"


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    if title:
        ax.set_title(title, fontsize=8, color="white", pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=7, color="white")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7, color="white")


def graficar(df_proc, cycles, res, real, opt, valid_segments, insights):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(DARK_BG)
    fig.suptitle(
        "EX-5600 — Análisis de Productividad y Comparativa de Optimización",
        fontsize=14, color="white", fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.38)

    # ── Fila 0: señal gz (ancho completo) ─────────────────────────────────
    ax_gz = fig.add_subplot(gs[0, :])
    _style_ax(ax_gz, "Giroscopio Z suavizado — detección de ciclos", "Tiempo (s)", "gz (rad/s)")
    ax_gz.plot(df_proc["time"], df_proc["gz_smooth"], color=ACCENT, linewidth=0.8, label="gz suavizado")
    ax_gz.axhline(GZ_THRESHOLD, color=RED, linestyle="--", linewidth=0.8, alpha=0.7, label=f"umbral ±{GZ_THRESHOLD}")
    ax_gz.axhline(-GZ_THRESHOLD, color=RED, linestyle="--", linewidth=0.8, alpha=0.7)
    for s, e in valid_segments:
        ax_gz.axvspan(df_proc["time"].iloc[s], df_proc["time"].iloc[e], alpha=0.15, color=YELLOW)
    ax_gz.legend(fontsize=6, loc="upper right", facecolor=DARK_BG, labelcolor="white")

    # ── Fila 1: duración, eficiencia, productividad por ciclo, lado ───────
    ax_dur = fig.add_subplot(gs[1, 0])
    _style_ax(ax_dur, "Duración por ciclo", "Ciclo", "Segundos")
    colors_dur = [GREEN if d <= res["duration"].quantile(0.5) else RED for d in res["duration"]]
    ax_dur.bar(res["cycle"], res["duration"], color=colors_dur, width=0.7)
    ax_dur.axhline(res["duration"].mean(), color=YELLOW, linestyle="--", linewidth=1,
                   label=f"media {res['duration'].mean():.1f}s")
    if real["avg_cycle_sec"] > 0:
        ax_dur.axhline(opt["avg_cycle_sec"], color=GREEN, linestyle=":", linewidth=1.2,
                       label=f"objetivo {opt['avg_cycle_sec']:.1f}s")
    ax_dur.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white")

    ax_eff = fig.add_subplot(gs[1, 1])
    _style_ax(ax_eff, "Eficiencia por ciclo", "Ciclo", "Eficiencia (proxy)")
    colors_eff = [GREEN if e >= res["efficiency"].mean() else RED for e in res["efficiency"]]
    ax_eff.bar(res["cycle"], res["efficiency"], color=colors_eff, width=0.7)
    ax_eff.axhline(res["efficiency"].mean(), color=YELLOW, linestyle="--", linewidth=1, label="media")
    ax_eff.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white")

    ax_prod = fig.add_subplot(gs[1, 2])
    _style_ax(ax_prod, "Productividad por ciclo (proxy)", "Ciclo", "Prod. proxy")
    res["cycle_rate"] = 3600 / res["duration"].replace(0, np.nan)
    res["prod_cycle"] = res["cycle_rate"] * res["effort"]
    ax_prod.plot(res["cycle"], res["prod_cycle"], color=ORANGE, marker="o", markersize=3, linewidth=1.2)
    ax_prod.fill_between(res["cycle"], res["prod_cycle"], alpha=0.2, color=ORANGE)

    ax_side = fig.add_subplot(gs[1, 3])
    _style_ax(ax_side, "Lado de descarga", "Ciclo", "")
    side_num = res["side"].map({"derecha": 1, "izquierda": -1})
    colors_side = [ACCENT if s == 1 else RED for s in side_num]
    ax_side.bar(res["cycle"], side_num, color=colors_side, width=0.7)
    ax_side.set_yticks([-1, 1])
    ax_side.set_yticklabels(["izquierda", "derecha"], fontsize=6, color="white")

    # ── Fila 2: gaps, histograma duración, scatter esfuerzo, scatter carga ─
    ax_gaps = fig.add_subplot(gs[2, 0])
    _style_ax(ax_gaps, "Gaps entre ciclos", "Intervalo", "Segundos")
    gaps_arr = real["gaps_arr"]
    if len(gaps_arr) > 0:
        colors_gap = [GREEN if g <= real["median_gap_sec"] else RED for g in gaps_arr]
        ax_gaps.bar(range(len(gaps_arr)), gaps_arr, color=colors_gap, width=0.7)
        ax_gaps.axhline(real["median_gap_sec"], color=YELLOW, linestyle="--", linewidth=1,
                        label=f"mediana {real['median_gap_sec']:.1f}s")
        ax_gaps.axhline(TARGET_MAX_GAP_SEC, color=GREEN, linestyle=":", linewidth=1.2,
                        label=f"objetivo {TARGET_MAX_GAP_SEC}s")
        ax_gaps.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white")

    ax_hist = fig.add_subplot(gs[2, 1])
    _style_ax(ax_hist, "Distribución duración ciclo", "Segundos", "Frecuencia")
    if len(res) >= 2:
        ax_hist.hist(res["duration"], bins=min(15, len(res)), color=PURPLE, edgecolor="#444", alpha=0.85)
    ax_hist.axvline(res["duration"].mean(), color=YELLOW, linestyle="--", linewidth=1,
                    label=f"media {res['duration'].mean():.1f}s")
    ax_hist.axvline(opt["avg_cycle_sec"], color=GREEN, linestyle=":", linewidth=1.5,
                    label=f"objetivo {opt['avg_cycle_sec']:.1f}s")
    ax_hist.legend(fontsize=6, facecolor=DARK_BG, labelcolor="white")

    ax_scat1 = fig.add_subplot(gs[2, 2])
    _style_ax(ax_scat1, "Esfuerzo vs Duración", "Duración (s)", "Esfuerzo")
    sc = ax_scat1.scatter(res["duration"], res["effort"], c=res["efficiency"],
                          cmap="RdYlGn", s=40, alpha=0.85)
    plt.colorbar(sc, ax=ax_scat1, label="Eficiencia").ax.yaxis.label.set_color("white")

    ax_scat2 = fig.add_subplot(gs[2, 3])
    _style_ax(ax_scat2, "Carga (pitch) vs Duración", "Duración (s)", "Rango pitch (rad)")
    ax_scat2.scatter(res["duration"], res["pitch_range"], color=ACCENT, s=40, alpha=0.85)

    # ── Fila 3: comparativa REAL vs OPTIMIZADO (barras) + insights ─────────
    ax_comp = fig.add_subplot(gs[3, :3])
    _style_ax(ax_comp, "Comparativa: Escenario Real vs Optimizado", "", "")
    _draw_comparativa(ax_comp, real, opt)

    ax_ins = fig.add_subplot(gs[3, 3])
    _style_ax(ax_ins, "Recomendaciones", "", "")
    _draw_insights_panel(ax_ins, insights, real, opt)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    return fig


def _draw_comparativa(ax, real: dict, opt: dict):
    """Gráfico de barras agrupadas: métricas reales vs optimizadas."""
    metricas = [
        ("Ciclos/hr",       real["cycles_per_hour"],    opt["cycles_per_hour"],    "ciclos/hr"),
        ("Utilización %",   real["utilization_pct"],    opt["utilization_pct"],    "%"),
        ("t/hr",            opt["real_tph"],             opt["opt_tph"],            "t/hr"),
        ("Ciclo prom. (s)", real["avg_cycle_sec"],       opt["avg_cycle_sec"],      "s"),
        ("Gap mediano (s)", real["median_gap_sec"],      TARGET_MAX_GAP_SEC,        "s"),
    ]

    labels = [m[0] for m in metricas]
    vals_real = [m[1] for m in metricas]
    vals_opt = [m[2] for m in metricas]
    units = [m[3] for m in metricas]

    x = np.arange(len(labels))
    w = 0.35

    bars_r = ax.bar(x - w / 2, vals_real, w, label="Real", color=RED, alpha=0.85)
    bars_o = ax.bar(x + w / 2, vals_opt, w, label="Optimizado", color=GREEN, alpha=0.85)

    # Etiquetas de valor
    for bar, val, unit in zip(bars_r, vals_real, units):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=6.5, color="white")
    for bar, val, unit in zip(bars_o, vals_opt, units):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=6.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, color="white")
    ax.tick_params(colors="white", labelsize=7)
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor="white")

    # Anotación de impacto
    delta_txt = (
        f"  Δ t/hr: +{opt['delta_tph']:.0f} t  (+{opt['delta_pct']:.0f}%)  |  "
        f"Δ t/día (20h): +{opt['delta_tpd']:.0f} t"
    )
    ax.text(0.5, 1.04, delta_txt, transform=ax.transAxes,
            ha="center", fontsize=9, color=YELLOW, fontweight="bold")

    for sp in ax.spines.values():
        sp.set_edgecolor("#444")


def _draw_insights_panel(ax, insights: list, real: dict, opt: dict):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#0f3460")
    for sp in ax.spines.values():
        sp.set_edgecolor(ACCENT)
        sp.set_linewidth(1.5)

    y = 0.97
    ax.text(0.04, y, "RECOMENDACIONES", transform=ax.transAxes,
            fontsize=8, color=YELLOW, fontweight="bold", va="top")
    y -= 0.12

    for ins in insights:
        # Dividir líneas largas
        words = ins.split()
        line, lines = "", []
        for w in words:
            if len(line) + len(w) + 1 > 38:
                lines.append(line)
                line = w
            else:
                line = (line + " " + w).strip()
        if line:
            lines.append(line)

        for l in lines:
            if y < 0.02:
                break
            ax.text(0.04, y, l, transform=ax.transAxes,
                    fontsize=6.5, color="white", va="top")
            y -= 0.09
        y -= 0.03


# ---------------------------------------------------------------------------
# Guardar reporte JSON
# ---------------------------------------------------------------------------

def guardar_reporte(res: pd.DataFrame, real: dict, opt: dict, insights: list) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "productivity_report.json"

    cycles_data = []
    for _, row in res.iterrows():
        cycles_data.append({
            "cycle_id": int(row["cycle"]),
            "duration_sec": round(float(row["duration"]), 2),
            "effort": round(float(row["effort"]), 2),
            "pitch_range_rad": round(float(row["pitch_range"]), 4),
            "smoothness": round(float(row["smoothness"]), 4),
            "lifting_time_sec": round(float(row["lifting_time"]), 2),
            "side": row["side"],
            "efficiency": round(float(row["efficiency"]), 4),
        })

    real_clean = {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else
                      (int(v) if isinstance(v, (int, np.integer)) else v))
                  for k, v in real.items() if k != "gaps_arr"}
    real_clean["gaps_arr"] = real["gaps_arr"].tolist()

    opt_clean = {k: round(float(v), 3) if isinstance(v, (float, np.floating)) else v
                 for k, v in opt.items()}

    report = {
        "real": real_clean,
        "optimizado": opt_clean,
        "cycles": cycles_data,
        "recommendations": insights,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Reporte JSON guardado en: {out}")
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EX-5600 — Reporte de productividad con comparativa de optimización"
    )
    parser.add_argument("--imu", type=str, default=None,
                        help="Ruta explícita al archivo .npy (opcional)")
    args = parser.parse_args()

    imu_path = find_imu_file(args.imu)
    print(f"[INFO] Cargando IMU: {imu_path}")
    df = load_imu(imu_path)
    print(f"[INFO] {len(df)} muestras, {df['time'].iloc[-1]:.1f}s de duración\n")

    df_proc = preprocess(df)
    cycles, valid_segments = detectar_ciclos(df_proc)
    res = calcular_metricas(cycles)

    print(f"Segmentos válidos detectados: {len(valid_segments)}")
    print(f"Ciclos completos detectados:  {len(cycles)}")

    if len(res) == 0:
        print("\n⚠️  No se detectaron ciclos válidos. Revisa los umbrales.")
        return

    real = calcular_productividad(df_proc, cycles, res)
    opt = calcular_optimizado(real, res)
    insights = generar_insights(res, real, opt)

    # ── Consola ──────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  ESCENARIO REAL")
    print("=" * 55)
    print(f"  Ciclos totales:       {real['total_cycles']}")
    print(f"  Ciclos/hora:          {real['cycles_per_hour']:.1f}")
    print(f"  Utilización:          {real['utilization_pct']:.1f}%")
    print(f"  Tiempo muerto:        {real['idle_ratio_pct']:.1f}%")
    print(f"  Duración media ciclo: {real['avg_cycle_sec']:.1f}s")
    print(f"  Mejor ciclo:          {real['best_cycle_sec']:.1f}s")
    print(f"  Peor ciclo:           {real['worst_cycle_sec']:.1f}s")
    print(f"  Consistencia (CV):    {real['consistency_cv']:.2f}")
    print(f"  Gap mediano:          {real['median_gap_sec']:.1f}s")
    print(f"  Gap P90:              {real['p90_gap_sec']:.1f}s")
    print(f"  Toneladas/hora:       {opt['real_tph']:.0f} t/hr")

    print("\n" + "=" * 55)
    print("  ESCENARIO OPTIMIZADO")
    print("=" * 55)
    print(f"  Ciclos/hora:          {opt['cycles_per_hour']:.1f}  (+{opt['extra_cycles_per_hour']:.1f})")
    print(f"  Utilización:          {opt['utilization_pct']:.1f}%")
    print(f"  Duración media ciclo: {opt['avg_cycle_sec']:.1f}s")
    print(f"  Reducción de gaps:    {opt['gap_reduction_sec']:.1f}s por ciclo")
    print(f"  Toneladas/hora:       {opt['opt_tph']:.0f} t/hr  (+{opt['delta_tph']:.0f}, +{opt['delta_pct']:.0f}%)")
    print(f"  Toneladas/día (20h):  +{opt['delta_tpd']:.0f} t adicionales")

    print("\n" + "=" * 55)
    print("  RECOMENDACIONES")
    print("=" * 55)
    for ins in insights:
        print(f"  {ins}")

    print("\nTop 10 ciclos más eficientes:")
    print(res.sort_values("efficiency", ascending=False).head(10)
          .to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # ── Gráficos ─────────────────────────────────────────────────────────
    fig = graficar(df_proc, cycles, res, real, opt, valid_segments, insights)

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUTS_DIR / "productivity_report.png"
    fig.savefig(str(fig_path), dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    print(f"\n[✓] Gráfico guardado en: {fig_path}")

    guardar_reporte(res, real, opt, insights)

    plt.show()


if __name__ == "__main__":
    main()