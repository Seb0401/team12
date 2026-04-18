"""
EX-5600 Shovel — Reporte de Productividad Interactivo
======================================================
Dashboard navegable con 5 vistas accesibles mediante botones:

  [1] Resumen General    — KPIs principales y recomendaciones
  [2] Ciclos Detallados  — Top 10 mejores ciclos + tabla completa
  [3] Ciclo Óptimo       — Análisis del mejor ciclo y cómo replicarlo
  [4] Comparativa        — Real vs Optimizado (barras + impacto)
  [5] Señal IMU          — Giroscopio Z y detección de ciclos

Uso:
    python productivity_report.py
    python productivity_report.py --imu ruta/al/archivo.npy
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
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
BUCKET_TONNES = 52.0

TARGET_UTILIZATION_PCT = 85.0
TARGET_MAX_GAP_SEC = 8.0

# Paleta de colores
DARK_BG  = "#1a1a2e"
PANEL_BG = "#16213e"
BTN_BG   = "#0f3460"
BTN_ACT  = "#e94560"
ACCENT   = "#00d4ff"
GREEN    = "#00ff88"
RED      = "#ff6b6b"
YELLOW   = "#ffd700"
ORANGE   = "#ff9f43"
PURPLE   = "#a29bfe"
WHITE    = "#ffffff"


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
    raise FileNotFoundError(f"No se encontró archivo IMU en {INPUTS_DIR}")


def load_imu(path):
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

def preprocess(df):
    df = df.copy()
    df["acc_mag"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    df["gz_smooth"] = df["gz"].rolling(window=GZ_SMOOTH_WINDOW, min_periods=1).mean()
    df["pitch"] = np.arcsin(
        np.clip(2 * (df["qw"] * df["qy"] - df["qz"] * df["qx"]), -1.0, 1.0)
    )
    return df


# ---------------------------------------------------------------------------
# Detección de ciclos
# ---------------------------------------------------------------------------

def detectar_ciclos(df):
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

def calcular_metricas(cycles):
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
# Productividad real
# ---------------------------------------------------------------------------

def calcular_productividad(df, cycles, res):
    total_t = df["time"].iloc[-1] - df["time"].iloc[0]
    total_h = total_t / 3600 if total_t > 0 else 1e-9
    cph = len(cycles) / total_h

    dt = df["time"].diff().mean()
    idle_t = (df["acc_mag"] < IDLE_THRESHOLD).sum() * (dt if not np.isnan(dt) else 0)
    idle_r = idle_t / total_t if total_t > 0 else 0
    util = max(0.0, 1.0 - idle_r)

    gaps = [cycles[i+1]["time"].iloc[0] - cycles[i]["time"].iloc[-1]
            for i in range(len(cycles) - 1)]
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])

    cv = (res["duration"].std() / res["duration"].mean()
          if len(res) > 1 and res["duration"].mean() > 0 else 0.0)

    return {
        "total_time_sec": total_t,
        "total_cycles": len(cycles),
        "cycles_per_hour": cph,
        "utilization_pct": util * 100,
        "idle_ratio_pct": idle_r * 100,
        "avg_effort": res["effort"].mean() if len(res) > 0 else 0.0,
        "consistency_cv": cv,
        "avg_cycle_sec": res["duration"].mean() if len(res) > 0 else 0,
        "best_cycle_sec": res["duration"].min() if len(res) > 0 else 0,
        "worst_cycle_sec": res["duration"].max() if len(res) > 0 else 0,
        "median_gap_sec": float(np.median(gaps_arr)),
        "p90_gap_sec": float(np.percentile(gaps_arr, 90)),
        "gaps_arr": gaps_arr,
    }


# ---------------------------------------------------------------------------
# Escenario optimizado
# ---------------------------------------------------------------------------

def calcular_optimizado(real, res):
    gaps_arr = real["gaps_arr"]
    med_gap = real["median_gap_sec"]

    opt_gaps = np.clip(gaps_arr, None, med_gap)
    gap_reduction = float(gaps_arr.mean() - opt_gaps.mean())

    avg_cyc = real["avg_cycle_sec"] if real["avg_cycle_sec"] > 0 else 1.0
    extra_cph = gap_reduction * real["cycles_per_hour"] / avg_cyc

    opt_util = max(real["utilization_pct"], TARGET_UTILIZATION_PCT)
    opt_cph = real["cycles_per_hour"] + extra_cph
    util_bonus = (opt_util - real["utilization_pct"]) / 100.0
    opt_cph += opt_cph * util_bonus

    if len(res) >= 4:
        opt_avg_cycle = float(np.percentile(res["duration"], 25))
    else:
        opt_avg_cycle = real["best_cycle_sec"] * 1.05

    opt_cph_from_cycle = 3600 / opt_avg_cycle if opt_avg_cycle > 0 else opt_cph
    opt_cph_final = min(opt_cph, opt_cph_from_cycle)

    real_tph = real["cycles_per_hour"] * BUCKET_TONNES * (real["utilization_pct"] / 100)
    opt_tph = opt_cph_final * BUCKET_TONNES * (opt_util / 100)
    delta_tph = opt_tph - real_tph

    return {
        "cycles_per_hour": opt_cph_final,
        "utilization_pct": opt_util,
        "avg_cycle_sec": opt_avg_cycle,
        "gap_reduction_sec": gap_reduction,
        "extra_cycles_per_hour": extra_cph,
        "real_tph": real_tph,
        "opt_tph": opt_tph,
        "delta_tph": delta_tph,
        "delta_tpd": delta_tph * 20,
        "delta_pct": (delta_tph / real_tph * 100) if real_tph > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Insights
# ---------------------------------------------------------------------------

def generar_insights(res, real, opt):
    ins = []
    if real["consistency_cv"] > 0.15:
        ins.append(f"⚠️  Alta variabilidad (CV={real['consistency_cv']:.2f}) — estandarizar técnica.")
    if real["idle_ratio_pct"] > 15:
        ins.append(f"⚠️  {real['idle_ratio_pct']:.0f}% tiempo muerto — mejorar coordinación con camiones.")
    if real["utilization_pct"] < TARGET_UTILIZATION_PCT:
        ins.append(f"⚠️  Utilización {real['utilization_pct']:.0f}% vs objetivo {TARGET_UTILIZATION_PCT:.0f}%.")
    if real["median_gap_sec"] > TARGET_MAX_GAP_SEC:
        ins.append(f"⚠️  Gap mediano {real['median_gap_sec']:.1f}s > objetivo {TARGET_MAX_GAP_SEC}s.")
    if real["p90_gap_sec"] > real["median_gap_sec"] * 2:
        ins.append(f"⚠️  P90 gaps ({real['p90_gap_sec']:.1f}s) >> mediana — esperas prolongadas recurrentes.")
    if len(res) > 1 and res["efficiency"].max() > res["efficiency"].mean() * 1.3:
        best_idx = res["efficiency"].idxmax()
        ins.append(f"💡 Ciclo #{int(res.loc[best_idx,'cycle'])} es el más eficiente — replicar su técnica.")
    ins.append(
        f"💡 Optimizando: +{opt['delta_tph']:.0f} t/hr (+{opt['delta_pct']:.0f}%) "
        f"= +{opt['delta_tpd']:.0f} t/día (20h)."
    )
    return ins


# ---------------------------------------------------------------------------
# Helpers de estilo
# ---------------------------------------------------------------------------

def _style_ax(ax, title="", xlabel="", ylabel="", title_size=11):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#555")
    if title:
        ax.set_title(title, fontsize=title_size, color=WHITE, pad=6, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8, color="#aaa")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8, color="#aaa")


def _clear_content_area(fig, content_axes):
    """Elimina todos los ejes del área de contenido."""
    for ax in content_axes:
        fig.delaxes(ax)
    return []


# ---------------------------------------------------------------------------
# ══════════════════════════  VISTAS  ══════════════════════════════════════
# ---------------------------------------------------------------------------

def vista_resumen(fig, content_rect, res, real, opt, insights):
    """Vista 1: KPIs principales + recomendaciones."""
    gs = gridspec.GridSpec(2, 3,
                           left=content_rect[0], right=content_rect[2],
                           bottom=content_rect[1], top=content_rect[3],
                           hspace=0.45, wspace=0.35)
    axes = []

    # ── Panel KPIs ────────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[0, :2])
    axes.append(ax_kpi)
    ax_kpi.set_facecolor(BTN_BG)
    ax_kpi.set_xticks([]); ax_kpi.set_yticks([])
    for sp in ax_kpi.spines.values():
        sp.set_edgecolor(ACCENT); sp.set_linewidth(1.5)
    ax_kpi.set_title("KPIs — Escenario Real", fontsize=11, color=YELLOW,
                     fontweight="bold", pad=6)

    kpis = [
        ("Ciclos totales",       f"{real['total_cycles']}",                    WHITE),
        ("Ciclos / hora",        f"{real['cycles_per_hour']:.1f}",             ACCENT),
        ("Utilización",          f"{real['utilization_pct']:.1f}%",
         GREEN if real['utilization_pct'] >= 75 else RED),
        ("Tiempo muerto",        f"{real['idle_ratio_pct']:.1f}%",
         GREEN if real['idle_ratio_pct'] <= 15 else RED),
        ("Duración media ciclo", f"{real['avg_cycle_sec']:.1f} s",             WHITE),
        ("Mejor ciclo",          f"{real['best_cycle_sec']:.1f} s",            GREEN),
        ("Peor ciclo",           f"{real['worst_cycle_sec']:.1f} s",           RED),
        ("Consistencia (CV)",    f"{real['consistency_cv']:.2f}",
         GREEN if real['consistency_cv'] <= 0.15 else ORANGE),
        ("Gap mediano",          f"{real['median_gap_sec']:.1f} s",
         GREEN if real['median_gap_sec'] <= TARGET_MAX_GAP_SEC else RED),
        ("Gap P90",              f"{real['p90_gap_sec']:.1f} s",               ORANGE),
        ("Toneladas / hora",     f"{opt['real_tph']:.0f} t",                   YELLOW),
        ("Duración total",       f"{real['total_time_sec']/60:.1f} min",       WHITE),
    ]

    cols = 3
    for idx, (label, value, color) in enumerate(kpis):
        col = idx % cols
        row = idx // cols
        x = 0.03 + col * 0.34
        y = 0.88 - row * 0.22
        ax_kpi.text(x, y, label, transform=ax_kpi.transAxes,
                    fontsize=8, color="#aaa", va="top")
        ax_kpi.text(x, y - 0.10, value, transform=ax_kpi.transAxes,
                    fontsize=13, color=color, va="top", fontweight="bold",
                    fontfamily="monospace")

    # ── Panel impacto optimización ─────────────────────────────────────────
    ax_imp = fig.add_subplot(gs[0, 2])
    axes.append(ax_imp)
    ax_imp.set_facecolor("#0a1628")
    ax_imp.set_xticks([]); ax_imp.set_yticks([])
    for sp in ax_imp.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2)
    ax_imp.set_title("Potencial de Mejora", fontsize=10, color=GREEN,
                     fontweight="bold", pad=6)

    imp_lines = [
        ("OPTIMIZADO", YELLOW, 12, "bold"),
        (f"+{opt['delta_tph']:.0f} t/hr", GREEN, 22, "bold"),
        (f"+{opt['delta_pct']:.0f}% productividad", WHITE, 10, "normal"),
        ("", WHITE, 6, "normal"),
        (f"+{opt['delta_tpd']:.0f} t/día", ORANGE, 16, "bold"),
        ("(turno 20 horas)", WHITE, 8, "normal"),
        ("", WHITE, 6, "normal"),
        (f"Ciclos/hr: {real['cycles_per_hour']:.0f} → {opt['cycles_per_hour']:.0f}", ACCENT, 9, "normal"),
        (f"Util: {real['utilization_pct']:.0f}% → {opt['utilization_pct']:.0f}%", ACCENT, 9, "normal"),
        (f"Ciclo: {real['avg_cycle_sec']:.0f}s → {opt['avg_cycle_sec']:.0f}s", ACCENT, 9, "normal"),
    ]
    y = 0.95
    for text, color, size, weight in imp_lines:
        ax_imp.text(0.5, y, text, transform=ax_imp.transAxes,
                    fontsize=size, color=color, fontweight=weight,
                    ha="center", va="top")
        y -= 0.09 if size > 10 else 0.08

    # ── Panel recomendaciones ──────────────────────────────────────────────
    ax_rec = fig.add_subplot(gs[1, :])
    axes.append(ax_rec)
    ax_rec.set_facecolor(BTN_BG)
    ax_rec.set_xticks([]); ax_rec.set_yticks([])
    for sp in ax_rec.spines.values():
        sp.set_edgecolor(YELLOW); sp.set_linewidth(1.5)
    ax_rec.set_title("Recomendaciones", fontsize=11, color=YELLOW,
                     fontweight="bold", pad=6)

    y = 0.92
    for ins in insights:
        color = RED if "⚠️" in ins else (GREEN if "✅" in ins else ACCENT)
        ax_rec.text(0.02, y, ins, transform=ax_rec.transAxes,
                    fontsize=9, color=color, va="top")
        y -= 0.13
        if y < 0.02:
            break

    return axes


def vista_ciclos(fig, content_rect, res):
    """Vista 2: Top 10 ciclos + gráficos de eficiencia y duración."""
    gs = gridspec.GridSpec(2, 2,
                           left=content_rect[0], right=content_rect[2],
                           bottom=content_rect[1], top=content_rect[3],
                           hspace=0.45, wspace=0.35)
    axes = []

    top10 = res.sort_values("efficiency", ascending=False).head(10).reset_index(drop=True)

    # ── Tabla Top 10 ──────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[0, :])
    axes.append(ax_tbl)
    ax_tbl.set_facecolor(BTN_BG)
    ax_tbl.set_xticks([]); ax_tbl.set_yticks([])
    for sp in ax_tbl.spines.values():
        sp.set_edgecolor(YELLOW); sp.set_linewidth(1.5)
    ax_tbl.set_title("🏆  Top 10 Ciclos Más Eficientes", fontsize=11,
                     color=YELLOW, fontweight="bold", pad=6)

    headers = ["#", "Ciclo", "Duración (s)", "Esfuerzo", "Pitch (rad)",
               "Suavidad", "T. Elevación (s)", "Lado", "Eficiencia"]
    col_x   = [0.01, 0.06, 0.16, 0.27, 0.38, 0.49, 0.60, 0.73, 0.84]

    # Cabecera
    for hdr, cx in zip(headers, col_x):
        ax_tbl.text(cx, 0.93, hdr, transform=ax_tbl.transAxes,
                    fontsize=7.5, color=YELLOW, fontweight="bold", va="top")

    ax_tbl.axhline(0.88, color="#555", linewidth=0.8, transform=ax_tbl.transAxes)

    # Filas
    row_h = 0.082
    for rank, (_, row) in enumerate(top10.iterrows()):
        y = 0.86 - rank * row_h
        bg_color = "#1e2d4a" if rank % 2 == 0 else BTN_BG
        ax_tbl.axhspan(y - row_h * 0.85, y + row_h * 0.1,
                       xmin=0, xmax=1, color=bg_color, transform=ax_tbl.transAxes)

        eff_color = GREEN if row["efficiency"] >= top10["efficiency"].mean() else ORANGE
        vals = [
            (f"{rank+1}", WHITE),
            (f"{int(row['cycle'])}", ACCENT),
            (f"{row['duration']:.1f}", WHITE),
            (f"{row['effort']:.1f}", WHITE),
            (f"{row['pitch_range']:.3f}", WHITE),
            (f"{row['smoothness']:.3f}", WHITE),
            (f"{row['lifting_time']:.1f}", WHITE),
            (row["side"], PURPLE),
            (f"{row['efficiency']:.2f}", eff_color),
        ]
        for (val, color), cx in zip(vals, col_x):
            ax_tbl.text(cx, y, val, transform=ax_tbl.transAxes,
                        fontsize=7.5, color=color, va="top", fontfamily="monospace")

    # ── Barras de eficiencia ───────────────────────────────────────────────
    ax_eff = fig.add_subplot(gs[1, 0])
    axes.append(ax_eff)
    _style_ax(ax_eff, "Eficiencia — todos los ciclos", "Ciclo", "Eficiencia")
    colors_eff = [GREEN if e >= res["efficiency"].mean() else RED for e in res["efficiency"]]
    ax_eff.bar(res["cycle"], res["efficiency"], color=colors_eff, width=0.7, alpha=0.9)
    ax_eff.axhline(res["efficiency"].mean(), color=YELLOW, linestyle="--",
                   linewidth=1.2, label=f"media {res['efficiency'].mean():.2f}")
    # Marcar top 10
    for _, row in top10.iterrows():
        ax_eff.bar(row["cycle"], row["efficiency"], color=YELLOW, width=0.7, alpha=0.5)
    ax_eff.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Duración por ciclo ─────────────────────────────────────────────────
    ax_dur = fig.add_subplot(gs[1, 1])
    axes.append(ax_dur)
    _style_ax(ax_dur, "Duración por ciclo", "Ciclo", "Segundos")
    colors_dur = [GREEN if d <= res["duration"].quantile(0.5) else RED
                  for d in res["duration"]]
    ax_dur.bar(res["cycle"], res["duration"], color=colors_dur, width=0.7, alpha=0.9)
    ax_dur.axhline(res["duration"].mean(), color=YELLOW, linestyle="--",
                   linewidth=1.2, label=f"media {res['duration'].mean():.1f}s")
    ax_dur.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


def vista_ciclo_optimo(fig, content_rect, res, cycles, real, opt):
    """Vista 3: Análisis del ciclo óptimo y cómo replicarlo."""
    gs = gridspec.GridSpec(2, 3,
                           left=content_rect[0], right=content_rect[2],
                           bottom=content_rect[1], top=content_rect[3],
                           hspace=0.50, wspace=0.38)
    axes = []

    best_idx = res["efficiency"].idxmax()
    best_row = res.loc[best_idx]
    best_cycle_data = cycles[int(best_row["cycle"])]

    # ── Señal del ciclo óptimo ─────────────────────────────────────────────
    ax_sig = fig.add_subplot(gs[0, :2])
    axes.append(ax_sig)
    _style_ax(ax_sig,
              f"Ciclo #{int(best_row['cycle'])} — El Más Eficiente "
              f"(eficiencia={best_row['efficiency']:.2f})",
              "Tiempo relativo (s)", "Valor")
    t_rel = best_cycle_data["time"] - best_cycle_data["time"].iloc[0]
    ax_sig.plot(t_rel, best_cycle_data["gz_smooth"], color=ACCENT,
                linewidth=1.5, label="gz suavizado")
    ax_sig.plot(t_rel, best_cycle_data["pitch"], color=ORANGE,
                linewidth=1.5, label="pitch (rad)")
    ax_sig.plot(t_rel, best_cycle_data["acc_mag"] / best_cycle_data["acc_mag"].max(),
                color=PURPLE, linewidth=1, alpha=0.7, label="acc_mag (norm.)")
    ax_sig.axhline(0, color="#555", linewidth=0.5)
    ax_sig.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Panel de características del ciclo óptimo ─────────────────────────
    ax_char = fig.add_subplot(gs[0, 2])
    axes.append(ax_char)
    ax_char.set_facecolor("#0a1628")
    ax_char.set_xticks([]); ax_char.set_yticks([])
    for sp in ax_char.spines.values():
        sp.set_edgecolor(GREEN); sp.set_linewidth(2)
    ax_char.set_title("Características\ndel Ciclo Óptimo", fontsize=10,
                      color=GREEN, fontweight="bold", pad=4)

    char_lines = [
        ("Duración",      f"{best_row['duration']:.1f} s",       ACCENT),
        ("Esfuerzo",      f"{best_row['effort']:.1f}",            WHITE),
        ("Rango pitch",   f"{best_row['pitch_range']:.3f} rad",   WHITE),
        ("Suavidad",      f"{best_row['smoothness']:.3f}",        WHITE),
        ("T. elevación",  f"{best_row['lifting_time']:.1f} s",    WHITE),
        ("Lado descarga", best_row["side"],                        PURPLE),
        ("Eficiencia",    f"{best_row['efficiency']:.2f}",        GREEN),
    ]
    y = 0.90
    for label, val, color in char_lines:
        ax_char.text(0.05, y, label, transform=ax_char.transAxes,
                     fontsize=8, color="#aaa", va="top")
        ax_char.text(0.55, y, val, transform=ax_char.transAxes,
                     fontsize=9, color=color, va="top", fontweight="bold",
                     fontfamily="monospace")
        y -= 0.12

    # ── Comparación: mejor vs promedio vs peor ─────────────────────────────
    ax_comp3 = fig.add_subplot(gs[1, :2])
    axes.append(ax_comp3)
    _style_ax(ax_comp3, "Mejor vs Promedio vs Peor ciclo", "", "Valor")

    worst_idx = res["efficiency"].idxmin()
    worst_row = res.loc[worst_idx]
    avg_row = res.mean(numeric_only=True)

    metricas_comp = ["duration", "effort", "pitch_range", "smoothness", "efficiency"]
    labels_comp   = ["Duración (s)", "Esfuerzo", "Pitch (rad)", "Suavidad", "Eficiencia"]

    x = np.arange(len(metricas_comp))
    w = 0.25

    def _norm(vals):
        mx = max(abs(v) for v in vals) or 1
        return [v / mx for v in vals]

    best_vals  = [float(best_row[m]) for m in metricas_comp]
    avg_vals   = [float(avg_row[m])  for m in metricas_comp]
    worst_vals = [float(worst_row[m]) for m in metricas_comp]

    # Normalizar para visualización comparativa
    all_vals = best_vals + avg_vals + worst_vals
    mx = max(abs(v) for v in all_vals) or 1
    bv = [v/mx for v in best_vals]
    av = [v/mx for v in avg_vals]
    wv = [v/mx for v in worst_vals]

    ax_comp3.bar(x - w, bv, w, label=f"Mejor (#{int(best_row['cycle'])})",
                 color=GREEN, alpha=0.9)
    ax_comp3.bar(x,     av, w, label="Promedio", color=YELLOW, alpha=0.9)
    ax_comp3.bar(x + w, wv, w, label=f"Peor (#{int(worst_row['cycle'])})",
                 color=RED, alpha=0.9)
    ax_comp3.set_xticks(x)
    ax_comp3.set_xticklabels(labels_comp, fontsize=7.5, color="white")
    ax_comp3.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")
    ax_comp3.set_ylabel("Valor normalizado", fontsize=8, color="#aaa")

    # ── Panel: cómo replicar el ciclo óptimo ──────────────────────────────
    ax_how = fig.add_subplot(gs[1, 2])
    axes.append(ax_how)
    ax_how.set_facecolor(BTN_BG)
    ax_how.set_xticks([]); ax_how.set_yticks([])
    for sp in ax_how.spines.values():
        sp.set_edgecolor(YELLOW); sp.set_linewidth(1.5)
    ax_how.set_title("Cómo Replicarlo", fontsize=10, color=YELLOW,
                     fontweight="bold", pad=4)

    dur_diff = real["avg_cycle_sec"] - float(best_row["duration"])
    eff_diff = float(best_row["efficiency"]) - float(avg_row["efficiency"])

    tips = [
        f"⏱  Reducir duración {dur_diff:.1f}s",
        f"   (actual: {real['avg_cycle_sec']:.1f}s → objetivo: {best_row['duration']:.1f}s)",
        "",
        f"📐 Mantener pitch range ≥ {best_row['pitch_range']:.3f} rad",
        f"   (mayor carga por ciclo)",
        "",
        f"🔄 Lado de descarga: {best_row['side']}",
        f"   (consistencia en dirección)",
        "",
        f"📈 Eficiencia objetivo: {best_row['efficiency']:.2f}",
        f"   (+{eff_diff:.2f} vs promedio actual)",
        "",
        f"🎯 Si todos los ciclos fueran",
        f"   como el mejor:",
        f"   +{(best_row['efficiency']/avg_row['efficiency']-1)*100:.0f}% eficiencia",
    ]
    y = 0.95
    for tip in tips:
        color = ACCENT if tip.startswith(("⏱", "📐", "🔄", "📈", "🎯")) else WHITE
        ax_how.text(0.04, y, tip, transform=ax_how.transAxes,
                    fontsize=7.5, color=color, va="top")
        y -= 0.065
        if y < 0.02:
            break

    return axes


def vista_comparativa(fig, content_rect, res, real, opt):
    """Vista 4: Comparativa real vs optimizado."""
    gs = gridspec.GridSpec(2, 2,
                           left=content_rect[0], right=content_rect[2],
                           bottom=content_rect[1], top=content_rect[3],
                           hspace=0.50, wspace=0.38)
    axes = []

    # ── Barras agrupadas: métricas clave ──────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, :])
    axes.append(ax_bar)
    _style_ax(ax_bar, "Comparativa de Métricas: Real vs Optimizado", "", "")

    metricas = [
        ("Ciclos/hr",       real["cycles_per_hour"],   opt["cycles_per_hour"],   ""),
        ("Utilización %",   real["utilization_pct"],   opt["utilization_pct"],   "%"),
        ("t/hr",            opt["real_tph"],            opt["opt_tph"],           "t"),
        ("Ciclo prom. (s)", real["avg_cycle_sec"],      opt["avg_cycle_sec"],     "s"),
        ("Gap mediano (s)", real["median_gap_sec"],     TARGET_MAX_GAP_SEC,       "s"),
    ]
    labels    = [m[0] for m in metricas]
    vals_real = [m[1] for m in metricas]
    vals_opt  = [m[2] for m in metricas]
    units     = [m[3] for m in metricas]

    x = np.arange(len(labels))
    w = 0.35
    bars_r = ax_bar.bar(x - w/2, vals_real, w, label="Real",       color=RED,   alpha=0.85)
    bars_o = ax_bar.bar(x + w/2, vals_opt,  w, label="Optimizado", color=GREEN, alpha=0.85)

    for bar, val, unit in zip(bars_r, vals_real, units):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=8, color="white")
    for bar, val, unit in zip(bars_o, vals_opt, units):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=8, color="white")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=9, color="white")
    ax_bar.legend(fontsize=9, facecolor=DARK_BG, labelcolor="white")
    ax_bar.text(0.5, 1.04,
                f"Δ t/hr: +{opt['delta_tph']:.0f} t  (+{opt['delta_pct']:.0f}%)  |  "
                f"Δ t/día (20h): +{opt['delta_tpd']:.0f} t",
                transform=ax_bar.transAxes, ha="center", fontsize=10,
                color=YELLOW, fontweight="bold")

    # ── Línea de productividad acumulada real vs proyectada ────────────────
    ax_line = fig.add_subplot(gs[1, 0])
    axes.append(ax_line)
    _style_ax(ax_line, "Productividad acumulada (t)", "Ciclo", "Toneladas acumuladas")

    real_tph_per_cycle = opt["real_tph"] / real["cycles_per_hour"] if real["cycles_per_hour"] > 0 else 0
    opt_tph_per_cycle  = opt["opt_tph"]  / opt["cycles_per_hour"]  if opt["cycles_per_hour"]  > 0 else 0

    n = real["total_cycles"]
    x_cyc = np.arange(1, n + 1)
    ax_line.plot(x_cyc, x_cyc * real_tph_per_cycle, color=RED,   linewidth=2,
                 label=f"Real ({opt['real_tph']:.0f} t/hr)")
    ax_line.plot(x_cyc, x_cyc * opt_tph_per_cycle,  color=GREEN, linewidth=2,
                 linestyle="--", label=f"Optimizado ({opt['opt_tph']:.0f} t/hr)")
    ax_line.fill_between(x_cyc,
                         x_cyc * real_tph_per_cycle,
                         x_cyc * opt_tph_per_cycle,
                         alpha=0.15, color=GREEN)
    ax_line.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Distribución de duración: real vs objetivo ─────────────────────────
    ax_hist = fig.add_subplot(gs[1, 1])
    axes.append(ax_hist)
    _style_ax(ax_hist, "Distribución duración: Real vs Objetivo", "Segundos", "Frecuencia")
    if len(res) >= 2:
        ax_hist.hist(res["duration"], bins=min(15, len(res)),
                     color=RED, alpha=0.6, label="Real", edgecolor="#333")
    ax_hist.axvline(real["avg_cycle_sec"], color=RED, linestyle="--",
                    linewidth=1.5, label=f"media real {real['avg_cycle_sec']:.1f}s")
    ax_hist.axvline(opt["avg_cycle_sec"], color=GREEN, linestyle=":",
                    linewidth=2, label=f"objetivo {opt['avg_cycle_sec']:.1f}s")
    ax_hist.axvline(real["best_cycle_sec"], color=YELLOW, linestyle="-.",
                    linewidth=1.5, label=f"mejor {real['best_cycle_sec']:.1f}s")
    ax_hist.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


def vista_imu(fig, content_rect, df_proc, cycles, valid_segments, real):
    """Vista 5: Señal IMU completa y detección de ciclos."""
    gs = gridspec.GridSpec(3, 1,
                           left=content_rect[0], right=content_rect[2],
                           bottom=content_rect[1], top=content_rect[3],
                           hspace=0.45)
    axes = []

    # ── Giroscopio Z ──────────────────────────────────────────────────────
    ax_gz = fig.add_subplot(gs[0])
    axes.append(ax_gz)
    _style_ax(ax_gz,
              f"Giroscopio Z — {real['total_cycles']} ciclos detectados",
              "Tiempo (s)", "gz (rad/s)")
    ax_gz.plot(df_proc["time"], df_proc["gz_smooth"], color=ACCENT,
               linewidth=0.9, label="gz suavizado")
    ax_gz.axhline(GZ_THRESHOLD,  color=RED, linestyle="--", linewidth=0.8,
                  alpha=0.7, label=f"umbral ±{GZ_THRESHOLD}")
    ax_gz.axhline(-GZ_THRESHOLD, color=RED, linestyle="--", linewidth=0.8, alpha=0.7)
    for s, e in valid_segments:
        ax_gz.axvspan(df_proc["time"].iloc[s], df_proc["time"].iloc[e],
                      alpha=0.12, color=YELLOW)
    ax_gz.legend(fontsize=7, loc="upper right", facecolor=DARK_BG, labelcolor="white")

    # ── Pitch ─────────────────────────────────────────────────────────────
    ax_pitch = fig.add_subplot(gs[1])
    axes.append(ax_pitch)
    _style_ax(ax_pitch, "Pitch (orientación)", "Tiempo (s)", "Pitch (rad)")
    ax_pitch.plot(df_proc["time"], df_proc["pitch"], color=ORANGE,
                  linewidth=0.9, label="pitch")
    for s, e in valid_segments:
        ax_pitch.axvspan(df_proc["time"].iloc[s], df_proc["time"].iloc[e],
                         alpha=0.12, color=YELLOW)
    ax_pitch.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    # ── Magnitud acelerómetro ──────────────────────────────────────────────
    ax_acc = fig.add_subplot(gs[2])
    axes.append(ax_acc)
    _style_ax(ax_acc, "Magnitud Acelerómetro", "Tiempo (s)", "acc_mag")
    ax_acc.plot(df_proc["time"], df_proc["acc_mag"], color=PURPLE,
                linewidth=0.7, alpha=0.8, label="acc_mag")
    ax_acc.axhline(IDLE_THRESHOLD, color=RED, linestyle="--", linewidth=0.8,
                   label=f"umbral idle {IDLE_THRESHOLD}")
    ax_acc.legend(fontsize=7, facecolor=DARK_BG, labelcolor="white")

    return axes


# ---------------------------------------------------------------------------
# Dashboard principal con botones
# ---------------------------------------------------------------------------

class Dashboard:
    VIEWS = [
        ("📊 Resumen",      "resumen"),
        ("🏆 Top Ciclos",   "ciclos"),
        ("⭐ Ciclo Óptimo", "optimo"),
        ("📈 Comparativa",  "comparativa"),
        ("📡 Señal IMU",    "imu"),
    ]

    def __init__(self, df_proc, cycles, res, real, opt, valid_segments, insights):
        self.df_proc = df_proc
        self.cycles = cycles
        self.res = res
        self.real = real
        self.opt = opt
        self.valid_segments = valid_segments
        self.insights = insights

        self.current_view = "resumen"
        self.content_axes = []

        self._build_figure()
        self._show_view("resumen")

    def _build_figure(self):
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor(DARK_BG)

        # Título
        self.fig.text(0.5, 0.97,
                      "EX-5600 — Dashboard de Productividad",
                      ha="center", fontsize=15, color=WHITE,
                      fontweight="bold")

        # Área de botones: franja superior
        n = len(self.VIEWS)
        btn_w = 0.14
        btn_h = 0.045
        btn_y = 0.915
        gap = (1.0 - n * btn_w) / (n + 1)

        self.buttons = {}
        self.btn_axes = {}
        for i, (label, key) in enumerate(self.VIEWS):
            bx = gap + i * (btn_w + gap)
            ax_btn = self.fig.add_axes([bx, btn_y, btn_w, btn_h])
            btn = Button(ax_btn, label,
                         color=BTN_BG, hovercolor="#1e3a5f")
            btn.label.set_color(WHITE)
            btn.label.set_fontsize(9)
            btn.label.set_fontweight("bold")
            btn.on_clicked(lambda event, k=key: self._show_view(k))
            self.buttons[key] = btn
            self.btn_axes[key] = ax_btn

        # Línea separadora
        self.fig.add_axes([0.02, 0.905, 0.96, 0.002]).set_facecolor("#444")

        # Área de contenido
        self.content_rect = [0.03, 0.04, 0.97, 0.895]

    def _show_view(self, view_key):
        # Limpiar ejes anteriores
        for ax in self.content_axes:
            try:
                self.fig.delaxes(ax)
            except Exception:
                pass
        self.content_axes = []

        # Resaltar botón activo
        for key, ax_btn in self.btn_axes.items():
            ax_btn.set_facecolor(BTN_ACT if key == view_key else BTN_BG)

        self.current_view = view_key

        if view_key == "resumen":
            self.content_axes = vista_resumen(
                self.fig, self.content_rect,
                self.res, self.real, self.opt, self.insights)
        elif view_key == "ciclos":
            self.content_axes = vista_ciclos(
                self.fig, self.content_rect, self.res)
        elif view_key == "optimo":
            self.content_axes = vista_ciclo_optimo(
                self.fig, self.content_rect,
                self.res, self.cycles, self.real, self.opt)
        elif view_key == "comparativa":
            self.content_axes = vista_comparativa(
                self.fig, self.content_rect,
                self.res, self.real, self.opt)
        elif view_key == "imu":
            self.content_axes = vista_imu(
                self.fig, self.content_rect,
                self.df_proc, self.cycles, self.valid_segments, self.real)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ---------------------------------------------------------------------------
# Guardar reporte JSON
# ---------------------------------------------------------------------------

def guardar_reporte(res, real, opt, insights):
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
        "top10_cycles": cycles_data[:10],
        "all_cycles": cycles_data,
        "recommendations": insights,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[✓] Reporte JSON guardado en: {out}")
    return out


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EX-5600 — Dashboard interactivo de productividad"
    )
    parser.add_argument("--imu", type=str, default=None,
                        help="Ruta explícita al archivo .npy (opcional)")
    args = parser.parse_args()

    imu_path = find_imu_file(args.imu)
    print(f"[INFO] Cargando IMU: {imu_path}")
    df = load_imu(imu_path)
    print(f"[INFO] {len(df)} muestras, {df['time'].iloc[-1]:.1f}s de duración")

    df_proc = preprocess(df)
    cycles, valid_segments = detectar_ciclos(df_proc)
    res = calcular_metricas(cycles)

    print(f"[INFO] Ciclos detectados: {len(cycles)}")

    if len(res) == 0:
        print("⚠️  No se detectaron ciclos válidos.")
        return

    real = calcular_productividad(df_proc, cycles, res)
    opt  = calcular_optimizado(real, res)
    insights = generar_insights(res, real, opt)

    # Consola rápida
    print(f"\n  Real:       {real['cycles_per_hour']:.1f} ciclos/hr | "
          f"{real['utilization_pct']:.0f}% util | {opt['real_tph']:.0f} t/hr")
    print(f"  Optimizado: {opt['cycles_per_hour']:.1f} ciclos/hr | "
          f"{opt['utilization_pct']:.0f}% util | {opt['opt_tph']:.0f} t/hr "
          f"(+{opt['delta_tph']:.0f} t/hr, +{opt['delta_pct']:.0f}%)")

    guardar_reporte(res, real, opt, insights)

    dash = Dashboard(df_proc, cycles, res, real, opt, valid_segments, insights)
    dash.show()


if __name__ == "__main__":
    main()
