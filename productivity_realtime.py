"""
EX-5600 Shovel — Reporte en Tiempo Real (solo IMU)
===================================================
Simula un stream de datos IMU procesando chunk a chunk y actualizando
un dashboard de matplotlib en vivo.

Uso:
    python realtime_report.py                  # modo tiempo_real (default)
    python realtime_report.py --modo completo  # análisis batch completo
    python realtime_report.py --step 5         # chunks más pequeños (más fluido)
    python realtime_report.py --pausa 0.05     # más rápido
"""

import argparse
import glob
import json
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

INPUTS_DIR = Path("./inputs")
OUTPUTS_DIR = Path("./outputs")
IMU_GLOB = "*imu*.npy"

# Ruta de fallback si no hay archivo en inputs/
FALLBACK_IMU = Path("C:/Hackathon/Archive/40343737_20260313_110600_to_112100_imu.npy")

# Parámetros de detección de ciclos
GZ_THRESHOLD = 0.4          # umbral de giro Z para detectar swing
MIN_SEGMENT_TIME = 2.5      # segundos mínimos para considerar un segmento válido
GZ_SMOOTH_WINDOW = 50       # ventana de suavizado del giroscopio Z
IDLE_THRESHOLD = 0.3        # magnitud acelerométrica por debajo = idle

# Specs EX-5600
BUCKET_TONNES = 52.0        # toneladas nominales por balde


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def find_imu_file() -> Path:
    """Busca el archivo IMU en inputs/ o usa el fallback."""
    matches = glob.glob(str(INPUTS_DIR / IMU_GLOB))
    if matches:
        return Path(matches[0])
    if FALLBACK_IMU.exists():
        print(f"[INFO] Usando archivo IMU de fallback: {FALLBACK_IMU}")
        return FALLBACK_IMU
    raise FileNotFoundError(
        f"No se encontró archivo IMU en {INPUTS_DIR} ni en {FALLBACK_IMU}"
    )


def load_imu(path: Path) -> pd.DataFrame:
    """Carga el .npy y devuelve un DataFrame con columnas nombradas."""
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
    """Calcula features derivados sobre el DataFrame acumulado."""
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
    """
    Detecta ciclos de excavación a partir del giroscopio Z suavizado.

    Retorna lista de DataFrames, uno por ciclo detectado.
    """
    df["moving_left"] = df["gz_smooth"] > GZ_THRESHOLD

    segments = []
    in_segment = False
    start = 0

    for i in range(len(df)):
        val = df["moving_left"].iloc[i]
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i))
            in_segment = False

    # Filtrar segmentos cortos
    valid_segments = []
    for s, e in segments:
        duration = df["time"].iloc[e] - df["time"].iloc[s]
        if duration > MIN_SEGMENT_TIME:
            valid_segments.append((s, e))

    # Construir ciclos bidireccionales
    cycles = []
    for i in range(len(valid_segments) - 1):
        s = valid_segments[i][0]
        e = valid_segments[i + 1][0]
        cycle = df.iloc[s:e].copy()
        has_right = (cycle["gz_smooth"] < -GZ_THRESHOLD).sum() > 20
        has_left = (cycle["gz_smooth"] > GZ_THRESHOLD).sum() > 20
        if has_right or has_left:
            cycles.append(cycle)

    return cycles, valid_segments


# ---------------------------------------------------------------------------
# Métricas por ciclo
# ---------------------------------------------------------------------------

def calcular_metricas(cycles: list) -> pd.DataFrame:
    """Calcula métricas de productividad para cada ciclo detectado."""
    results = []
    for i, cycle in enumerate(cycles):
        duration = cycle["time"].iloc[-1] - cycle["time"].iloc[0]
        effort = cycle["acc_mag"].sum()
        pitch_range = cycle["pitch"].max() - cycle["pitch"].min()
        smoothness = cycle["acc_mag"].std()

        lifting = cycle["pitch"].diff() > 0
        dt_mean = cycle["time"].diff().mean()
        lifting_time = lifting.sum() * dt_mean if not np.isnan(dt_mean) else 0.0

        mean_gz = cycle["gz_smooth"].mean()
        side = "izquierda" if mean_gz > 0 else "derecha"

        efficiency = (pitch_range * effort) / duration if duration > 0 else 0.0

        results.append({
            "cycle": i,
            "duration": duration,
            "effort": effort,
            "pitch_range": pitch_range,
            "smoothness": smoothness,
            "lifting_time": lifting_time,
            "side": side,
            "efficiency": efficiency,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Modelo de productividad
# ---------------------------------------------------------------------------

def calcular_productividad(df: pd.DataFrame, cycles: list, results_df: pd.DataFrame) -> dict:
    """Calcula métricas de productividad estilo OEE minero."""
    total_time = df["time"].iloc[-1] - df["time"].iloc[0]
    total_time_hours = total_time / 3600 if total_time > 0 else 1e-9

    cycles_per_hour = len(cycles) / total_time_hours

    df_idle = df["acc_mag"] < IDLE_THRESHOLD
    dt_mean = df["time"].diff().mean()
    idle_time = df_idle.sum() * (dt_mean if not np.isnan(dt_mean) else 0.0)
    idle_ratio = idle_time / total_time if total_time > 0 else 0.0
    utilization = max(0.0, 1.0 - idle_ratio)

    avg_payload_proxy = results_df["effort"].mean() if len(results_df) > 0 else 0.0
    productivity = cycles_per_hour * avg_payload_proxy * utilization

    # Gaps entre ciclos
    gaps = []
    for i in range(len(cycles) - 1):
        gap = cycles[i + 1]["time"].iloc[0] - cycles[i]["time"].iloc[-1]
        gaps.append(gap)
    gaps_arr = np.array(gaps) if gaps else np.array([0.0])

    median_gap = float(np.median(gaps_arr))
    p90_gap = float(np.percentile(gaps_arr, 90))
    optimized_gaps = np.clip(gaps_arr, None, median_gap)
    gap_reduction = float(gaps_arr.mean() - optimized_gaps.mean())

    avg_cycle_time = results_df["duration"].mean() if len(results_df) > 0 else 1.0
    extra_cycles_per_hour = (gap_reduction * cycles_per_hour / avg_cycle_time
                             if avg_cycle_time > 0 else 0.0)
    extra_tonnes_per_hour = extra_cycles_per_hour * BUCKET_TONNES

    consistency_cv = (results_df["duration"].std() / results_df["duration"].mean()
                      if len(results_df) > 1 and results_df["duration"].mean() > 0
                      else 0.0)

    return {
        "total_time_sec": total_time,
        "cycles_per_hour": cycles_per_hour,
        "utilization_pct": utilization * 100,
        "idle_ratio_pct": idle_ratio * 100,
        "avg_payload_proxy": avg_payload_proxy,
        "productivity_proxy": productivity,
        "consistency_cv": consistency_cv,
        "median_gap_sec": median_gap,
        "p90_gap_sec": p90_gap,
        "gap_reduction_sec": gap_reduction,
        "extra_cycles_per_hour": extra_cycles_per_hour,
        "extra_tonnes_per_hour": extra_tonnes_per_hour,
        "extra_tonnes_per_day_20h": extra_tonnes_per_hour * 20,
    }


# ---------------------------------------------------------------------------
# Insights / Recomendaciones
# ---------------------------------------------------------------------------

def generar_insights(results_df: pd.DataFrame, prod: dict) -> list:
    """Genera recomendaciones accionables basadas en las métricas."""
    insights = []

    if len(results_df) == 0:
        return ["⏳ Esperando datos suficientes para generar insights..."]

    if results_df["duration"].std() > 5:
        insights.append("⚠️  Alta variabilidad en tiempos de ciclo → operador inconsistente")

    if prod["idle_ratio_pct"] > 20:
        insights.append(f"⚠️  {prod['idle_ratio_pct']:.0f}% tiempo muerto → revisar posicionamiento de camiones")

    if prod["utilization_pct"] < 75:
        insights.append(f"⚠️  Utilización {prod['utilization_pct']:.0f}% < 75% objetivo → reducir esperas")

    if len(results_df) > 1:
        if results_df["efficiency"].max() > results_df["efficiency"].mean() * 1.3:
            insights.append("💡 Hay ciclos mucho más eficientes → optimización posible")

    if prod["median_gap_sec"] > 10:
        insights.append(f"⚠️  Gap mediano entre ciclos: {prod['median_gap_sec']:.1f}s → mejorar coordinación con camiones")

    if prod["extra_tonnes_per_hour"] > 0:
        insights.append(
            f"💡 Reduciendo gaps al mediano: +{prod['extra_cycles_per_hour']:.1f} ciclos/hr "
            f"= +{prod['extra_tonnes_per_hour']:.0f} t/hr "
            f"(+{prod['extra_tonnes_per_day_20h']:.0f} t/día)"
        )

    if prod["consistency_cv"] < 0.1 and len(results_df) >= 3:
        insights.append("✅ Operador muy consistente (CV < 10%)")

    if not insights:
        insights.append(
            f"✅ Buen desempeño: {prod['cycles_per_hour']:.0f} ciclos/hr, "
            f"{prod['utilization_pct']:.0f}% utilización"
        )

    return insights


# ---------------------------------------------------------------------------
# Dashboard matplotlib
# ---------------------------------------------------------------------------

def crear_dashboard():
    """Crea la figura del dashboard con subplots organizados."""
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle(
        "EX-5600 — Dashboard de Productividad en Tiempo Real",
        fontsize=14, color="white", fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = {
        "gz":          fig.add_subplot(gs[0, :]),   # fila 0, toda la fila
        "efficiency":  fig.add_subplot(gs[1, 0]),
        "duration":    fig.add_subplot(gs[1, 1]),
        "productivity":fig.add_subplot(gs[1, 2]),
        "side":        fig.add_subplot(gs[2, 0]),
        "gaps":        fig.add_subplot(gs[2, 1]),
        "kpis":        fig.add_subplot(gs[2, 2]),
    }

    for ax in axes.values():
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.title.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")

    return fig, axes


def actualizar_dashboard(fig, axes, df_proc, cycles, results_df, prod, insights, valid_segments):
    """Redibuja todos los subplots con los datos actuales."""

    # ── Señal gz ──────────────────────────────────────────────────────────
    ax = axes["gz"]
    ax.cla()
    ax.set_facecolor("#16213e")
    ax.plot(df_proc["time"], df_proc["gz_smooth"], color="#00d4ff", linewidth=0.8, label="gz suavizado")
    ax.axhline(GZ_THRESHOLD, color="#ff6b6b", linestyle="--", linewidth=0.8, alpha=0.7, label=f"umbral ±{GZ_THRESHOLD}")
    ax.axhline(-GZ_THRESHOLD, color="#ff6b6b", linestyle="--", linewidth=0.8, alpha=0.7)
    for s, e in valid_segments:
        ax.axvspan(df_proc["time"].iloc[s], df_proc["time"].iloc[e], alpha=0.15, color="#ffd700")
    ax.set_title(f"Giroscopio Z — {len(cycles)} ciclos detectados", fontsize=9)
    ax.set_xlabel("Tiempo (s)", fontsize=7)
    ax.set_ylabel("gz (rad/s)", fontsize=7)
    ax.legend(fontsize=6, loc="upper right", facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    if len(results_df) == 0:
        for key in ["efficiency", "duration", "productivity", "side", "gaps", "kpis"]:
            axes[key].cla()
            axes[key].set_facecolor("#16213e")
            axes[key].text(0.5, 0.5, "Esperando ciclos...", ha="center", va="center",
                           color="#888", fontsize=8, transform=axes[key].transAxes)
            for spine in axes[key].spines.values():
                spine.set_edgecolor("#444")
        _draw_insights(axes["kpis"], insights, prod)
        return

    # ── Eficiencia por ciclo ───────────────────────────────────────────────
    ax = axes["efficiency"]
    ax.cla()
    ax.set_facecolor("#16213e")
    colors_eff = ["#00ff88" if e >= results_df["efficiency"].mean() else "#ff6b6b"
                  for e in results_df["efficiency"]]
    ax.bar(results_df["cycle"], results_df["efficiency"], color=colors_eff, width=0.7)
    ax.axhline(results_df["efficiency"].mean(), color="#ffd700", linestyle="--", linewidth=1, label="media")
    ax.set_title("Eficiencia por ciclo", fontsize=8)
    ax.set_xlabel("Ciclo", fontsize=7)
    ax.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── Distribución de duración ───────────────────────────────────────────
    ax = axes["duration"]
    ax.cla()
    ax.set_facecolor("#16213e")
    if len(results_df) >= 2:
        ax.hist(results_df["duration"], bins=min(15, len(results_df)),
                color="#7b68ee", edgecolor="#444", alpha=0.85)
    else:
        ax.bar(results_df["cycle"], results_df["duration"], color="#7b68ee")
    ax.axvline(results_df["duration"].mean(), color="#ffd700", linestyle="--", linewidth=1,
               label=f"media {results_df['duration'].mean():.1f}s")
    ax.set_title("Distribución duración ciclo", fontsize=8)
    ax.set_xlabel("Segundos", fontsize=7)
    ax.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="white")
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── Productividad acumulada ────────────────────────────────────────────
    ax = axes["productivity"]
    ax.cla()
    ax.set_facecolor("#16213e")
    results_df["cycle_rate"] = 3600 / results_df["duration"].replace(0, np.nan)
    results_df["prod_cycle"] = results_df["cycle_rate"] * results_df["effort"]
    ax.plot(results_df["cycle"], results_df["prod_cycle"],
            color="#ff9f43", marker="o", markersize=3, linewidth=1.2)
    ax.fill_between(results_df["cycle"], results_df["prod_cycle"], alpha=0.2, color="#ff9f43")
    ax.set_title("Productividad por ciclo (proxy)", fontsize=8)
    ax.set_xlabel("Ciclo", fontsize=7)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── Lado de descarga ───────────────────────────────────────────────────
    ax = axes["side"]
    ax.cla()
    ax.set_facecolor("#16213e")
    side_numeric = results_df["side"].map({"derecha": 1, "izquierda": -1})
    colors_side = ["#00d4ff" if s == 1 else "#ff6b6b" for s in side_numeric]
    ax.bar(results_df["cycle"], side_numeric, color=colors_side, width=0.7)
    ax.set_yticks([-1, 1])
    ax.set_yticklabels(["izquierda", "derecha"], fontsize=6, color="white")
    ax.set_title("Lado de descarga por ciclo", fontsize=8)
    ax.set_xlabel("Ciclo", fontsize=7)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── Gaps entre ciclos ──────────────────────────────────────────────────
    ax = axes["gaps"]
    ax.cla()
    ax.set_facecolor("#16213e")
    if len(cycles) > 1:
        gaps = [cycles[i + 1]["time"].iloc[0] - cycles[i]["time"].iloc[-1]
                for i in range(len(cycles) - 1)]
        gaps_arr = np.array(gaps)
        ax.bar(range(len(gaps_arr)), gaps_arr, color="#a29bfe", width=0.7)
        ax.axhline(np.median(gaps_arr), color="#ffd700", linestyle="--", linewidth=1,
                   label=f"mediana {np.median(gaps_arr):.1f}s")
        ax.legend(fontsize=6, facecolor="#1a1a2e", labelcolor="white")
    ax.set_title("Gaps entre ciclos (s)", fontsize=8)
    ax.set_xlabel("Intervalo", fontsize=7)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # ── KPIs + Insights ────────────────────────────────────────────────────
    _draw_insights(axes["kpis"], insights, prod)


def _draw_insights(ax, insights, prod):
    """Dibuja el panel de KPIs e insights textuales."""
    ax.cla()
    ax.set_facecolor("#0f3460")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#00d4ff")
        spine.set_linewidth(1.5)

    lines = [
        ("KPIs", "#ffd700", 10, "bold"),
        (f"Ciclos/hr:    {prod.get('cycles_per_hour', 0):.1f}", "#00ff88", 8, "normal"),
        (f"Utilización:  {prod.get('utilization_pct', 0):.1f}%", "#00ff88", 8, "normal"),
        (f"Tiempo muerto:{prod.get('idle_ratio_pct', 0):.1f}%", "#ff9f43", 8, "normal"),
        (f"Consist. (CV):{prod.get('consistency_cv', 0):.2f}", "#00d4ff", 8, "normal"),
        (f"+t/hr optim.: {prod.get('extra_tonnes_per_hour', 0):.0f} t", "#ff6b6b", 8, "normal"),
        ("", "white", 6, "normal"),
        ("INSIGHTS", "#ffd700", 9, "bold"),
    ]

    y = 0.97
    for text, color, size, weight in lines:
        ax.text(0.05, y, text, transform=ax.transAxes,
                fontsize=size, color=color, fontweight=weight,
                verticalalignment="top", fontfamily="monospace")
        y -= 0.10

    for ins in insights[:4]:
        ax.text(0.05, y, ins, transform=ax.transAxes,
                fontsize=6.5, color="white",
                verticalalignment="top", wrap=True)
        y -= 0.13

    ax.set_title("KPIs & Recomendaciones", fontsize=8, color="white")


# ---------------------------------------------------------------------------
# Guardar reporte JSON
# ---------------------------------------------------------------------------

def guardar_reporte(results_df: pd.DataFrame, prod: dict, insights: list) -> Path:
    """Serializa el reporte final a outputs/realtime_report.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS_DIR / "realtime_report.json"

    cycles_data = []
    if len(results_df) > 0:
        for _, row in results_df.iterrows():
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

    report = {
        "summary": {k: round(v, 3) if isinstance(v, float) else v
                    for k, v in prod.items()},
        "cycles": cycles_data,
        "recommendations": insights,
    }

    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n[✓] Reporte guardado en: {out}")
    return out


# ---------------------------------------------------------------------------
# Modo COMPLETO (batch)
# ---------------------------------------------------------------------------

def modo_completo(df: pd.DataFrame) -> None:
    """Procesa todos los datos de una vez y muestra los gráficos finales."""
    print("\n=== MODO COMPLETO ===")

    df_proc = preprocess(df)
    cycles, valid_segments = detectar_ciclos(df_proc)
    results_df = calcular_metricas(cycles)

    print(f"Segmentos válidos: {len(valid_segments)}")
    print(f"Ciclos detectados: {len(cycles)}")

    if len(results_df) > 0:
        prod = calcular_productividad(df_proc, cycles, results_df)
        insights = generar_insights(results_df, prod)

        print("\n--- MÉTRICAS ---")
        print(f"Ciclos/hora:    {prod['cycles_per_hour']:.1f}")
        print(f"Utilización:    {prod['utilization_pct']:.1f}%")
        print(f"Tiempo muerto:  {prod['idle_ratio_pct']:.1f}%")
        print(f"Consistencia:   CV={prod['consistency_cv']:.2f}")
        print(f"Gap mediano:    {prod['median_gap_sec']:.1f}s")
        print(f"Gap P90:        {prod['p90_gap_sec']:.1f}s")
        print(f"\n--- OPTIMIZACIÓN ---")
        print(f"Reducción gap:  {prod['gap_reduction_sec']:.1f}s")
        print(f"Ciclos extra/hr:{prod['extra_cycles_per_hour']:.1f}")
        print(f"Toneladas extra/hr: {prod['extra_tonnes_per_hour']:.0f} t")
        print(f"Toneladas extra/día (20h): {prod['extra_tonnes_per_day_20h']:.0f} t")

        print("\n--- INSIGHTS ---")
        for ins in insights:
            print(ins)

        print("\nTop 10 ciclos más eficientes:")
        print(results_df.sort_values("efficiency", ascending=False).head(10).to_string(index=False))

        fig, axes = crear_dashboard()
        actualizar_dashboard(fig, axes, df_proc, cycles, results_df, prod, insights, valid_segments)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        guardar_reporte(results_df, prod, insights)
        plt.show()
    else:
        print("No se detectaron ciclos válidos.")


# ---------------------------------------------------------------------------
# Modo TIEMPO REAL (streaming simulado)
# ---------------------------------------------------------------------------

def modo_tiempo_real(df: pd.DataFrame, step: int = 10, pausa: float = 0.2) -> None:
    """
    Simula un stream procesando el IMU en chunks de `step` filas.
    Actualiza el dashboard en cada paso.
    """
    print(f"\n=== MODO TIEMPO REAL (step={step}, pausa={pausa}s) ===")
    print("Presiona Ctrl+C para detener y guardar el reporte.\n")

    plt.ion()
    fig, axes = crear_dashboard()
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.pause(0.1)

    buffer = []
    results_df = pd.DataFrame()
    prod = {k: 0.0 for k in [
        "cycles_per_hour", "utilization_pct", "idle_ratio_pct",
        "avg_payload_proxy", "productivity_proxy", "consistency_cv",
        "median_gap_sec", "p90_gap_sec", "gap_reduction_sec",
        "extra_cycles_per_hour", "extra_tonnes_per_hour",
        "extra_tonnes_per_day_20h", "total_time_sec",
    ]}
    insights = ["⏳ Iniciando análisis..."]
    cycles = []
    valid_segments = []

    try:
        for i in range(0, len(df), step):
            chunk = df.iloc[i: i + step]
            buffer.append(chunk)
            df_partial = pd.concat(buffer, ignore_index=True)

            df_proc = preprocess(df_partial)
            cycles, valid_segments = detectar_ciclos(df_proc)
            results_df = calcular_metricas(cycles)

            if len(results_df) > 0:
                prod = calcular_productividad(df_proc, cycles, results_df)
                insights = generar_insights(results_df, prod)

            t_actual = df_partial["time"].iloc[-1]
            print(
                f"\rTiempo: {t_actual:6.1f}s | "
                f"Ciclos: {len(cycles):3d} | "
                f"Ciclos/hr: {prod['cycles_per_hour']:5.1f} | "
                f"Util: {prod['utilization_pct']:4.1f}%",
                end="", flush=True,
            )

            actualizar_dashboard(
                fig, axes, df_proc, cycles, results_df, prod, insights, valid_segments
            )
            plt.pause(pausa)

    except KeyboardInterrupt:
        print("\n\n⛔ Proceso detenido por el usuario.")

    finally:
        plt.ioff()
        print("\n\n=== RESUMEN FINAL ===")
        if len(results_df) > 0:
            print(f"Ciclos detectados:  {len(cycles)}")
            print(f"Ciclos/hora:        {prod['cycles_per_hour']:.1f}")
            print(f"Utilización:        {prod['utilization_pct']:.1f}%")
            print(f"Tiempo muerto:      {prod['idle_ratio_pct']:.1f}%")
            print(f"Consistencia CV:    {prod['consistency_cv']:.2f}")
            print(f"Gap mediano:        {prod['median_gap_sec']:.1f}s")
            print(f"Extra t/hr optim.:  {prod['extra_tonnes_per_hour']:.0f} t")
            print(f"Extra t/día (20h):  {prod['extra_tonnes_per_day_20h']:.0f} t")
            print("\nInsights:")
            for ins in insights:
                print(f"  {ins}")
            guardar_reporte(results_df, prod, insights)
        else:
            print("No se detectaron ciclos válidos.")

        plt.show()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EX-5600 — Reporte de productividad en tiempo real (IMU)"
    )
    parser.add_argument(
        "--modo", choices=["tiempo_real", "completo"], default="tiempo_real",
        help="Modo de ejecución (default: tiempo_real)"
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help="Filas IMU por chunk en modo tiempo_real (default: 10 ≈ 1s)"
    )
    parser.add_argument(
        "--pausa", type=float, default=0.15,
        help="Segundos de pausa entre chunks (default: 0.15)"
    )
    parser.add_argument(
        "--imu", type=str, default=None,
        help="Ruta explícita al archivo .npy (opcional)"
    )
    args = parser.parse_args()

    # Cargar datos
    imu_path = Path(args.imu) if args.imu else find_imu_file()
    print(f"[INFO] Cargando IMU: {imu_path}")
    df = load_imu(imu_path)
    print(f"[INFO] {len(df)} muestras, {df['time'].iloc[-1]:.1f}s de duración")

    if args.modo == "completo":
        modo_completo(df)
    else:
        modo_tiempo_real(df, step=args.step, pausa=args.pausa)


if __name__ == "__main__":
    main()
