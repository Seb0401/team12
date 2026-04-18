import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# =========================
# CONFIGURACIÓN
# =========================

MODO = "tiempo_real"   # "completo" o "tiempo_real"
STEP = 10              # 10 filas ≈ 1 segundo
PAUSA = 0.2            # velocidad de simulación

# =========================
# 1. CARGAR DATOS
# =========================

data = np.load("40343737_20260313_110600_to_112100_imu.npy")

df = pd.DataFrame(data, columns=[
    'time',
    'ax', 'ay', 'az',
    'gx', 'gy', 'gz',
    'qw', 'qx', 'qy', 'qz'
])

df['time'] = (df['time'] - df['time'].iloc[0]) / 1e9

# =========================
# FUNCIONES
# =========================

def preprocess(df):
    df = df.copy()

    df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)
    df['gz_smooth'] = df['gz'].rolling(window=50).mean()

    def quaternion_to_pitch(qw, qx, qy, qz):
        return np.arcsin(2 * (qw*qy - qz*qx))

    df['pitch'] = quaternion_to_pitch(df['qw'], df['qx'], df['qy'], df['qz'])

    return df


def detectar_ciclos(df, threshold=0.4, min_segment_time=2.5):

    df['moving_left'] = df['gz_smooth'] > threshold

    segments = []
    in_segment = False

    for i in range(len(df)):

        if df['moving_left'].iloc[i] and not in_segment:
            start = i
            in_segment = True

        elif not df['moving_left'].iloc[i] and in_segment:
            end = i
            segments.append((start, end))
            in_segment = False

    valid_segments = []

    for start, end in segments:
        duration = df['time'].iloc[end] - df['time'].iloc[start]
        if duration > min_segment_time:
            valid_segments.append((start, end))

    cycles = []

    for i in range(len(valid_segments) - 1):
        start = valid_segments[i][0]
        end = valid_segments[i+1][0]

        cycle = df.iloc[start:end].copy()

        has_return = (cycle['gz_smooth'] < -threshold).sum() > 20

        if has_return:
            cycles.append(cycle)

    return cycles


def calcular_metricas(cycles):

    results = []

    for i, cycle in enumerate(cycles):

        duration = cycle['time'].iloc[-1] - cycle['time'].iloc[0]
        effort = cycle['acc_mag'].sum()
        pitch_range = cycle['pitch'].max() - cycle['pitch'].min()

        efficiency = (pitch_range * effort) / duration if duration > 0 else 0

        results.append({
            'cycle': i,
            'duration': duration,
            'effort': effort,
            'pitch_range': pitch_range,
            'efficiency': efficiency
        })

    return pd.DataFrame(results)


def graficar(df, results_df):

    plt.clf()

    # Señal
    plt.subplot(2,1,1)
    plt.plot(df['time'], df['gz_smooth'])
    plt.title("Giro (gz)")

    # Eficiencia
    if len(results_df) > 0:
        plt.subplot(2,1,2)
        plt.plot(results_df['cycle'], results_df['efficiency'], marker='o')
        plt.title("Eficiencia por ciclo")

    plt.pause(0.01)


# =========================
# 2. MODO COMPLETO
# =========================

if MODO == "completo":

    df_proc = preprocess(df)
    cycles = detectar_ciclos(df_proc)
    results_df = calcular_metricas(cycles)

    print(results_df.sort_values(by='efficiency', ascending=False).head(10))

    graficar(df_proc, results_df)
    plt.show()


# =========================
# 3. MODO TIEMPO REAL
# =========================

elif MODO == "tiempo_real":

    plt.ion()

    buffer = []

    try:
        for i in range(0, len(df), STEP):

            chunk = df.iloc[i:i+STEP]
            buffer.append(chunk)

            df_partial = pd.concat(buffer)

            df_proc = preprocess(df_partial)
            cycles = detectar_ciclos(df_proc)
            results_df = calcular_metricas(cycles)

            print(f"\nTiempo: {df_partial['time'].iloc[-1]:.1f}s | Ciclos: {len(cycles)}")

            graficar(df_proc, results_df)

            time.sleep(PAUSA)

    except KeyboardInterrupt:
        print("\n⛔ Proceso detenido por el usuario (todo bien)")

    plt.ioff()
    plt.show()
