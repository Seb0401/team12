import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

# Tiempo en segundos
df['time'] = (df['time'] - df['time'].iloc[0]) / 1e9

# =========================
# 2. FEATURES BASE
# =========================

df['acc_mag'] = np.sqrt(df['ax']**2 + df['ay']**2 + df['az']**2)

# Suavizado más fuerte (clave)
df['gz_smooth'] = df['gz'].rolling(window=50).mean()

# =========================
# 3. ORIENTACIÓN (PITCH)
# =========================

def quaternion_to_pitch(qw, qx, qy, qz):
    return np.arcsin(2 * (qw*qy - qz*qx))

df['pitch'] = quaternion_to_pitch(df['qw'], df['qx'], df['qy'], df['qz'])

# =========================
# 4. DETECCIÓN ROBUSTA DE CICLOS
# =========================

threshold = 0.4  # ajustable
min_segment_time = 2.5  # segundos

# Detectar movimiento a la izquierda
df['moving_left'] = df['gz_smooth'] > threshold
# Si está invertido usa:
# df['moving_left'] = df['gz_smooth'] < -threshold

# ---- detectar segmentos continuos ----
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

# ---- filtrar segmentos cortos ----
valid_segments = []

for start, end in segments:
    duration = df['time'].iloc[end] - df['time'].iloc[start]

    if duration > min_segment_time:
        valid_segments.append((start, end))

print("Segmentos válidos (giro izquierda):", len(valid_segments))

# ---- construir ciclos reales ----
cycles = []

for i in range(len(valid_segments) - 1):
    start = valid_segments[i][0]
    end = valid_segments[i+1][0]

    cycle = df.iloc[start:end].copy()

    # Validar que haya giro de regreso (derecha)
    has_return = (cycle['gz_smooth'] < -threshold).sum() > 20

    if has_return:
        cycles.append(cycle)

print("Ciclos reales detectados:", len(cycles))

# =========================
# 5. VISUALIZACIÓN
# =========================

plt.figure(figsize=(12,4))
plt.plot(df['time'], df['gz_smooth'], label='gz_smooth')

for start, end in valid_segments:
    plt.axvspan(df['time'].iloc[start], df['time'].iloc[end], alpha=0.2)

plt.title("Segmentos reales de giro a la izquierda")
plt.xlabel("Tiempo (s)")
plt.ylabel("Giro Z")
plt.legend()
plt.show()

# =========================
# 6. MÉTRICAS POR CICLO
# =========================

results = []

for i, cycle in enumerate(cycles):

    duration = cycle['time'].iloc[-1] - cycle['time'].iloc[0]

    effort = cycle['acc_mag'].sum()
    pitch_range = cycle['pitch'].max() - cycle['pitch'].min()
    smoothness = cycle['acc_mag'].std()

    lifting = cycle['pitch'].diff() > 0
    lifting_time = lifting.sum() * cycle['time'].diff().mean()

    results.append({
        'cycle': i,
        'duration': duration,
        'effort': effort,
        'pitch_range': pitch_range,
        'smoothness': smoothness,
        'lifting_time': lifting_time
    })

results_df = pd.DataFrame(results)

# =========================
# 7. EFICIENCIA
# =========================

if len(results_df) > 0:
    results_df['efficiency'] = (
        results_df['pitch_range'] * results_df['effort']
    ) / results_df['duration']

# =========================
# 8. RANKING
# =========================

if len(results_df) > 0:
    best_cycles = results_df.sort_values(by='efficiency', ascending=False)

    print("\nTop 10 mejores ciclos:")
    print(best_cycles.head(10))

# =========================
# 9. GRÁFICOS
# =========================

if len(results_df) > 0:

    plt.figure()
    plt.hist(results_df['duration'], bins=20)
    plt.title("Distribución de tiempo por ciclo")
    plt.show()

    plt.figure()
    plt.scatter(results_df['duration'], results_df['effort'])
    plt.title("Esfuerzo vs Tiempo")
    plt.show()

    plt.figure()
    plt.scatter(results_df['duration'], results_df['pitch_range'])
    plt.title("Carga vs Tiempo")
    plt.show()

# =========================
# 10. INSIGHTS
# =========================

print("\n--- INSIGHTS ---")

if len(results_df) > 0:

    if results_df['duration'].std() > 5:
        print("⚠️ Alta variabilidad → operador inconsistente")

    idle_ratio = len(df[df['acc_mag'] < 0.2]) / len(df)

    if idle_ratio > 0.2:
        print("⚠️ Mucho tiempo muerto")

    if results_df['efficiency'].max() > results_df['efficiency'].mean() * 1.3:
        print("💡 Hay ciclos mucho mejores → optimización posible")

    print("\nTiempo promedio:", results_df['duration'].mean())
    print("Mejor eficiencia:", results_df['efficiency'].max())

else:
    print("No se detectaron ciclos válidos.")

top_20 = results_df.nlargest(int(len(results_df)*0.2), 'efficiency')

ideal = {
    'duration': top_20['duration'].mean(),
    'effort': top_20['effort'].mean(),
    'pitch_range': top_20['pitch_range'].mean(),
    'efficiency': top_20['efficiency'].mean()
}

