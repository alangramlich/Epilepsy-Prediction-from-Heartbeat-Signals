# === Celda 1: Tiempos → segundos → muestras + timeline (con colores) ===
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# ----- Parámetros editables -----
hora_inicio_registro = "19.39.33"
hora_inicio_crisis   = "19.58.36"
hora_fin_crisis      = "19.59.46"
hora_fin_registro    = "20.22.58"

frecuencia_muestreo = 512
LI_periodo_preictal = 15   # min antes de crisis
LS_periodo_preictal = 0    # min después de crisis

# ----- Helpers -----
def hms_a_segundos_rel(hms_ref: str, hms_evt: str) -> int:
    t0 = datetime.strptime(hms_ref, "%H.%M.%S")
    t1 = datetime.strptime(hms_evt, "%H.%M.%S")
    if t1 < t0:
        t1 += timedelta(days=1)
    return int((t1 - t0).total_seconds())

def seg_a_muestras(t_s: int, fs: int) -> int:
    return int(t_s * fs)

# ----- Cálculos -----
t0 = 0
t_crisis_ini_s = hms_a_segundos_rel(hora_inicio_registro, hora_inicio_crisis)
t_crisis_fin_s = hms_a_segundos_rel(hora_inicio_registro, hora_fin_crisis)
t_reg_fin_s    = hms_a_segundos_rel(hora_inicio_registro, hora_fin_registro)

pre_ini_s = max(0, t_crisis_ini_s - LI_periodo_preictal * 60)
pre_fin_s = t_crisis_fin_s + LS_periodo_preictal * 60

# A muestras
m_crisis_ini = seg_a_muestras(t_crisis_ini_s, frecuencia_muestreo)
m_crisis_fin = seg_a_muestras(t_crisis_fin_s, frecuencia_muestreo)
m_reg_fin    = seg_a_muestras(t_reg_fin_s, frecuencia_muestreo)
m_pre_ini    = seg_a_muestras(pre_ini_s, frecuencia_muestreo)
m_pre_fin    = seg_a_muestras(pre_fin_s, frecuencia_muestreo)

# ----- Tabla resumen -----
tabla_tiempos = pd.DataFrame([
    {"evento": "inicio_registro", "tiempo_s": t0,               "muestra": 0},
    {"evento": "inicio_crisis",   "tiempo_s": t_crisis_ini_s,   "muestra": m_crisis_ini},
    {"evento": "fin_crisis",      "tiempo_s": t_crisis_fin_s,   "muestra": m_crisis_fin},
    {"evento": "fin_registro",    "tiempo_s": t_reg_fin_s,      "muestra": m_reg_fin},
    {"evento": "preictal_ini",    "tiempo_s": pre_ini_s,        "muestra": m_pre_ini},
    {"evento": "preictal_fin",    "tiempo_s": pre_fin_s,        "muestra": m_pre_fin},
])
print("\n=== Tiempos y muestras (resumen) ===")
print(tabla_tiempos.to_string(index=False))

# ----- Gráfico con colores -----
plt.figure(figsize=(10, 3))

# Registro completo (gris)
plt.hlines(1.0, 0, t_reg_fin_s, linewidth=6, color="lightgray", label="Registro")

# Preictal (azul)
plt.hlines(1.15, pre_ini_s, pre_fin_s, linewidth=6, color="royalblue", label="Preictal")

# Crisis (rojo)
plt.hlines(0.85, t_crisis_ini_s, t_crisis_fin_s, linewidth=6, color="crimson", label="Crisis")

# Marcadores (círculos)
plt.plot([0], [1.0], "o", color="black")
plt.plot([t_crisis_ini_s], [1.0], "o", color="royalblue")
plt.plot([t_crisis_fin_s], [1.0], "o", color="crimson")
plt.plot([t_reg_fin_s], [1.0], "o", color="black")

plt.yticks([])
plt.xlabel("Tiempo [s]")
plt.title("Línea de tiempo: registro, preictal y crisis")
plt.legend(loc="upper left", frameon=False)
plt.tight_layout()
plt.show()



# === Celda 2: Leer EDF y graficar ECG crudo (manteniendo Fs hardcodeada) ===
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyedflib

# --- Parámetros (de Celda 1) ---
nombre = Path(r"PN00-1.edf")   # ruta al EDF
canal = 33
fs = frecuencia_muestreo       # usamos la hardcodeada (512 Hz)
ventana_seg = 10               # duración de ventana a graficar
inicio_seg = int(pre_ini_s)    # segundo desde el que arranca la ventana

# --- Lectura del canal ECG ---
if not nombre.exists():
    raise FileNotFoundError(f"No se encontró el archivo EDF en: {nombre}")

edf = pyedflib.EdfReader(str(nombre))
try:
    ecg_crudo = edf.readSignal(canal)
finally:
    edf.close()

# --- Eje temporal y recorte ---
eje_t = np.arange(ecg_crudo.size) / fs
ini = max(0, inicio_seg * fs)
fin = min(ini + ventana_seg * fs, ecg_crudo.size)

# --- Gráfico ECG crudo ---
plt.figure(figsize=(10, 4))
plt.plot(eje_t[ini:fin], ecg_crudo[ini:fin], color="black")
plt.title(f"ECG crudo – {ventana_seg}s desde t={inicio_seg}s")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [u]")
plt.grid(True)

# Líneas verticales de referencia si caen dentro de la ventana
for xline, lab, col in [
    (pre_ini_s, "pre_ini", "blue"),
    (pre_fin_s, "pre_fin", "blue"),
    (t_crisis_ini_s, "crisis_ini", "red"),
    (t_crisis_fin_s, "crisis_fin", "red"),
]:
    if eje_t[ini] <= xline <= eje_t[fin-1]:
        plt.axvline(xline, linestyle="--", linewidth=1, color=col)
        plt.text(xline, plt.ylim()[1], f" {lab}", va="top", fontsize=8, color=col)

plt.tight_layout()
plt.show()

# Resumen útil
dur_total_s = ecg_crudo.size / fs
print(f"Muestras: {ecg_crudo.size} | Fs usada: {fs} Hz | Duración total: {dur_total_s:.1f} s")
print(f"Ventana graficada: [{eje_t[ini]:.2f}s, {eje_t[fin-1]:.2f}s]")
