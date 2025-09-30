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



# === Celda 2: Leer EDF y graficar ECG crudo ===
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
from biosppy.signals import ecg as ekg
# --- Parámetros (de Celda 1) ---
nombre = Path(r"PN00-1.edf")   # ruta al EDF
canal = 33
fs = frecuencia_muestreo       # usamos 512 Hz
ventana_seg = 10               # duración de ventana a graficar
inicio_seg = int(pre_ini_s)    # segundo desde el que arranca la ventana
archivo_edf = pyedflib.EdfReader("PN00-1.edf")
datos_canal = archivo_edf.readSignal(canal)
datos_canal = datos_canal[:512*8] #recorto la señal a los primeros 8 seg

# --------------------------------------------GRAFICO EL ECG SIN FILTRAR
frecuencia_muestreo = 512  
duracion = 8  # segundos

# número de muestras
n_muestras = duracion * frecuencia_muestreo  

# segmento desde el inicio
segmento = datos_canal

# eje de tiempo en segundos
tiempo = np.arange(0, duracion, 1/frecuencia_muestreo)

# graficar
plt.figure(figsize=(12,4))
plt.plot(tiempo, segmento)
plt.title("Primeros 15 segundos de la señal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [uV]")  # o la unidad de tu registro
plt.grid(True)
plt.show()
#----------------------------------------------------------------------------

#---------------------------------APLICO LOS FILTROS DE BIOSPPY
out = ekg.ecg(signal=datos_canal, sampling_rate=frecuencia_muestreo, show=True)
# out tiene muchas salidas (ver documentacion biosspy)
senial_filt = out['filtered'] #este es el ECG filtrado
rpeaks = out['rpeaks'] #Este es el vector de rpeaks
#-------------------GRAFICO EL ECG FILTRADO


import numpy as np
import matplotlib.pyplot as plt

frecuencia_muestreo = 512  # Hz
duracion = len(senial_filt) / frecuencia_muestreo
tiempo = np.arange(0, duracion, 1/frecuencia_muestreo)

plt.figure(figsize=(15,8))

# --- Gráfico 1: ECG filtrado ---
plt.subplot(2,1,1)
plt.plot(tiempo, senial_filt, color="blue")
plt.title("ECG filtrado")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [uV]")
plt.grid(True)

# --- Gráfico 2: R-peaks (posiciones) ---
plt.subplot(2,1,2)
plt.plot(tiempo, senial_filt, color="lightgray", linewidth=0.8, label="ECG filtrado")
plt.plot(tiempo[rpeaks], senial_filt[rpeaks], 'ro', label="R-peaks")
plt.title("Detección de R-peaks")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [uV]")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



#==========================================Ahora paso el rpeaks a segundos


diferencias = np.diff(rpeaks)  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
intervalos_nn = diferencias/512*1000

import matplotlib.pyplot as plt
import numpy as np

# Suponiendo que intervalos_nn ya está en milisegundos
pares_latidos = np.arange(1, len(intervalos_nn) + 1)

plt.figure(figsize=(12,5))
plt.plot(pares_latidos, intervalos_nn, marker='o', linestyle='-', color='b')

plt.title("Serie de intervalos NN")
plt.xlabel("Par de latidos")
plt.ylabel("Intervalo NN [ms]")
plt.grid(True)
plt.show()

# ============================Calcular los parametros
import scipy.signal as signal
import math 
import biosppy
from biosppy.signals import ecg as ekg
import numpy as np
import matplotlib.pyplot as plt
import pyedflib
import nolds
import neurokit2 as nk
from datetime import datetime, timedelta
import pandas as pd
import pyhrv.tools as tools 
import pyhrv.time_domain as td 
import pyhrv.frequency_domain as fd 
import pyhrv.nonlinear as nl 
import numpy as np
# temporales
parametros_temporales = td.hr_parameters(nni=intervalos_nn)
hr_mean = parametros_temporales['hr_mean']
hr_min = parametros_temporales['hr_min']
hr_max = parametros_temporales['hr_max']
hr_std = parametros_temporales['hr_std']
# frecuenciales
parametros_frecuenciales = fd.welch_psd(intervalos_nn)
aux = td.nn50(nni=intervalos_nn)
p_nn_50 = aux['pnn50']
VLF_power = parametros_frecuenciales['fft_abs'][0] #0 = very low frequency
lf_power = parametros_frecuenciales['fft_abs'][1] #1 = low frequency
hf_power = parametros_frecuenciales['fft_abs'][2] #2 = high frequency
# no lineales
nonlinear_features = nl.poincare(intervalos_nn)
sd1 = nonlinear_features['sd1']
sd2 = nonlinear_features['sd2']
sd_ratio = nonlinear_features['sd_ratio']
elipse_area = nonlinear_features['ellipse_area']
nonlinear_features = nl.sample_entropy(nni = intervalos_nn)
entropia = nonlinear_features['sampen']


# tengo en ecg_crudo el ecg completo de la señal
# de la tabla puedo sacar el resto de variables
# ahora: preprocesar con biosppy 
# luego: alg deteccion picos R
# desp: mostrar el vector de muestras 
# desp: el vector de diferencias 
# (todo lo anterior graficando los primeros 5 latidos)
# siguiente paso: sacar la primer ventana de 180 latidos
# calcular los parametros de esa ventana
# luego explicar el ventaneo y la clasificacion
# sacar para el paciente 1
# y listo el pollo desplumada la gallina me consigo un trabajo mejor