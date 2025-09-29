# %% IMPORTACIONES Y DEFINICIONES
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

#defino las funciones
def abrir_y_corroborar(nombre_archivo, canal, freq_muestreo):
    ruta_archivo_edf = nombre_archivo
    archivo_edf = pyedflib.EdfReader(ruta_archivo_edf)
    cantidad_canales = archivo_edf.signals_in_file 
    num_muestras = archivo_edf.getNSamples()[canal]  
    frecuencia_muestreo = 512
    datos_canal = archivo_edf.readSignal(canal)
    eje_temp=np.arange(0, num_muestras/frecuencia_muestreo, 1/frecuencia_muestreo)
    archivo_edf.close()
    
    out = ekg.ecg(signal=datos_canal, sampling_rate=frecuencia_muestreo, show=True)
    senial_filt = out['filtered']
    rpeaks = out['rpeaks']
    heart_rate_ts0 = out['heart_rate_ts'] #LOS TIEMPOS EN SEGUNDOS EN QUE SE CALCULO 
    heart_rate0 = out['heart_rate']       #EL VALOR DE FRECUENCIA CARDIACA
    eje_temp0 = out['ts'] 
    plt.plot(eje_temp[5000:5500], senial_filt[5000:5500])
    return rpeaks, eje_temp, senial_filt

def calcular_segundos(hora_inicio, hora_evento, mismo_dia=True):
    # Convertir las horas a objetos datetime
    inicio = datetime.strptime(hora_inicio, "%H.%M.%S")
    evento = datetime.strptime(hora_evento, "%H.%M.%S")
    
    # Si el evento ocurre al día siguiente, sumamos 1 día al evento
    if not mismo_dia:
        evento += timedelta(days=1)
    
    # Calcular la diferencia en segundos
    segundos_transcurridos = int((evento - inicio).total_seconds())
    
    return segundos_transcurridos

def ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal):
    clasificacion = []
    ventanas_rpeaks = []
    for i in range(0, len(rpeaks), ventana_len - ventana_solap):
        # extraigo la ventana de la señal
        ventana_rpeaks = rpeaks[i:i + ventana_len]    #la ultima ventana puede tener longitud menor
        # Extraer la señal correspondiente en senial_filtrada
        ventana_ecg = ecg[ventana_rpeaks[0]:ventana_rpeaks[-1] + 1]    
        if (len(ventana_rpeaks) == ventana_len):
            ventanas_rpeaks.append(ventana_rpeaks)
            if (i + ventana_len < len(rpeaks) and rpeaks[i] > muestra_inicial_preictal and rpeaks[i+ventana_len] < muestra_final_preictal):
                clasificacion.append('preictal')
            else:
                    clasificacion.append('no_preictal')
    return ventanas_rpeaks, clasificacion

def ventanear_2(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1, muestra_inicial_preictal2, muestra_final_preictal2):
    clasificacion = []
    ventanas_rpeaks = []
    for i in range(0, len(rpeaks), ventana_len - ventana_solap):
        # extraigo la ventana de la señal
        ventana_rpeaks = rpeaks[i:i + ventana_len]    #la ultima ventana puede tener longitud menor
        # Extraer la señal correspondiente en senial_filtrada
        ventana_ecg = ecg[ventana_rpeaks[0]:ventana_rpeaks[-1] + 1]    
        if (len(ventana_rpeaks) == ventana_len):
            ventanas_rpeaks.append(ventana_rpeaks)
            if (i + ventana_len < len(rpeaks) and ((rpeaks[i] > muestra_inicial_preictal1 and rpeaks[i+ventana_len] < muestra_final_preictal1) 
                                                or (rpeaks[i] > muestra_inicial_preictal2 and rpeaks[i+ventana_len] < muestra_final_preictal2))):
                clasificacion.append('preictal')
            else:
                    clasificacion.append('no_preictal')
    return ventanas_rpeaks, clasificacion
def ventanear_3(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1, muestra_inicial_preictal2, muestra_final_preictal2, muestra_inicial_preictal3, muestra_final_preictal3):
    clasificacion = []
    ventanas_rpeaks = []
    for i in range(0, len(rpeaks), ventana_len - ventana_solap):
        # extraigo la ventana de la señal
        ventana_rpeaks = rpeaks[i:i + ventana_len]    #la ultima ventana puede tener longitud menor
        # Extraer la señal correspondiente en senial_filtrada
        ventana_ecg = ecg[ventana_rpeaks[0]:ventana_rpeaks[-1] + 1]    
        if (len(ventana_rpeaks) == ventana_len):
            ventanas_rpeaks.append(ventana_rpeaks)
            if (i + ventana_len < len(rpeaks) and ((rpeaks[i] > muestra_inicial_preictal1 and rpeaks[i+ventana_len] < muestra_final_preictal1) 
                                                or (rpeaks[i] > muestra_inicial_preictal2 and rpeaks[i+ventana_len] < muestra_final_preictal2))
                                                or (rpeaks[i] > muestra_inicial_preictal3 and rpeaks[i+ventana_len] < muestra_final_preictal3)):
                clasificacion.append('preictal')
            else:
                    clasificacion.append('no_preictal')
    return ventanas_rpeaks, clasificacion
def calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion):
    # esta funcion es larga
    # Temporales
    time_features = []
    nni = []
    hr_mean = []
    hr_min = []
    hr_max = []
    hr_std = []
    p_nn_50 = []
    for i in range (len(ventanas_intervalos_nn)):
        #time_features.append(td.time_domain(ventanas_intervalos_nn[i]))
        #nni.append(time_features[i]['nni_diff_mean'])
        nni.append(td.nni_differences_parameters(nni=ventanas_intervalos_nn[i])['nni_diff_mean'])
        aux = td.hr_parameters(nni=ventanas_intervalos_nn[i])
        hr_mean.append(aux['hr_mean'])
        hr_min.append(aux['hr_mean'])
        hr_max.append(aux['hr_max'])
        hr_std.append(aux['hr_std'])
        aux = td.nn50(nni=ventanas_intervalos_nn[i])
        p_nn_50.append(aux['pnn50'])
    # Frecuenciales
    freq_features = []
    VLF_power = []
    lf_power = []
    hf_power = []
    for i in range (len(ventanas_intervalos_nn)):
        freq_features.append(fd.welch_psd(ventanas_intervalos_nn[i]))
        VLF_power.append(freq_features[i]['fft_abs'][0]) #0,1,2 porque asi lo hace la libreria esta
        lf_power.append(freq_features[i]['fft_abs'][1])
        hf_power.append(freq_features[i]['fft_abs'][2])
    # No lineales
    nonlinear_features = []
    sd1 = []
    sd2 = []
    sd_ratio = []
    elipse_area = []
    entropia = []

    for i in range (len(ventanas_intervalos_nn)):
        nonlinear_features.append(nl.poincare(ventanas_intervalos_nn[i]))
        sd1.append(nonlinear_features[i]['sd1'])
        sd2.append(nonlinear_features[i]['sd2'])
        sd_ratio.append(nonlinear_features[i]['sd_ratio'])
        elipse_area.append(nonlinear_features[i]['ellipse_area']) 
        aux = nl.sample_entropy(nni = ventanas_intervalos_nn[i])
        entropia.append(aux['sampen'])
        
    # Armo el DataFrame
    data = {
        'nni_diff_mean': [nni[i] for i in range(len(nni))],
        'hr_mean': [hr_mean[i] for i in range(len(hr_mean))],
        'hr_max': [hr_max[i] for i in range(len(hr_max))],
        'hr_std': [hr_std[i] for i in range(len(hr_std))],
        'pnn50' : [p_nn_50[i] for i in range(len(p_nn_50))],
        'VLF_power': [VLF_power[i] for i in range(len(VLF_power))],
        'LF_power': [lf_power[i] for i in range(len(lf_power))],
        'HF_power': [hf_power[i] for i in range(len(hf_power))],
        'sd1': [sd1[i] for i in range(len(sd1))],
        'sd2': [sd2[i] for i in range(len(sd2))],
        'sd_ratio': [sd_ratio[i] for i in range(len(sd_ratio))],
        'elipse_area': [elipse_area[i] for i in range(len(elipse_area))],
        'SampEn': [entropia[i] for i in range(len(entropia))],
        'clasificacion': [clasificacion[i] for i in range(len(clasificacion))]
    }
    df = pd.DataFrame(data)
    with pd.ExcelWriter(nombre_archivo_excell, mode='a', engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=nombre_paciente, index=False)
def guardar_vectores_en_txt(vector1, vector2, vector3, nombre_archivo_txt):
    # Asegurarse de que los vectores solo contengan enteros o flotantes
    nombre_archivo_ecg = nombre_archivo_txt+'.ecg'
    nombre_archivo_txt=nombre_archivo_txt+'.txt'
    vector1 = [int(i) for i in vector1]
    vector2 = [float(i) for i in vector2]

    # Guardar los elementos de los primeros dos vectores en el archivo de texto
    with open(nombre_archivo_txt, 'w') as archivo:
        # Guardar el primer vector (enteros)
        for elemento in vector1:
            archivo.write(f"{elemento}\n")
        
        # Agregar una línea en blanco
        archivo.write("\n")
        
        # Guardar el segundo vector (flotantes)
        for elemento in vector2:
            archivo.write(f"{elemento}\n")
    
    # Guardar el tercer vector (ECG) en un archivo binario comprimido
    np.save(nombre_archivo_ecg, vector3)
# %% IMPORTACIONES Y DEFINICIONES
import pyhrv.tools as tools
import pyhrv.time_domain as td 
import pyhrv.frequency_domain as fd
import pyhrv.nonlinear as nl 
import numpy as np

LI_periodo_preictal = 15 #limite inferior en minutos
LS_periodo_preictal = 0 #limite superior en minutos
ventana_len = 180
ventana_solap = 60
nombre_archivo_excell = 'vent180_solap60_15minpreictal.xlsx'













# %% PACIENTE 0 SEIZURE 1
frecuencia_muestreo = 512
canal = 33
nombre = 'D:\database Tesis\siena-scalp-eeg-database-1.0.0\siena-scalp-eeg-database-1.0.0\PN00\PN00-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "19.39.33"
hora_inicio_crisis = "19.58.36"
hora_fin_crisis = "19.59.46"
hora_fin_registro = "20.22.58"
nombre_paciente = "P0_SZ1"

segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 

muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []

ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

# diferencias = np.diff(nn_intervals[0])
intervalo_nn=ventanas_intervalos_nn[0]
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)


vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 0 SEIZURE 2
frecuencia_muestreo = 512
canal = 33
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN00/PN00-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "02.18.17"
hora_inicio_crisis = "02.38.37"
hora_fin_crisis = "02.39.31"
hora_fin_registro = "02.56.19"
nombre_paciente = "P0_SZ2"

segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []

ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):
    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista


calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 0 SEIZURE 3 - descartado por no tener sentido
# frecuencia_muestreo = 512
# canal = 33
# nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN00/PN00-3.edf'
# rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
# hora_inicio_registro = "18.15.44"
# hora_inicio_crisis = "18.28.29"
# hora_fin_crisis = "19.29.29"
# hora_fin_registro = "18.57.13"
# nombre_paciente = "P0_SZ3"


# segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
# segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
# segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
# muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
# muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

# muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
# muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 

# muestra_inicial_interictal = 0
# muestra_final_interictal = 0

# clasificacion = []
# ventanas_rpeaks = []

# ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

# ventanas_ms = []
# time_domain_features = []
# freq_features = []
# nonlinear_features = []
# nn_intervals = []
# ventanas_intervalos_nn = []

# for i in range(len(ventanas_rpeaks)):
#     diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
#     ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista


# calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)


# %% PACIENTE 0 SEIZURE 4
frecuencia_muestreo = 512
canal = 33
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN00/PN00-4.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "20.51.43"
hora_inicio_crisis = "21.08.29"
hora_fin_crisis = "21.09.43"
hora_fin_registro = "21.26.25"
nombre_paciente = "P0_SZ4"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 

muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []

ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []




for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 0 SEIZURE 5

frecuencia_muestreo = 512
canal = 33
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN00/PN00-5.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "22.22.04"
hora_inicio_crisis = "22.37.08"
hora_fin_crisis = "22.38.15"
hora_fin_registro = "22.57.27"
nombre_paciente = "P0_SZ5"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 

muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []

ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 1 - dudoso el ECG, lo suspendo 

# frecuencia_muestreo = 512
# canal = 32
# nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN01/PN01-1.edf'
# rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
# hora_inicio_registro = "19.00.44"
# hora_inicio_crisis1 = "21.51.02"
# hora_fin_crisis1 = "21.51.56"
# hora_inicio_crisis2 = "07.53.17"
# hora_fin_crisis2 = "07.53.17"
# hora_fin_registro = "08.29.41"
# nombre_paciente = "P1"


# segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
# segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)

# segundos_inicio_crisis2 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis2, False)
# segundos_fin_crisis2 = calcular_segundos(hora_inicio_registro, hora_fin_crisis2, False)

# segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, False)

# muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
# muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

# muestra_inicial_crisis2 = segundos_inicio_crisis2 * frecuencia_muestreo
# muestra_final_crisis2 = segundos_fin_crisis2 * frecuencia_muestreo

# muestra_inicial_preictal1 = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
# muestra_final_preictal1 = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 

# muestra_inicial_preictal2 = (segundos_inicio_crisis2 - LI_periodo_preictal*60) * frecuencia_muestreo
# muestra_final_preictal2 = (segundos_fin_crisis2 + LS_periodo_preictal*60) * frecuencia_muestreo
# clasificacion = []
# ventanas_rpeaks = []
# ventanas_rpeaks, clasificacion = ventanear_2(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1, muestra_inicial_preictal2, muestra_final_preictal2)

# ventanas_ms = []
# time_domain_features = []
# freq_features = []
# nonlinear_features = []
# nn_intervals = []
# ventanas_intervalos_nn = []

# for i in range(len(ventanas_rpeaks)):

#     diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
#     ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

# calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
# %% PACIENTE 3 SEIZURE 1



frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN03/PN03-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "22.44.37"
hora_inicio_crisis = "09.29.10"
hora_fin_crisis = "09.31.01"
hora_fin_registro = "11.41.34"
nombre_paciente = "P3_SZ1"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, False)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, False)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, False)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 

muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []

ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 3 SEIZURE 2

frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN03/PN03-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "21.31.04"
hora_inicio_crisis = "07.13.05"
hora_fin_crisis = "07.15.18"
hora_fin_registro = "08.47.04"
nombre_paciente = "P3_SZ2"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, False)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, False)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, False)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 

muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)


vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 5 SEIZURE 2
frecuencia_muestreo = 512
canal = 31
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN05/PN05-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "06.46.02"
hora_inicio_crisis = "08.45.25"
hora_fin_crisis = "08.46.00"
hora_fin_registro = "09.19.47"
nombre_paciente = "P5_SZ2"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 5 SEIZURE 3

frecuencia_muestreo = 512
canal = 31
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN05/PN05-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "06.01.23"
hora_inicio_crisis = "07.55.19"
hora_fin_crisis = "07.55.49"
hora_fin_registro = "08.06.57"
nombre_paciente = "P5_SZ3"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 5 SEIZURE 4
ventana_len = 180
ventana_solap = 60


frecuencia_muestreo = 512
canal = 31
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN05/PN05-4.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "06.38.35"
hora_inicio_crisis = "07.38.43"
hora_fin_crisis = "07.39.22"
hora_fin_registro = "08.00.23"
nombre_paciente = "P5_SZ4"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista

calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 6 SEIZURE 1

frecuencia_muestreo = 512
canal = 31
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN06/PN06-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "04.21.22"
hora_inicio_crisis = "05.54.25"
hora_fin_crisis = "05.55.29"
hora_fin_registro = "07.01.37"
nombre_paciente = "P6_SZ1"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 6 SEIZURE 2



frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN06/PN06-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "21.11.29"
hora_inicio_crisis = "23.39.09"
hora_fin_crisis = "23.40.18"
hora_fin_registro = "00.41.49"
nombre_paciente = "P6_SZ2"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, False)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 6 SEIZURE 3



frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN06/PN06-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "06.25.51"
hora_inicio_crisis = "08.10.26"
hora_fin_crisis = "08.11.08"
hora_fin_registro = "08.41.51"
nombre_paciente = "P6_SZ3"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 6 SEIZURE 4

frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN06/PN06-4.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "11.16.09"
hora_inicio_crisis = "12.55.08"
hora_fin_crisis = "12.56.11"
hora_fin_registro = "13.12.54"
nombre_paciente = "P6_SZ4"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 6 SEIZURE 5

frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN06/PN06-5.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "13.24.41"
hora_inicio_crisis = "14.44.24"
hora_fin_crisis = "14.45.08"
hora_fin_registro = "15.04.43"
nombre_paciente = "P6_SZ5"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 7



frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN07/PN07-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "23.18.10"
hora_inicio_crisis = "05.25.49"
hora_fin_crisis = "05.26.51"
hora_fin_registro = "08.01.58"
nombre_paciente = "P7"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, False)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, False)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, False)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 9 SEIZURE 1

frecuencia_muestreo = 512
canal = 31
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN09/PN09-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "14.08.54"
hora_inicio_crisis = "16.09.43"
hora_fin_crisis = "16.11.03"
hora_fin_registro = "16.26.07"
nombre_paciente = "P9_SZ1"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo


muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 9 SEIZURE 2

frecuencia_muestreo = 512
canal = 31
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN09/PN09-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "15.02.09"
hora_inicio_crisis = "17.00.56"
hora_fin_crisis = "17.01.55"
hora_fin_registro = "17.21.02"
nombre_paciente = "P9_SZ2"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo
muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 9 SEIZURE 3

frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN09/PN09-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "14.20.23"
hora_inicio_crisis = "16.20.44"
hora_fin_crisis = "16.21.48"
hora_fin_registro = "16.34.28"
nombre_paciente = "P9_SZ3"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo
muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 10 SEIZURE 1

frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN10/PN10-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "05.40.05"
hora_inicio_crisis = "07.45.50"
hora_fin_crisis = "07.46.59"
hora_fin_registro = "08.26.07"
nombre_paciente = "P10_SZ1"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 10 SEIZURE 2

frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN10/PN10-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "09.30.15"
hora_inicio_crisis = "11.40.13"
hora_fin_crisis = "11.41.04"
hora_fin_registro = "11.52.29"
nombre_paciente = "P10_SZ2"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 10 SEIZURE 3

frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN10/PN10-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "13.33.18"
hora_inicio_crisis = "15.43.53"
hora_fin_crisis = "15.45.02"
hora_fin_registro = "15.58.31"
nombre_paciente = "P10_SZ3"


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 10 SEIZURE 4, 5, 6
ventana_len = 180
ventana_solap = 60


frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN10/PN10-4.5.6.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "12.11.21"

hora_inicio_crisis1 = "12.49.50"
hora_fin_crisis1 = "12.49.55"

hora_inicio_crisis2 = "14.00.25"
hora_fin_crisis2 = "14.00.44"

hora_inicio_crisis3 = "15.18.26"
hora_fin_crisis3 = "15.19.23"

hora_fin_registro = "16.48.49"
nombre_paciente = "P10_SZ456"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal1 = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal1 = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 



segundos_inicio_crisis2 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis2, True)
segundos_fin_crisis2 = calcular_segundos(hora_inicio_registro, hora_fin_crisis2, True)
muestra_inicial_crisis2 = segundos_inicio_crisis2 * frecuencia_muestreo
muestra_final_crisis2 = segundos_fin_crisis2 * frecuencia_muestreo
muestra_inicial_preictal2 = (segundos_inicio_crisis2 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal2 = (segundos_fin_crisis2+LS_periodo_preictal*60) * frecuencia_muestreo 



segundos_inicio_crisis3 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis3, True)
segundos_fin_crisis3 = calcular_segundos(hora_inicio_registro, hora_fin_crisis3, True)
muestra_inicial_crisis3 = segundos_inicio_crisis3 * frecuencia_muestreo
muestra_final_crisis3 = segundos_fin_crisis3 * frecuencia_muestreo
muestra_inicial_preictal3 = (segundos_inicio_crisis3 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal3 = (segundos_fin_crisis3+LS_periodo_preictal*60) * frecuencia_muestreo

ventanas_rpeaks, clasificacion = ventanear_3(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1, muestra_inicial_preictal2, muestra_final_preictal2,muestra_inicial_preictal3, muestra_final_preictal3)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1,muestra_inicial_crisis2, muestra_final_crisis2,muestra_inicial_crisis3, muestra_final_crisis3]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 10 SEIZURE 7, 8, 9

frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN10/PN10-7.8.9.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "16.49.25"

hora_inicio_crisis1 = "17.35.13"
hora_fin_crisis1 = "17.36.01"

hora_inicio_crisis2 = "18.20.24"
hora_fin_crisis2 = "18.20.42"

hora_inicio_crisis3 = "20.24.48"
hora_fin_crisis3 = "20.25.03"

hora_fin_registro = "20.58.40"
nombre_paciente = "P10_SZ789"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal1 = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal1 = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 



segundos_inicio_crisis2 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis2, True)
segundos_fin_crisis2 = calcular_segundos(hora_inicio_registro, hora_fin_crisis2, True)
muestra_inicial_crisis2 = segundos_inicio_crisis2 * frecuencia_muestreo
muestra_final_crisis2 = segundos_fin_crisis2 * frecuencia_muestreo
muestra_inicial_preictal2 = (segundos_inicio_crisis2 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal2 = (segundos_fin_crisis2+LS_periodo_preictal*60) * frecuencia_muestreo 



segundos_inicio_crisis3 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis3, True)
segundos_fin_crisis3 = calcular_segundos(hora_inicio_registro, hora_fin_crisis3, True)
muestra_inicial_crisis3 = segundos_inicio_crisis3 * frecuencia_muestreo
muestra_final_crisis3 = segundos_fin_crisis3 * frecuencia_muestreo
muestra_inicial_preictal3 = (segundos_inicio_crisis3 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal3 = (segundos_fin_crisis3+LS_periodo_preictal*60) * frecuencia_muestreo

ventanas_rpeaks, clasificacion = ventanear_3(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1, muestra_inicial_preictal2, muestra_final_preictal2,muestra_inicial_preictal3, muestra_final_preictal3)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1,muestra_inicial_crisis2, muestra_final_crisis2,muestra_inicial_crisis3, muestra_final_crisis3]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 10 SEIZURE 10

frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN10/PN10-10.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "08.45.22"

hora_inicio_crisis1 = "10.58.19"
hora_fin_crisis1 = "10.58.33"


hora_fin_registro = "11.08.07"
nombre_paciente = "P10_SZ10"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)


segundos_inicio_crisis = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)
muestra_inicial_crisis = segundos_inicio_crisis * frecuencia_muestreo
muestra_final_crisis = segundos_fin_crisis * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis, muestra_final_crisis]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 11 



frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN11/PN11-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "11.31.25"

hora_inicio_crisis1 = "13.37.19"
hora_fin_crisis1 = "13.38.14"


hora_fin_registro = "13.56.02"
nombre_paciente = "P11"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo
muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 12 SEIZURE 1, 2



frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN12/PN12-1.2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "15.51.31"

hora_inicio_crisis1 = "16.13.23"
hora_fin_crisis1 = "16.14.26"

hora_inicio_crisis2 = "18.31.01"
hora_fin_crisis2 = "18.32.09"

hora_fin_registro = "18.34.25"
nombre_paciente = "P12_SZ1"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo
muestra_inicial_preictal1 = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal1 = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo


segundos_inicio_crisis2 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis2, True)
segundos_fin_crisis2 = calcular_segundos(hora_inicio_registro, hora_fin_crisis2, True)
muestra_inicial_crisis2 = segundos_inicio_crisis2 * frecuencia_muestreo
muestra_final_crisis2 = segundos_fin_crisis2 * frecuencia_muestreo
muestra_inicial_preictal2 = (segundos_inicio_crisis2 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal2 = (segundos_fin_crisis2+LS_periodo_preictal*60) * frecuencia_muestreo

ventanas_rpeaks, clasificacion = ventanear_2(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal1, muestra_final_preictal1, muestra_inicial_preictal2, muestra_final_preictal2)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1, muestra_inicial_crisis2, muestra_inicial_crisis3]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 12 SEIZURE 3


frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN12/PN12-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "08.42.35"

hora_inicio_crisis1 = "08.55.27"
hora_fin_crisis1 = "08.57.03"

hora_fin_registro = "09.15.44"
nombre_paciente = "P12_SZ3"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)

vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)
# %% PACIENTE 12 SEIZURE 4


frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN12/PN12-4.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "15.59.19"

hora_inicio_crisis1 = "18.42.51"
hora_fin_crisis1 = "18.43.54"

hora_fin_registro = "18.49.13"
nombre_paciente = "P12_SZ4"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo
muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 13 SEIZURE 1


frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN13/PN13-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "08.24.28"

hora_inicio_crisis1 = "10.22.10"
hora_fin_crisis1 = "10.22.58"

hora_fin_registro = "11.00.23"
nombre_paciente = "P13_SZ1"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 13 SEIZURE 2


frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN13/PN13-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "06.55.02"

hora_inicio_crisis1 = "08.55.51"
hora_fin_crisis1 = "08.56.56"

hora_fin_registro = "09.30.14"
nombre_paciente = "P13_SZ2"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 13 SEIZURE 3


frecuencia_muestreo = 512
canal = 32
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN13/PN13-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "12.00.01"

hora_inicio_crisis1 = "14.05.54"
hora_fin_crisis1 = "14.08.25"

hora_fin_registro = "15.28.48"
nombre_paciente = "P13_SZ3"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 14 SEIZURE 1


frecuencia_muestreo = 512
canal = 29
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN14/PN14-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "11.44.58"

hora_inicio_crisis1 = "13.46.00"
hora_fin_crisis1 = "13.46.27"

hora_fin_registro = "13.56.06"
nombre_paciente = "P14_SZ1"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 14 SEIZURE 2


frecuencia_muestreo = 512
canal = 29
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN14/PN14-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "15.50.13"

hora_inicio_crisis1 = "17.54.52"
hora_fin_crisis1 = "17.55.04"

hora_fin_registro = "18.22.58"
nombre_paciente = "P14_SZ2"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 14 SEIZURE 3


frecuencia_muestreo = 512
canal = 29
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN14/PN14-3.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "16.17.45"

hora_inicio_crisis1 = "21.10.05"
hora_fin_crisis1 = "21.10.46"

hora_fin_registro = "06.57.40"
nombre_paciente = "P14_SZ3"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, False)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 14 SEIZURE 4


frecuencia_muestreo = 512
canal = 29
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN14/PN14-4.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "14.18.30"

hora_inicio_crisis1 = "15.49.33"
hora_fin_crisis1 = "15.50.56"

hora_fin_registro = "18.22.32"
nombre_paciente = "P14_SZ4"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 16 SEIZURE 1


frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN16/PN16-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "20.45.21"

hora_inicio_crisis1 = "22.45.05"
hora_fin_crisis1 = "22.47.08"

hora_fin_registro = "23.03.11"
nombre_paciente = "P16_SZ1"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 16 SEIZURE 2


frecuencia_muestreo = 512
canal = 28
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN16/PN16-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "00.53.55"

hora_inicio_crisis1 = "03.16.49"
hora_fin_crisis1 = "03.18.36"

hora_fin_registro = "03.28.52"
nombre_paciente = "P16_SZ2"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 17 SEIZURE 1


frecuencia_muestreo = 512
canal = 29
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN17/PN17-1.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "20.14.28"

hora_inicio_crisis1 = "22.34.48"
hora_fin_crisis1 = "22.35.58"

hora_fin_registro = "22.49.05"
nombre_paciente = "P17_SZ1"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

# %% PACIENTE 17 SEIZURE 2


frecuencia_muestreo = 512
canal = 29
nombre = '../siena-scalp-eeg-database-1.0.0/siena-scalp-eeg-database-1.0.0/PN17/PN17-2.edf'
rpeaks, eje_temp, ecg = abrir_y_corroborar(nombre, canal, frecuencia_muestreo)
hora_inicio_registro = "13.52.18"

hora_inicio_crisis1 = "16.01.09"
hora_fin_crisis1 = "16.02.32"

hora_fin_registro = "16.25.10"
nombre_paciente = "P17_SZ2"

segundos_fin_registro = calcular_segundos(hora_inicio_registro, hora_fin_registro, True)

segundos_inicio_crisis1 = calcular_segundos(hora_inicio_registro, hora_inicio_crisis1, True)
segundos_fin_crisis1 = calcular_segundos(hora_inicio_registro, hora_fin_crisis1, True)
muestra_inicial_crisis1 = segundos_inicio_crisis1 * frecuencia_muestreo
muestra_final_crisis1 = segundos_fin_crisis1 * frecuencia_muestreo

muestra_inicial_preictal = (segundos_inicio_crisis1 - LI_periodo_preictal*60) * frecuencia_muestreo
muestra_final_preictal = (segundos_fin_crisis1+LS_periodo_preictal*60) * frecuencia_muestreo 


muestra_inicial_interictal = 0
muestra_final_interictal = 0
clasificacion = []
ventanas_rpeaks = []


ventanas_rpeaks, clasificacion = ventanear_1(rpeaks, ecg, ventana_len, ventana_solap, muestra_inicial_preictal, muestra_final_preictal)

ventanas_ms = []
time_domain_features = []
freq_features = []
nonlinear_features = []
nn_intervals = []
ventanas_intervalos_nn = []

for i in range(len(ventanas_rpeaks)):

    diferencias = np.diff(ventanas_rpeaks[i])  # Esto calcula nn_intervals[i][j] - nn_intervals[i][j+1]
    ventanas_intervalos_nn.append(diferencias/frecuencia_muestreo*1000)  # Agregar las diferencias a la lista
calculo_parametros_y_guardar(nombre_archivo_excell, nombre_paciente, ventanas_intervalos_nn, clasificacion)
vector_crisis = [muestra_inicial_crisis1, muestra_final_crisis1]
guardar_vectores_en_txt(vector_crisis, rpeaks, ecg, nombre_paciente)

