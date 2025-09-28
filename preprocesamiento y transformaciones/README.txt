Estos scripts chequean:
	-Cuantas crisis tuvo el paciente
	-la hora
	-preprocesa la señal de ECG
	-obtiene el tacograma (transforma el ECG a señal NNms)
	-ventanea la señal de NNms en N ventanas, con longitud de de Ventana y solapamiento. A cada Ventana le añade "clasificacion", que puede ser preictal o no preictal.
	-calcula los parametros de HRV para cada vendana. Mantiene la clasificacion "preictal" o "no_preictal".
	-salida final del script: un excel con los parametros de HRV y la clasificacion, para entrenar un modelo de Machine Learning.