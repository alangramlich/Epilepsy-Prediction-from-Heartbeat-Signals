En estos scripts se ingresa manualmente:
	-cuantas crisis tuvo el paciente
	-la hora de cada crisis
	-como se define la ventana
	-en que canal esta el registro de ecg
	-el nombre del archivo
	-hora de inicio y fin del registro
	-hora de inicio y fin de la/s crisis
	-momento de inicio del periodo preictal (varia para cada dataset)

Con estos datos, el script sabe que ventanas son preictales y cuales no.

Luego, el mismo script:
	-realiza el preprocesamiento de la señal (biosppy)
	-ventanea la señal ()
