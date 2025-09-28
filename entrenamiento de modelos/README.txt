📄 Documentación – Script de Entrenamiento Masivo de Modelos Random Forest
Archivo: Entrenamiento_definitivo_V1.py
________________


✅ ¿Qué hace este script?
Este script automatiza el proceso de análisis de múltiples archivos Excel que contienen hojas de pacientes con datos de HRV (variabilidad de la frecuencia cardíaca) y clasificaciones asociadas (preictal / no_preictal). Su propósito es entrenar modelos de clasificación utilizando Random Forest y registrar los resultados.
________________


🧭 ¿Cómo funciona?
1. Ventana emergente de selección de carpeta
 Al ejecutar el script, se abrirá una ventana para que el usuario seleccione la carpeta raíz donde están guardados los archivos Excel a procesar (incluye subcarpetas).

2. Exploración de archivos
 El script buscará todos los archivos .xlsx y .xls dentro de esa carpeta y subcarpetas.

3. Procesamiento de cada archivo Excel
 Para cada archivo encontrado:

   * Se leen todas las hojas del archivo.

   * Se seleccionan únicamente aquellas hojas que contengan:

      * Las características: nni_diff_mean, CSI, hr_std, HF_power, SampEn, pnn50

      * Y una columna llamada clasificacion con al menos 2 instancias de cada clase (preictal y no_preictal).

         * Se entrena un modelo Random Forest variando:

            * El porcentaje de entrenamiento (70%, 75%, 80%, 85%, 90%)

            * La profundidad máxima del árbol (5, 10, 15, 20, None)

               4. Resultados generados
 Para cada archivo analizado se generan:

                  * Un archivo Excel (.xlsx) con los resultados de métricas:

                     * Accuracy, Precision, Recall, F1 para la clase preictal.

                        * Dos gráficos (.png):

                           * Métricas promedio por profundidad del árbol.

                           * Evolución de métricas según porcentaje de entrenamiento.

                              5. Carpeta de salida
 Todos los resultados se guardan en una carpeta llamada Resultados que se crea automáticamente al lado del script .py.

________________


📁 Ejemplo de archivos generados
Si el script analiza un archivo llamado dataset3.xlsx, se crearán:
CopyEdit
Resultados/
├── resultados_dataset3.xlsx
├── grafico_max_depth_dataset3.png
└── grafico_train_split_dataset3.png


________________


⚠️ Requisitos
                                 * Python 3.7+

                                 * Bibliotecas:

                                    * pandas

                                    * scikit-learn

                                    * matplotlib

                                    * seaborn

                                    * openpyxl

                                    * tkinter (incluido en la mayoría de las instalaciones de Python)

Para instalar dependencias:
bash
CopyEdit
pip install pandas scikit-learn matplotlib seaborn openpyxl


________________


📝 Notas
                                       * Las hojas sin datos suficientes serán ignoradas automáticamente.

                                       * El script es seguro ante errores y continúa con el resto de los archivos si alguno falla.

                                       * El modelo se entrena usando estratificación para asegurar balance de clases.