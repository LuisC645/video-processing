Este repositorio contiene un conjunto de herramientas y scripts para el preprocesamiento de videos utilizando ángulos de Rodrigues y la biblioteca MediaPipe. 
El objetivo es facilitar el análisis y la manipulación de videos para una extracción de caragterísticas más generalizada

**Características**

**Detección de Puntos de Referencia (Landmarks)**: Utiliza MediaPipe para detectar y extraer puntos de referencia de los cuerpos en los videos.
**Cálculo de Ángulos de Rodrigues**: Implementación de cálculos de ángulos de Rodrigues para la representación y manipulación de rotaciones en 3D.
**Recorte de Rostro**: Extrae y recorta solo la región del rostro en el video.
**Corrección de Cámara**: Ajusta la orientación de la cámara para mantener la cara en posición horizontal, estandarizando así la mayor cantidad de características del rostro.
**Preprocesamiento de Datos**: Filtrado y normalización de datos de video para prepararlos para análisis posteriores.

**Requisitos**
- Python 3.7+
- MediaPipe
- OpenCV
- NumPy
