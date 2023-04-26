# Clasificación (SVC, ET, RF)
## UACH - INGENIERÍA EN CIENCIAS DE LA COMPUTACIÓN 
### José Carlos Chaparro Morales - 329613
### Omar Alonso Escápita Chacón - 338886

Se desarrolló un modelo de clasificación de noticias utilizando SVC, Random Forest y Extra Trees, inicialmente al hacer el análisis de los datos se encontró que el dataset seleccionado estába muy desbalanceado, además de contener demasiadas clases. Para corregir un poco este defecto se optó por combinar las clases similares o repetidas, así como eliminar los valores nulos en el campo de la descripción y reducir cada categoría proporcionalmente. Una vez hecho estos cambios se encontró una mejora en los valores de las metricas pasando de alrededor de 0.2 en las primeras pruebas a 0.6 al final.

- [Enlace al deploy en Streamlit](https://josecchaparro-classification-project-streamlit-intento-d6fg26.streamlit.app/).

El proyecto consta de los siguientes archivos y carpetas:

- **SVM_Proyecto.ipynb**: un cuaderno de Jupyter que contiene el código para la exploración de datos, entrenamiento de los modelos y evaluación de su rendimiento. Se puede acceder al cuaderno en una versión de colab con el siguiente.

- **dataset**: el conjunto de datos de noticias y sus categorias.

- **forest_clf_grid_search.pkl**: un archivo pickle que contiene el modelo  RndomForest de aprendizaje automático entrenado.

- **svc.pkl**: un archivo pickle que contiene el modelo SVC de aprendizaje automático entrenado.

- **streamlit_intento.py**: un archivo de Python con el código utilizado para el deploy del modelo. El otro archivo de streamlit es la primer iteración del programa. Este es una versión optimizada

- **README.md**: un archivo de texto que contiene información sobre el proyecto.
