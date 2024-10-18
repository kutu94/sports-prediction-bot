Sports Prediction Bot
Este repositorio contiene un bot de predicción de resultados de fútbol, que ha sido desarrollado en tres versiones distintas. Cada versión ofrece una evolución en la funcionalidad, desde una versión básica de predicción de resultados hasta una aplicación web interactiva usando Streamlit.

Descripción general
Este bot predice el resultado de partidos de fútbol, así como estadísticas adicionales como el número de corners, faltas y tarjetas amarillas/rojas. El bot trabaja con datos históricos de la Primera División y Segunda División de España.

Versiones incluidas:
V1 - Versión básica: Predice resultados y estadísticas básicas como corners, faltas y tarjetas.
V2 - Versión mejorada: Incluye predicciones adicionales como goles por mitad y un mayor grado de precisión utilizando características adicionales.
V3 - Versión con interfaz web: Incorpora una aplicación web interactiva con Streamlit para predecir los partidos desde una interfaz gráfica.

Estructura del repositorio
/data: Archivos CSV con datos históricos de la Primera y Segunda División.
/v1_basic: Primera versión del bot, con predicciones básicas.
/v2_improved: Segunda versión del bot, con predicciones más avanzadas.
/v3_app: Versión con una aplicación web interactiva utilizando Streamlit.

Requisitos
Asegúrate de tener instaladas las siguientes dependencias antes de ejecutar el código:

pip install -r requirements.txt


Dependencias principales:

pandas
scikit-learn
xgboost
streamlit (para la versión 3)


Cada versión tiene su propio README con instrucciones más detalladas sobre cómo ejecutar el código.

**TENER EN CUENTA QUE HAY QUE ACTUALIZAR LOS ARCHIVOS .CSV DE LA CARPETA 'DATA' , PARA OBTENER UN ANALISIS MAS ACERTADO**