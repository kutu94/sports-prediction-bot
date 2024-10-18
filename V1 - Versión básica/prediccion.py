import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 1. Cargar y combinar los datos
# Lista para almacenar DataFrames de cada temporada
dataframes = []

# Lista de archivos CSV para Primera División

csv_files_primera = ['../data/Primera2018.csv', '../data/Primera2019.csv', '../data/Primera2020.csv',
                     '../data/Primera2021.csv', '../data/Primera2022.csv', '../data/Primera2023.csv', '../data/Primera2024.csv']

for file in csv_files_primera:
    df = pd.read_csv(file)
    df['Division'] = 'Primera'
    dataframes.append(df)

# Lista de archivos CSV para Segunda División
csv_files_segunda = ['../data/Segunda2018.csv', '../data/Segunda2019.csv', '../data/Segunda2020.csv',
                     '../data/Segunda2021.csv', '../data/Segunda2022.csv', '../data/Segunda2023.csv', '../data/Segunda2024.csv']


for file in csv_files_segunda:
    df = pd.read_csv(file)
    df['Division'] = 'Segunda'
    dataframes.append(df)

# Combinar todos los DataFrames en uno solo
data_total = pd.concat(dataframes, ignore_index=True)

# 2. Estandarizar nombres de equipos
team_name_mapping = {
    'Ath Bilbao': 'Athletic Club',
    'Ath Madrid': 'Atlético Madrid',
    'Espanol': 'Espanyol',
    'Sociedad': 'Real Sociedad',
    'Sp Gijon': 'Sporting Gijón',
    'Vallecano': 'Rayo Vallecano',
    'La Coruna': 'Deportivo La Coruña',
    'Alaves': 'Alavés',
    'Cordoba': 'Córdoba',
    'Leganes': 'Leganés',
    'Malaga': 'Málaga',
    'Alcorcon': 'Alcorcón',
    'Cadiz': 'Cádiz',
    'Santander': 'Racing Santander',
    # Añade más mapeos si encuentras otras inconsistencias
}

# Limpiar también el input del usuario

def standardize_team_name(name):
    return name.strip().lower()
    return team_name_mapping.get(name, name)

# Aplicar la limpieza y estandarización de los nombres de equipos en el dataset
data_total['HomeTeam'] = data_total['HomeTeam'].apply(standardize_team_name)
data_total['AwayTeam'] = data_total['AwayTeam'].apply(standardize_team_name)

# 3. Convertir la columna 'Date' al formato datetime
data_total['Date'] = pd.to_datetime(data_total['Date'], format='%d/%m/%Y', errors='coerce')

# Verificar si hay fechas nulas
if data_total['Date'].isnull().sum() > 0:
    print("Advertencia: Hay fechas nulas en el dataset. Se eliminarán esas filas.")
    data_total.dropna(subset=['Date'], inplace=True)

# 4. Seleccionar las columnas necesarias
required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                    'HY', 'AY', 'HR', 'AR', 'HF', 'AF', 'HC', 'AC']

# Verificar que las columnas existen
missing_columns = [col for col in required_columns if col not in data_total.columns]
if missing_columns:
    print(f"Las siguientes columnas faltan en el DataFrame: {missing_columns}")
    # Si faltan columnas, no se puede continuar
    exit()

# Seleccionar las columnas
data = data_total[required_columns].copy()

# 5. Manejar valores nulos en las columnas seleccionadas
# Eliminar filas con valores nulos en las columnas necesarias
data.dropna(subset=required_columns, inplace=True)

# Asegurarse de que los tipos de datos son correctos
numerical_columns = ['FTHG', 'FTAG', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF', 'HC', 'AC']
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Verificar si hay valores nulos después de la conversión
if data[numerical_columns].isnull().sum().sum() > 0:
    print("Advertencia: Hay valores nulos en columnas numéricas después de la conversión. Se eliminarán esas filas.")
    data.dropna(subset=numerical_columns, inplace=True)

# 6. Añadir características adicionales
# Forma reciente (últimos 5 partidos)

def get_recent_form(team, date, data, n_matches=5):
    past_matches = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)]
    past_matches = past_matches.sort_values(by='Date', ascending=False).head(n_matches)
    results = past_matches['FTR']
    points = 0
    for result, home_team, away_team in zip(results, past_matches['HomeTeam'], past_matches['AwayTeam']):
        if (team == home_team and result == 'H') or (team == away_team and result == 'A'):
            points += 3
        elif result == 'D':
            points += 1
    return points / (n_matches * 3)  # Normalizado entre 0 y 1

# 7. Definir la función para obtener estadísticas del equipo
def get_team_stats(team, date, data):
    past_matches = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)]
    total_matches = len(past_matches)
    if total_matches == 0:
        return pd.Series({
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'avg_yellow_cards': 0,
            'avg_red_cards': 0,
            'avg_fouls': 0,
            'avg_corners': 0,
            'recent_form': 0
        })

    # Condiciones para casa y fuera
    home_matches = past_matches[past_matches['HomeTeam'] == team]
    away_matches = past_matches[past_matches['AwayTeam'] == team]

    # Calcular estadísticas usando operaciones vectorizadas
    goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
    goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
    yellow_cards = home_matches['HY'].sum() + away_matches['AY'].sum()
    red_cards = home_matches['HR'].sum() + away_matches['AR'].sum()
    fouls = home_matches['HF'].sum() + away_matches['AF'].sum()
    corners = home_matches['HC'].sum() + away_matches['AC'].sum()
    recent_form = get_recent_form(team, date, data, n_matches=5)

    return pd.Series({
        'avg_goals_scored': goals_scored / total_matches,
        'avg_goals_conceded': goals_conceded / total_matches,
        'avg_yellow_cards': yellow_cards / total_matches,
        'avg_red_cards': red_cards / total_matches,
        'avg_fouls': fouls / total_matches,
        'avg_corners': corners / total_matches,
        'recent_form': recent_form
    })

# 8. Crear el DataFrame de características
features = []

print("Generando características...")
for index, row in data.iterrows():
    date = row['Date']
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']

    home_stats = get_team_stats(home_team, date, data)
    away_stats = get_team_stats(away_team, date, data)

    feature_row = {
        'home_avg_goals_scored': home_stats['avg_goals_scored'],
        'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
        'home_avg_yellow_cards': home_stats['avg_yellow_cards'],
        'home_avg_red_cards': home_stats['avg_red_cards'],
        'home_avg_fouls': home_stats['avg_fouls'],
        'home_avg_corners': home_stats['avg_corners'],
        'home_recent_form': home_stats['recent_form'],
        'away_avg_goals_scored': away_stats['avg_goals_scored'],
        'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
        'away_avg_yellow_cards': away_stats['avg_yellow_cards'],
        'away_avg_red_cards': away_stats['avg_red_cards'],
        'away_avg_fouls': away_stats['avg_fouls'],
        'away_avg_corners': away_stats['avg_corners'],
        'away_recent_form': away_stats['recent_form'],
        'FTR': row['FTR'],
        'home_corners': row['HC'],
        'away_corners': row['AC'],
        'home_fouls': row['HF'],
        'away_fouls': row['AF'],
        'home_yellow_cards': row['HY'],
        'away_yellow_cards': row['AY'],
    }

    features.append(feature_row)

features_df = pd.DataFrame(features)

# 9. Preparar los datos para el modelo
le_result = LabelEncoder()
features_df['result_code'] = le_result.fit_transform(features_df['FTR'])

# Variables predictoras y objetivo para clasificación
X_class = features_df.drop(['FTR', 'result_code', 'home_corners', 'away_corners', 'home_fouls', 'away_fouls',
                            'home_yellow_cards', 'away_yellow_cards'], axis=1)
y_class = features_df['result_code']

# Variables predictoras y objetivo para regresión (corners, faltas, tarjetas)
X_reg = X_class.copy()
y_corners_home = features_df['home_corners']
y_corners_away = features_df['away_corners']
y_fouls_home = features_df['home_fouls']
y_fouls_away = features_df['away_fouls']
y_cards_home = features_df['home_yellow_cards']
y_cards_away = features_df['away_yellow_cards']

# Manejar valores nulos en X
if X_class.isnull().sum().sum() > 0:
    print("Advertencia: Hay valores nulos en las características. Se eliminarán esas filas.")
    X_class = X_class.dropna()
    y_class = y_class[X_class.index]
    X_reg = X_reg.loc[X_class.index]
    y_corners_home = y_corners_home[X_class.index]
    y_corners_away = y_corners_away[X_class.index]
    y_fouls_home = y_fouls_home[X_class.index]
    y_fouls_away = y_fouls_away[X_class.index]
    y_cards_home = y_cards_home[X_class.index]
    y_cards_away = y_cards_away[X_class.index]

# 10. Manejar el desequilibrio de clases con SMOTE
smote = SMOTE(random_state=42)
X_class_balanced, y_class_balanced = smote.fit_resample(X_class, y_class)

# 11. Entrenar el modelo de clasificación
model_class = RandomForestClassifier(n_estimators=100, random_state=42)

# Validación cruzada
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_index, test_index in tscv.split(X_class_balanced):
    X_train_cv, X_test_cv = X_class_balanced.iloc[train_index], X_class_balanced.iloc[test_index]
    y_train_cv, y_test_cv = y_class_balanced.iloc[train_index], y_class_balanced.iloc[test_index]
    model_class.fit(X_train_cv, y_train_cv)
    y_pred_cv = model_class.predict(X_test_cv)
    score = accuracy_score(y_test_cv, y_pred_cv)
    scores.append(score)
print(f'Scores de validación cruzada (clasificación): {scores}')
print(f'Precisión promedio (clasificación): {np.mean(scores):.2f}')

# Entrenar el modelo en todos los datos de entrenamiento
model_class.fit(X_class_balanced, y_class_balanced)

# 12. Entrenar modelos de regresión para predicción de corners, faltas y tarjetas
# Escalado de características
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Modelos de regresión
model_corners_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_corners_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_fouls_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_fouls_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_cards_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_cards_away = RandomForestRegressor(n_estimators=100, random_state=42)

# Entrenar los modelos
model_corners_home.fit(X_reg_scaled, y_corners_home)
model_corners_away.fit(X_reg_scaled, y_corners_away)
model_fouls_home.fit(X_reg_scaled, y_fouls_home)
model_fouls_away.fit(X_reg_scaled, y_fouls_away)
model_cards_home.fit(X_reg_scaled, y_cards_home)
model_cards_away.fit(X_reg_scaled, y_cards_away)

# 13. Función para predecir el resultado y estadísticas de un partido
def predecir_partido(equipo_local, equipo_visitante, fecha_partido, modelo_class, data):
    # Convertir fecha_partido a datetime si es una cadena
    if isinstance(fecha_partido, str):
        fecha_partido = pd.to_datetime(fecha_partido, format='%d/%m/%Y', errors='coerce')
        if pd.isnull(fecha_partido):
            print("Error: La fecha del partido no tiene el formato correcto (dd/mm/yyyy).")
            return None

    # Verificar que los equipos existen
    equipos_disponibles = set(data['HomeTeam']).union(set(data['AwayTeam']))
    if equipo_local not in equipos_disponibles or equipo_visitante not in equipos_disponibles:
        print("Error: Uno o ambos equipos no están en el dataset.")
        return None

    local_stats = get_team_stats(equipo_local, fecha_partido, data)
    visitante_stats = get_team_stats(equipo_visitante, fecha_partido, data)

    X_nuevo = pd.DataFrame([{
        'home_avg_goals_scored': local_stats['avg_goals_scored'],
        'home_avg_goals_conceded': local_stats['avg_goals_conceded'],
        'home_avg_yellow_cards': local_stats['avg_yellow_cards'],
        'home_avg_red_cards': local_stats['avg_red_cards'],
        'home_avg_fouls': local_stats['avg_fouls'],
        'home_avg_corners': local_stats['avg_corners'],
        'home_recent_form': local_stats['recent_form'],
        'away_avg_goals_scored': visitante_stats['avg_goals_scored'],
        'away_avg_goals_conceded': visitante_stats['avg_goals_conceded'],
        'away_avg_yellow_cards': visitante_stats['avg_yellow_cards'],
        'away_avg_red_cards': visitante_stats['avg_red_cards'],
        'away_avg_fouls': visitante_stats['avg_fouls'],
        'away_avg_corners': visitante_stats['avg_corners'],
        'away_recent_form': visitante_stats['recent_form'],
    }])

    if X_nuevo.isnull().sum().sum() > 0:
        print("No hay suficientes datos históricos para realizar la predicción.")
        return None

    # Predicción de resultado
    resultado_cod = modelo_class.predict(X_nuevo)[0]
    resultado = le_result.inverse_transform([resultado_cod])[0]

    if resultado == 'H':
        pronostico = f'Victoria para {equipo_local}'
    elif resultado == 'A':
        pronostico = f'Victoria para {equipo_visitante}'
    else:
        pronostico = 'Empate'

    # Predicción de corners, faltas y tarjetas
    X_nuevo_reg = scaler.transform(X_nuevo)
    corners_home = model_corners_home.predict(X_nuevo_reg)[0]
    corners_away = model_corners_away.predict(X_nuevo_reg)[0]
    fouls_home = model_fouls_home.predict(X_nuevo_reg)[0]
    fouls_away = model_fouls_away.predict(X_nuevo_reg)[0]
    cards_home = model_cards_home.predict(X_nuevo_reg)[0]
    cards_away = model_cards_away.predict(X_nuevo_reg)[0]

    # Crear el diccionario de predicciones
    predicciones = {
        'pronostico': pronostico,
        'corners_home': corners_home,
        'corners_away': corners_away,
        'fouls_home': fouls_home,
        'fouls_away': fouls_away,
        'cards_home': cards_home,
        'cards_away': cards_away
    }

    return predicciones

# 14. Interacción con el usuario
def main():
    print("Bienvenido al sistema de predicción de partidos.")
    equipo_local = standardize_team_name(input("Ingrese el nombre del equipo local: "))
    equipo_visitante = standardize_team_name(input("Ingrese el nombre del equipo visitante: "))
    fecha_partido = input("Ingrese la fecha del partido (dd/mm/yyyy): ")

    predicciones = predecir_partido(equipo_local, equipo_visitante, fecha_partido, model_class, data)
    if predicciones:
        print(f"\nPronóstico para el partido {equipo_local} vs {equipo_visitante} el {fecha_partido}:")
        print(f"Resultado: {predicciones['pronostico']}")
        print(f"Corners a favor de {equipo_local}: {predicciones['corners_home']:.2f}")
        print(f"Corners a favor de {equipo_visitante}: {predicciones['corners_away']:.2f}")
        print(f"Faltas cometidas por {equipo_local}: {predicciones['fouls_home']:.2f}")
        print(f"Faltas cometidas por {equipo_visitante}: {predicciones['fouls_away']:.2f}")
        print(f"Tarjetas amarillas para {equipo_local}: {predicciones['cards_home']:.2f}")
        print(f"Tarjetas amarillas para {equipo_visitante}: {predicciones['cards_away']:.2f}")

if __name__ == "__main__":
    main()



