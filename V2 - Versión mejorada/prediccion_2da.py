# prediccion_mejorado.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
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

def standardize_team_name(name):
    return team_name_mapping.get(name.strip(), name.strip())

data_total['HomeTeam'] = data_total['HomeTeam'].apply(standardize_team_name)
data_total['AwayTeam'] = data_total['AwayTeam'].apply(standardize_team_name)

# 3. Convertir la columna 'Date' al formato datetime
data_total['Date'] = pd.to_datetime(data_total['Date'], format='%d/%m/%Y', errors='coerce')

# Eliminar filas con fechas nulas
data_total.dropna(subset=['Date'], inplace=True)

# 4. Seleccionar las columnas necesarias
required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                    'HTHG', 'HTAG', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF', 'HC', 'AC',
                    'B365H', 'B365D', 'B365A', 'AHh']

# Verificar que las columnas existen
missing_columns = [col for col in required_columns if col not in data_total.columns]
if missing_columns:
    print(f"Las siguientes columnas faltan en el DataFrame: {missing_columns}")
    # Si faltan columnas, no podemos continuar
    exit()

# Seleccionar las columnas
data = data_total[required_columns].copy()

# 5. Manejar valores nulos en las columnas seleccionadas
# Convertir columnas numéricas a tipo numérico
numerical_columns = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF', 'HC', 'AC',
                     'B365H', 'B365D', 'B365A', 'AHh']
data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')

# Eliminar filas con valores nulos en columnas numéricas
data.dropna(subset=numerical_columns, inplace=True)

# 6. Calcular probabilidades implícitas de las cuotas de apuestas
def get_implied_probabilities(row):
    odds = [row['B365H'], row['B365D'], row['B365A']]

    # Verificar si hay valores nulos o cero
    if any(pd.isnull(odds)) or any(odd == 0 for odd in odds):
        # Asignar probabilidades iguales si faltan datos
        probabilities = [1/3, 1/3, 1/3]
    else:
        probabilities = [1 / odd for odd in odds]
        total = sum(probabilities)
        # Verificar si el total es cero para evitar división por cero
        if total == 0:
            probabilities = [1/3, 1/3, 1/3]
        else:
            probabilities = [prob / total for prob in probabilities]  # Normalizar para que sumen 1
    return pd.Series({'prob_home_win': probabilities[0], 'prob_draw': probabilities[1], 'prob_away_win': probabilities[2]})

# Aplicar la función a los datos
data[['prob_home_win', 'prob_draw', 'prob_away_win']] = data.apply(get_implied_probabilities, axis=1)

# 7. Definir funciones para obtener estadísticas del equipo
def get_team_stats(team, date, data):
    # Partidos anteriores del equipo antes de la fecha dada
    past_matches = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)]
    total_matches = len(past_matches)
    if total_matches == 0:
        return pd.Series({
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'avg_goals_first_half': 0,
            'avg_goals_second_half': 0,
            'avg_yellow_cards': 0,
            'avg_red_cards': 0,
            'avg_fouls': 0,
            'avg_corners': 0,
            'recent_form': 0,
            'wins_recent': 0,
            'draws_recent': 0,
            'losses_recent': 0,
            'goal_diff': 0,
            'days_since_last_match': np.nan,
        })

    # Calcular estadísticas
    home_matches = past_matches[past_matches['HomeTeam'] == team]
    away_matches = past_matches[past_matches['AwayTeam'] == team]

    goals_scored = home_matches['FTHG'].sum() + away_matches['FTAG'].sum()
    goals_conceded = home_matches['FTAG'].sum() + away_matches['FTHG'].sum()
    goals_first_half = home_matches['HTHG'].sum() + away_matches['HTAG'].sum()
    goals_second_half = (home_matches['FTHG'] - home_matches['HTHG']).sum() + (away_matches['FTAG'] - away_matches['HTAG']).sum()

    yellow_cards = home_matches['HY'].sum() + away_matches['AY'].sum()
    red_cards = home_matches['HR'].sum() + away_matches['AR'].sum()
    fouls = home_matches['HF'].sum() + away_matches['AF'].sum()
    corners = home_matches['HC'].sum() + away_matches['AC'].sum()

    # Forma reciente
    recent_form = get_recent_form(team, date, data, n_matches=5)
    wins_recent = get_recent_results(team, date, data, n_matches=5, result='win')
    draws_recent = get_recent_results(team, date, data, n_matches=5, result='draw')
    losses_recent = get_recent_results(team, date, data, n_matches=5, result='loss')

    # Diferencia de goles promedio
    goal_diff = (goals_scored - goals_conceded) / total_matches if total_matches != 0 else 0

    # Días desde el último partido
    days_since_last_match = get_days_since_last_match(team, date, data)

    stats = pd.Series({
        'avg_goals_scored': goals_scored / total_matches if total_matches != 0 else 0,
        'avg_goals_conceded': goals_conceded / total_matches if total_matches != 0 else 0,
        'avg_goals_first_half': goals_first_half / total_matches if total_matches != 0 else 0,
        'avg_goals_second_half': goals_second_half / total_matches if total_matches != 0 else 0,
        'avg_yellow_cards': yellow_cards / total_matches if total_matches != 0 else 0,
        'avg_red_cards': red_cards / total_matches if total_matches != 0 else 0,
        'avg_fouls': fouls / total_matches if total_matches != 0 else 0,
        'avg_corners': corners / total_matches if total_matches != 0 else 0,
        'recent_form': recent_form,
        'wins_recent': wins_recent,
        'draws_recent': draws_recent,
        'losses_recent': losses_recent,
        'goal_diff': goal_diff,
        'days_since_last_match': days_since_last_match,
    })
    return stats

def get_recent_form(team, date, data, n_matches=5):
    past_matches = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)]
    past_matches = past_matches.sort_values(by='Date', ascending=False).head(n_matches)
    if past_matches.empty:
        return 0
    results = past_matches['FTR']
    points = 0
    for result, home_team, away_team in zip(results, past_matches['HomeTeam'], past_matches['AwayTeam']):
        if (team == home_team and result == 'H'):
            points += 3
        elif (team == away_team and result == 'A'):
            points += 3
        elif result == 'D':
            points += 1
    return points / (n_matches * 3)  # Normalizado entre 0 y 1

def get_recent_results(team, date, data, n_matches=5, result='win'):
    past_matches = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)]
    past_matches = past_matches.sort_values(by='Date', ascending=False).head(n_matches)
    if past_matches.empty:
        return 0
    count = 0
    for index, row in past_matches.iterrows():
        if result == 'win':
            if (row['HomeTeam'] == team and row['FTR'] == 'H') or (row['AwayTeam'] == team and row['FTR'] == 'A'):
                count += 1
        elif result == 'draw':
            if row['FTR'] == 'D':
                count += 1
        elif result == 'loss':
            if (row['HomeTeam'] == team and row['FTR'] == 'A') or (row['AwayTeam'] == team and row['FTR'] == 'H'):
                count += 1
    return count

def get_days_since_last_match(team, date, data):
    past_matches = data[((data['HomeTeam'] == team) | (data['AwayTeam'] == team)) & (data['Date'] < date)]
    if past_matches.empty:
        return np.nan
    last_match_date = past_matches['Date'].max()
    days_since = (date - last_match_date).days
    return days_since

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
        # Características del equipo local
        'home_avg_goals_scored': home_stats['avg_goals_scored'],
        'home_avg_goals_conceded': home_stats['avg_goals_conceded'],
        'home_avg_goals_first_half': home_stats['avg_goals_first_half'],
        'home_avg_goals_second_half': home_stats['avg_goals_second_half'],
        'home_avg_yellow_cards': home_stats['avg_yellow_cards'],
        'home_avg_red_cards': home_stats['avg_red_cards'],
        'home_avg_fouls': home_stats['avg_fouls'],
        'home_avg_corners': home_stats['avg_corners'],
        'home_recent_form': home_stats['recent_form'],
        'home_wins_recent': home_stats['wins_recent'],
        'home_draws_recent': home_stats['draws_recent'],
        'home_losses_recent': home_stats['losses_recent'],
        'home_goal_diff': home_stats['goal_diff'],
        'home_days_since_last_match': home_stats['days_since_last_match'],

        # Características del equipo visitante
        'away_avg_goals_scored': away_stats['avg_goals_scored'],
        'away_avg_goals_conceded': away_stats['avg_goals_conceded'],
        'away_avg_goals_first_half': away_stats['avg_goals_first_half'],
        'away_avg_goals_second_half': away_stats['avg_goals_second_half'],
        'away_avg_yellow_cards': away_stats['avg_yellow_cards'],
        'away_avg_red_cards': away_stats['avg_red_cards'],
        'away_avg_fouls': away_stats['avg_fouls'],
        'away_avg_corners': away_stats['avg_corners'],
        'away_recent_form': away_stats['recent_form'],
        'away_wins_recent': away_stats['wins_recent'],
        'away_draws_recent': away_stats['draws_recent'],
        'away_losses_recent': away_stats['losses_recent'],
        'away_goal_diff': away_stats['goal_diff'],
        'away_days_since_last_match': away_stats['days_since_last_match'],

        # Características del partido
        'prob_home_win': row['prob_home_win'],
        'prob_draw': row['prob_draw'],
        'prob_away_win': row['prob_away_win'],
        'AHh': row['AHh'],  # Handicap asiático
        'FTR': row['FTR'],
        'FTHG': row['FTHG'],
        'FTAG': row['FTAG'],
        'home_corners': row['HC'],
        'away_corners': row['AC'],
        'home_fouls': row['HF'],
        'away_fouls': row['AF'],
        'home_yellow_cards': row['HY'],
        'away_yellow_cards': row['AY'],
    }
    features.append(feature_row)

features_df = pd.DataFrame(features)

# 9. Manejar valores nulos en features_df
features_df.dropna(inplace=True)

# 10. Agregar nuevas características
# Total de goles
features_df['total_goals'] = features_df['FTHG'] + features_df['FTAG']

# Over/Under 2.5 goles
features_df['over_2_5'] = features_df['total_goals'].apply(lambda x: 1 if x > 2.5 else 0)

# 11. Preparar los datos para el modelo
# Codificar el resultado del partido
le_result = LabelEncoder()
features_df['result_code'] = le_result.fit_transform(features_df['FTR'])

# Variables predictoras y objetivo para clasificación
X = features_df.drop(['FTR', 'result_code', 'FTHG', 'FTAG', 'home_corners', 'away_corners',
                      'home_fouls', 'away_fouls', 'home_yellow_cards', 'away_yellow_cards',
                      'total_goals', 'over_2_5'], axis=1)
y = features_df['result_code']

# Variables para regresión y predicciones adicionales
X_reg = X.copy()
y_corners_home = features_df['home_corners']
y_corners_away = features_df['away_corners']
y_fouls_home = features_df['home_fouls']
y_fouls_away = features_df['away_fouls']
y_cards_home = features_df['home_yellow_cards']
y_cards_away = features_df['away_yellow_cards']
y_total_goals = features_df['total_goals']
y_over_under = features_df['over_2_5']

# 12. Escalar las características
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 13. Manejar el desequilibrio de clases con SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X_scaled, y)

# 14. Dividir los datos utilizando TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)

# 15. Definir el modelo y ajustar hiperparámetros para clasificación
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'class_weight': [None, 'balanced']
}

model_rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=tscv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_balanced, y_balanced)

best_model = grid_search.best_estimator_

print(f"Mejores hiperparámetros: {grid_search.best_params_}")

# 16. Evaluar el modelo con validación cruzada temporal
scores = cross_val_score(best_model, X_balanced, y_balanced, cv=tscv, scoring='accuracy')
print(f'Precisión promedio con validación cruzada temporal: {np.mean(scores):.2f}')

# 17. Métricas adicionales (validación cruzada manual)
y_true = []
y_pred = []

for train_index, test_index in tscv.split(X_balanced):
    X_train_cv, X_test_cv = X_balanced[train_index], X_balanced[test_index]
    y_train_cv, y_test_cv = y_balanced[train_index], y_balanced[test_index]

    # Entrenar el modelo en el conjunto de entrenamiento
    best_model.fit(X_train_cv, y_train_cv)

    # Realizar predicciones en el conjunto de prueba
    y_pred_cv = best_model.predict(X_test_cv)

    # Almacenar las etiquetas verdaderas y las predicciones
    y_true.extend(y_test_cv)
    y_pred.extend(y_pred_cv)

# Convertir listas a arreglos de NumPy si es necesario
y_true = np.array(y_true)
y_pred = np.array(y_pred)

print(classification_report(y_true, y_pred))
print("Matriz de confusión:")
print(confusion_matrix(y_true, y_pred))

# 18. Entrenar modelos de regresión para estadísticas adicionales
# No es necesario balancear los datos para regresión
X_reg_scaled = scaler.fit_transform(X_reg)

# Modelos de regresión
model_corners_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_corners_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_fouls_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_fouls_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_cards_home = RandomForestRegressor(n_estimators=100, random_state=42)
model_cards_away = RandomForestRegressor(n_estimators=100, random_state=42)
model_total_goals = RandomForestRegressor(n_estimators=100, random_state=42)
model_over_under = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar los modelos de regresión
model_corners_home.fit(X_reg_scaled, y_corners_home)
model_corners_away.fit(X_reg_scaled, y_corners_away)
model_fouls_home.fit(X_reg_scaled, y_fouls_home)
model_fouls_away.fit(X_reg_scaled, y_fouls_away)
model_cards_home.fit(X_reg_scaled, y_cards_home)
model_cards_away.fit(X_reg_scaled, y_cards_away)
model_total_goals.fit(X_reg_scaled, y_total_goals)
model_over_under.fit(X_reg_scaled, y_over_under)

# 19. Función para predecir el resultado y estadísticas de un partido
def predecir_partido(equipo_local, equipo_visitante, fecha_partido, modelo_class, data):
    # Convertir fecha_partido a datetime si es una cadena
    if isinstance(fecha_partido, str):
        fecha_partido = pd.to_datetime(fecha_partido, format='%d/%m/%Y', errors='coerce')
        if pd.isnull(fecha_partido):
            print("Error: La fecha del partido no tiene el formato correcto (dd/mm/yyyy).")
            return None

    # Estandarizar nombres de equipos
    equipo_local = standardize_team_name(equipo_local)
    equipo_visitante = standardize_team_name(equipo_visitante)

    # Verificar que los equipos existen
    equipos_disponibles = set(data['HomeTeam']).union(set(data['AwayTeam']))
    if equipo_local not in equipos_disponibles or equipo_visitante not in equipos_disponibles:
        print("Error: Uno o ambos equipos no están en el dataset.")
        return None

    # Obtener estadísticas de los equipos
    local_stats = get_team_stats(equipo_local, fecha_partido, data)
    visitante_stats = get_team_stats(equipo_visitante, fecha_partido, data)

    # Obtener probabilidades implícitas de las cuotas de apuestas
    # Para partidos futuros, podrías obtener las cuotas actuales de una API o sitio web
    # Aquí, asignaremos valores promedio
    prob_home_win = data['prob_home_win'].mean()
    prob_draw = data['prob_draw'].mean()
    prob_away_win = data['prob_away_win'].mean()

    AHh = data['AHh'].mean()  # Usamos el valor promedio del handicap asiático

    X_nuevo = pd.DataFrame([{
        # Características del equipo local
        'home_avg_goals_scored': local_stats['avg_goals_scored'],
        'home_avg_goals_conceded': local_stats['avg_goals_conceded'],
        'home_avg_goals_first_half': local_stats['avg_goals_first_half'],
        'home_avg_goals_second_half': local_stats['avg_goals_second_half'],
        'home_avg_yellow_cards': local_stats['avg_yellow_cards'],
        'home_avg_red_cards': local_stats['avg_red_cards'],
        'home_avg_fouls': local_stats['avg_fouls'],
        'home_avg_corners': local_stats['avg_corners'],
        'home_recent_form': local_stats['recent_form'],
        'home_wins_recent': local_stats['wins_recent'],
        'home_draws_recent': local_stats['draws_recent'],
        'home_losses_recent': local_stats['losses_recent'],
        'home_goal_diff': local_stats['goal_diff'],
        'home_days_since_last_match': local_stats['days_since_last_match'],

        # Características del equipo visitante
        'away_avg_goals_scored': visitante_stats['avg_goals_scored'],
        'away_avg_goals_conceded': visitante_stats['avg_goals_conceded'],
        'away_avg_goals_first_half': visitante_stats['avg_goals_first_half'],
        'away_avg_goals_second_half': visitante_stats['avg_goals_second_half'],
        'away_avg_yellow_cards': visitante_stats['avg_yellow_cards'],
        'away_avg_red_cards': visitante_stats['avg_red_cards'],
        'away_avg_fouls': visitante_stats['avg_fouls'],
        'away_avg_corners': visitante_stats['avg_corners'],
        'away_recent_form': visitante_stats['recent_form'],
        'away_wins_recent': visitante_stats['wins_recent'],
        'away_draws_recent': visitante_stats['draws_recent'],
        'away_losses_recent': visitante_stats['losses_recent'],
        'away_goal_diff': visitante_stats['goal_diff'],
        'away_days_since_last_match': visitante_stats['days_since_last_match'],

        # Características del partido
        'prob_home_win': prob_home_win,
        'prob_draw': prob_draw,
        'prob_away_win': prob_away_win,
        'AHh': AHh,
    }])

    # Manejar valores nulos
    X_nuevo.fillna(0, inplace=True)

    # Escalar las características
    X_nuevo_scaled = scaler.transform(X_nuevo)

    # Predicción de resultado
    resultado_cod = modelo_class.predict(X_nuevo_scaled)[0]
    resultado = le_result.inverse_transform([resultado_cod])[0]

    if resultado == 'H':
        pronostico = f'Victoria para {equipo_local}'
    elif resultado == 'A':
        pronostico = f'Victoria para {equipo_visitante}'
    else:
        pronostico = 'Empate'

    # Predicciones de estadísticas adicionales
    corners_home_pred = model_corners_home.predict(X_nuevo_scaled)[0]
    corners_away_pred = model_corners_away.predict(X_nuevo_scaled)[0]
    fouls_home_pred = model_fouls_home.predict(X_nuevo_scaled)[0]
    fouls_away_pred = model_fouls_away.predict(X_nuevo_scaled)[0]
    cards_home_pred = model_cards_home.predict(X_nuevo_scaled)[0]
    cards_away_pred = model_cards_away.predict(X_nuevo_scaled)[0]
    total_goals_pred = model_total_goals.predict(X_nuevo_scaled)[0]
    over_under_pred = model_over_under.predict(X_nuevo_scaled)[0]
    over_under_result = 'Más de 2.5 goles' if over_under_pred == 1 else 'Menos de 2.5 goles'

    # Crear el diccionario de predicciones
    predicciones = {
        'pronostico': pronostico,
        'corners_home': corners_home_pred,
        'corners_away': corners_away_pred,
        'fouls_home': fouls_home_pred,
        'fouls_away': fouls_away_pred,
        'cards_home': cards_home_pred,
        'cards_away': cards_away_pred,
        'total_goals': total_goals_pred,
        'over_under': over_under_result,
    }

    return predicciones

# 20. Interacción con el usuario
def main():
    print("Bienvenido al sistema de predicción de partidos mejorado.")
    equipo_local = input("Ingrese el nombre del equipo local: ")
    equipo_visitante = input("Ingrese el nombre del equipo visitante: ")
    fecha_partido = input("Ingrese la fecha del partido (dd/mm/yyyy): ")

    predicciones = predecir_partido(equipo_local, equipo_visitante, fecha_partido, best_model, data)
    if predicciones:
        print(f"\nPronóstico para el partido {equipo_local} vs {equipo_visitante} el {fecha_partido}:")
        print(f"Resultado: {predicciones['pronostico']}")
        print(f"Total de goles esperado: {predicciones['total_goals']:.2f}")
        print(f"Pronóstico Over/Under 2.5 goles: {predicciones['over_under']}")
        print(f"Corners a favor de {equipo_local}: {predicciones['corners_home']:.2f}")
        print(f"Corners a favor de {equipo_visitante}: {predicciones['corners_away']:.2f}")
        print(f"Faltas cometidas por {equipo_local}: {predicciones['fouls_home']:.2f}")
        print(f"Faltas cometidas por {equipo_visitante}: {predicciones['fouls_away']:.2f}")
        print(f"Tarjetas amarillas para {equipo_local}: {predicciones['cards_home']:.2f}")
        print(f"Tarjetas amarillas para {equipo_visitante}: {predicciones['cards_away']:.2f}")

if __name__ == "__main__":
    main()
