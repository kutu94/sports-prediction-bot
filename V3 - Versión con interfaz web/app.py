# app.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier  # Add this import
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import streamlit as st
import os
import warnings

warnings.filterwarnings('ignore')

# Define global functions so they can be accessed by both train_models and predecir_partido
from collections import defaultdict

team_stats_cache = defaultdict(dict)


def standardize_team_name(name):
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
        # Add more mappings if you find other inconsistencies
    }
    return team_name_mapping.get(name.strip().title(), name.strip().title())


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
    return points / (n_matches * 3)  # Normalized between 0 and 1


def get_team_stats(team, date, data):
    key = (team, date)
    if key in team_stats_cache:
        return team_stats_cache[key]

    past_matches_home = data[(data['HomeTeam'] == team) & (data['Date'] < date)]
    past_matches_away = data[(data['AwayTeam'] == team) & (data['Date'] < date)]
    total_matches = len(past_matches_home) + len(past_matches_away)
    if total_matches == 0:
        stats = pd.Series({
            'avg_goals_scored': 0,
            'avg_goals_conceded': 0,
            'avg_yellow_cards': 0,
            'avg_red_cards': 0,
            'avg_fouls': 0,
            'avg_corners': 0,
            'recent_form': 0
        })
        team_stats_cache[key] = stats
        return stats

    goals_scored = past_matches_home['FTHG'].sum() + past_matches_away['FTAG'].sum()
    goals_conceded = past_matches_home['FTAG'].sum() + past_matches_away['FTHG'].sum()
    yellow_cards = past_matches_home['HY'].sum() + past_matches_away['AY'].sum()
    red_cards = past_matches_home['HR'].sum() + past_matches_away['AR'].sum()
    fouls = past_matches_home['HF'].sum() + past_matches_away['AF'].sum()
    corners = past_matches_home['HC'].sum() + past_matches_away['AC'].sum()
    recent_form = get_recent_form(team, date, data, n_matches=5)

    stats = pd.Series({
        'avg_goals_scored': goals_scored / total_matches,
        'avg_goals_conceded': goals_conceded / total_matches,
        'avg_yellow_cards': yellow_cards / total_matches,
        'avg_red_cards': red_cards / total_matches,
        'avg_fouls': fouls / total_matches,
        'avg_corners': corners / total_matches,
        'recent_form': recent_form
    })
    team_stats_cache[key] = stats
    return stats


@st.cache_data
def load_data():
    # Load and combine the data
    dataframes = []
    csv_files_primera = ['../data/Primera2018.csv', '../data/Primera2019.csv', '../data/Primera2020.csv',
                         '../data/Primera2021.csv', '../data/Primera2022.csv', '../data/Primera2023.csv', '../data/Primera2024.csv']

    for file in csv_files_primera:
        df = pd.read_csv(file)
        df['Division'] = 'Primera'
        dataframes.append(df)

    csv_files_segunda = ['../data/Segunda2018.csv', '../data/Segunda2019.csv', '../data/Segunda2020.csv',
                         '../data/Segunda2021.csv', '../data/Segunda2022.csv', '../data/Segunda2023.csv', '../data/Segunda2024.csv']

    for file in csv_files_segunda:
        df = pd.read_csv(file)
        df['Division'] = 'Segunda'
        dataframes.append(df)

    data_total = pd.concat(dataframes, ignore_index=True)

    # Standardize team names
    data_total['HomeTeam'] = data_total['HomeTeam'].apply(standardize_team_name)
    data_total['AwayTeam'] = data_total['AwayTeam'].apply(standardize_team_name)

    # Convert 'Date' column to datetime format
    data_total['Date'] = pd.to_datetime(data_total['Date'], format='%d/%m/%Y', errors='coerce')

    # Remove rows with null dates
    data_total.dropna(subset=['Date'], inplace=True)

    # Select required columns
    required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
                        'HY', 'AY', 'HR', 'AR', 'HF', 'AF', 'HC', 'AC', 'Division']

    data = data_total[required_columns].copy()

    # Handle missing values
    data.dropna(subset=required_columns, inplace=True)
    numerical_columns = ['FTHG', 'FTAG', 'HY', 'AY', 'HR', 'AR', 'HF', 'AF', 'HC', 'AC']
    data[numerical_columns] = data[numerical_columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(subset=numerical_columns, inplace=True)

    return data


@st.cache_resource
def train_models(data):
    # Generate features
    features = []
    for index, row in data.iterrows():
        date = row['Date']
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']

        # Statistics for home and away teams
        home_stats = get_team_stats(home_team, date, data)
        away_stats = get_team_stats(away_team, date, data)

        # Create feature dictionary
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
            'Date': row['Date'],
            'Division': row['Division']
        }

        features.append(feature_row)

    features_df = pd.DataFrame(features)

    # Prepare data for the model
    le_result = LabelEncoder()
    features_df['result_code'] = le_result.fit_transform(features_df['FTR'])

    # Sort data by date
    features_df = features_df.sort_values(by='Date')

    # Define cutoff date for train-test split
    cutoff_date = '2022-07-01'  # For example
    train_data = features_df[features_df['Date'] < cutoff_date]
    test_data = features_df[features_df['Date'] >= cutoff_date]

    # Select feature columns
    features_columns = ['home_avg_goals_scored', 'home_avg_goals_conceded', 'home_avg_yellow_cards',
                        'home_avg_red_cards', 'home_avg_fouls', 'home_avg_corners', 'home_recent_form',
                        'away_avg_goals_scored', 'away_avg_goals_conceded', 'away_avg_yellow_cards',
                        'away_avg_red_cards', 'away_avg_fouls', 'away_avg_corners', 'away_recent_form']

    X_train = train_data[features_columns]
    y_train = train_data['result_code']
    X_test = test_data[features_columns]
    y_test = test_data['result_code']

    # Handle missing values
    X_train.dropna(inplace=True)
    y_train = y_train[X_train.index]

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    # Train classification model
    model_class = RandomForestClassifier(n_estimators=100, random_state=42)
    model_class.fit(X_train_balanced, y_train_balanced)

    # Evaluate the model
    y_pred = model_class.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy on test set: {accuracy * 100:.2f}%")

    # Save the model and necessary objects
    joblib.dump(model_class, 'modelo_clasificacion.pkl')
    joblib.dump(le_result, 'label_encoder.pkl')
    joblib.dump(features_columns, 'features_columns.pkl')
    joblib.dump(X_train, 'X_train.pkl')  # Save training data for future use
    joblib.dump(y_train, 'y_train.pkl')
    joblib.dump(train_data, 'train_data.pkl')

    # Train regression models for predicting corners, fouls, and cards
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    joblib.dump(scaler, 'scaler.pkl')

    y_corners_home = train_data['home_corners']
    y_corners_away = train_data['away_corners']
    y_fouls_home = train_data['home_fouls']
    y_fouls_away = train_data['away_fouls']
    y_cards_home = train_data['home_yellow_cards']
    y_cards_away = train_data['away_yellow_cards']

    model_corners_home = xgb.XGBRegressor(random_state=42)
    model_corners_home.fit(X_train_scaled, y_corners_home)
    joblib.dump(model_corners_home, 'modelo_corners_home.pkl')

    model_corners_away = xgb.XGBRegressor(random_state=42)
    model_corners_away.fit(X_train_scaled, y_corners_away)
    joblib.dump(model_corners_away, 'modelo_corners_away.pkl')

    model_fouls_home = xgb.XGBRegressor(random_state=42)
    model_fouls_home.fit(X_train_scaled, y_fouls_home)
    joblib.dump(model_fouls_home, 'modelo_fouls_home.pkl')

    model_fouls_away = xgb.XGBRegressor(random_state=42)
    model_fouls_away.fit(X_train_scaled, y_fouls_away)
    joblib.dump(model_fouls_away, 'modelo_fouls_away.pkl')

    model_cards_home = xgb.XGBRegressor(random_state=42)
    model_cards_home.fit(X_train_scaled, y_cards_home)
    joblib.dump(model_cards_home, 'modelo_cards_home.pkl')

    model_cards_away = xgb.XGBRegressor(random_state=42)
    model_cards_away.fit(X_train_scaled, y_cards_away)
    joblib.dump(model_cards_away, 'modelo_cards_away.pkl')

    return


def predecir_partido(equipo_local, equipo_visitante, fecha_partido, data):
    # Load necessary models and objects
    model_class = joblib.load('modelo_clasificacion.pkl')
    le_result = joblib.load('label_encoder.pkl')
    features_columns = joblib.load('features_columns.pkl')
    scaler = joblib.load('scaler.pkl')
    model_corners_home = joblib.load('modelo_corners_home.pkl')
    model_corners_away = joblib.load('modelo_corners_away.pkl')
    model_fouls_home = joblib.load('modelo_fouls_home.pkl')
    model_fouls_away = joblib.load('modelo_fouls_away.pkl')
    model_cards_home = joblib.load('modelo_cards_home.pkl')
    model_cards_away = joblib.load('modelo_cards_away.pkl')

    # Convert fecha_partido to datetime
    if isinstance(fecha_partido, str):
        fecha_partido = pd.to_datetime(fecha_partido, format='%d/%m/%Y', errors='coerce')
        if pd.isnull(fecha_partido):
            st.error("Error: La fecha del partido no tiene el formato correcto (dd/mm/yyyy).")
            return None

    # Standardize team names
    equipo_local = standardize_team_name(equipo_local)
    equipo_visitante = standardize_team_name(equipo_visitante)

    # Verify that teams exist in the dataset
    equipos_disponibles = set(data['HomeTeam']).union(set(data['AwayTeam']))
    if equipo_local not in equipos_disponibles or equipo_visitante not in equipos_disponibles:
        st.error("Error: Uno o ambos equipos no están en el dataset.")
        return None

    # Get team statistics
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
    }], columns=features_columns)

    if X_nuevo.isnull().sum().sum() > 0:
        st.error("No hay suficientes datos históricos para realizar la predicción.")
        return None

    # Scale features
    X_nuevo_scaled = scaler.transform(X_nuevo)

    # Predict result
    resultado_cod = model_class.predict(X_nuevo_scaled)[0]
    resultado = le_result.inverse_transform([resultado_cod])[0]

    if resultado == 'H':
        pronostico = f'Victoria para {equipo_local}'
    elif resultado == 'A':
        pronostico = f'Victoria para {equipo_visitante}'
    else:
        pronostico = 'Empate'

    # Predict other statistics
    corners_home = model_corners_home.predict(X_nuevo_scaled)[0]
    corners_away = model_corners_away.predict(X_nuevo_scaled)[0]
    fouls_home = model_fouls_home.predict(X_nuevo_scaled)[0]
    fouls_away = model_fouls_away.predict(X_nuevo_scaled)[0]
    cards_home = model_cards_home.predict(X_nuevo_scaled)[0]
    cards_away = model_cards_away.predict(X_nuevo_scaled)[0]

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


def main():
    st.title("Predicción de Partidos de Fútbol")

    # Load data and train models if necessary
    data = load_data()

    # Check if all necessary model files exist
    model_files = ['modelo_clasificacion.pkl', 'label_encoder.pkl', 'features_columns.pkl', 'scaler.pkl',
                   'modelo_corners_home.pkl', 'modelo_corners_away.pkl', 'modelo_fouls_home.pkl',
                   'modelo_fouls_away.pkl', 'modelo_cards_home.pkl', 'modelo_cards_away.pkl']

    models_exist = all(os.path.exists(f) for f in model_files)

    if not models_exist:
        with st.spinner('Entrenando modelos, por favor espera...'):
            train_models(data)

    equipo_local = st.text_input("Ingrese el nombre del equipo local")
    equipo_visitante = st.text_input("Ingrese el nombre del equipo visitante")
    fecha_partido = st.text_input("Ingrese la fecha del partido (dd/mm/yyyy)")

    if st.button("Predecir"):
        if fecha_partido == '':
            st.error("Por favor, ingrese la fecha del partido.")
        else:
            predicciones = predecir_partido(equipo_local, equipo_visitante, fecha_partido, data)
            if predicciones:
                st.subheader(f"Pronóstico para el partido {equipo_local} vs {equipo_visitante} el {fecha_partido}:")
                st.write(f"**Resultado:** {predicciones['pronostico']}")
                st.write(f"**Corners a favor de {equipo_local}:** {predicciones['corners_home']:.2f}")
                st.write(f"**Corners a favor de {equipo_visitante}:** {predicciones['corners_away']:.2f}")
                st.write(f"**Faltas cometidas por {equipo_local}:** {predicciones['fouls_home']:.2f}")
                st.write(f"**Faltas cometidas por {equipo_visitante}:** {predicciones['fouls_away']:.2f}")
                st.write(f"**Tarjetas amarillas para {equipo_local}:** {predicciones['cards_home']:.2f}")
                st.write(f"**Tarjetas amarillas para {equipo_visitante}:** {predicciones['cards_away']:.2f}")
            else:
                st.error("No se pudo realizar la predicción. Verifique los datos ingresados.")


if __name__ == "__main__":
    main()
