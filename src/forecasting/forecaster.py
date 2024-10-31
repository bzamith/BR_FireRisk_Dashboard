import joblib
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Attention, Bidirectional, Dense,
    Lambda, LSTM
)
from tqdm import tqdm
from typing import Tuple


# Configuration
FORECASTING_VARIABLES = [
    "precipitacao_total",
    "pressao_atmosferica",
    "temperatura_ar",
    "umidade_relativa",
    "velocidade_vento"
]
FORECASTER_OBJECTIVE = 'val_loss'
NB_TRIALS = 20
OBSERVATION_WINDOW = 14
EPOCHS = 50
BATCH_SIZE = 32

FORECASTING_PATH = "forecasting"
FORECASTING_FORECASTERS_PATH = f"{FORECASTING_PATH}/forecasters"
FORECASTING_SCALERS_PATH = f"{FORECASTING_PATH}/scalers"
FORECASTING_METRICS_PATH = f"{FORECASTING_PATH}/metrics"
FORECASTING_PLOTS_PATH = f"{FORECASTING_PATH}/plots"

np.random.seed(42)
tf.random.set_seed(42)


def prepare_data(
    df: pd.DataFrame,
    station: str,
    variable: str
) -> Tuple[np.array, np.array, RobustScaler]:
    station_data = df[df['codigo_estacao'] == station][['data', variable]]
    station_data = station_data.set_index('data')

    if station_data.empty:
        print(f"Warning: No data found for station '{station}' and variable '{variable}'. Skipping...")
        return None, None, None

    scaler = RobustScaler()
    station_data_scaled = scaler.fit_transform(station_data)

    X, y = [], []
    for i in range(OBSERVATION_WINDOW, len(station_data_scaled)):
        X.append(station_data_scaled[i - OBSERVATION_WINDOW:i])
        y.append(station_data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    return X, y, scaler


def build_lstm_model(input_shape: Tuple[int]) -> Sequential:
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def build_attention_lstm_model(input_shape: Tuple[int]) -> Model:
    model = Sequential()
    model.add(Bidirectional(LSTM(50, activation='relu', return_sequences=True), input_shape=input_shape))
    model.add(Lambda(lambda x: [x, x]))  # Query and Value are the same for self-attention
    model.add(Attention())
    model.add(Lambda(lambda x: K.mean(x, axis=1)))  # Aggregating attention output
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def train_forecaster(
    df: pd.DataFrame,
    station: str,
    variable: str
) -> dict:
    station_results = {}
    X, y, scaler = prepare_data(df, station, variable)
    if X is None or y is None:
        return {}
    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.2, shuffle=False)

    model = build_lstm_model(
        input_shape=(X_train.shape[1], X_train.shape[2])
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=0
    )

    y_pred = model.predict(X_test)
    y_pred_unscaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    mape_unscaled = mean_absolute_percentage_error(y_test_unscaled, y_pred_unscaled)
    mse_unscaled = mean_squared_error(y_test_unscaled, y_pred_unscaled)

    station_results = {
        "model": model,
        "scaler": scaler,
        "history": history.history,
        "mape": mape_unscaled,
        "mse": mse_unscaled,
        "val_loss_history": history.history['val_loss'],
        "y_pred_unscaled": y_pred_unscaled,
        "y_test_unscaled": y_test_unscaled
    }

    return station_results


def plot_results(
    station_results: dict,
    station: str,
) -> None:
    for variable, metrics in station_results.items():
        plt.plot(metrics['val_loss_history'], label=f"{station} - {variable}")
    plt.title(f"Validation Loss over Epochs for {station}")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Loss")
    plt.legend()
    plt.savefig(f"{FORECASTING_PLOTS_PATH}/{station}_plot.png", format='png')
    plt.close()


def save_models_and_metrics(
    station_results: dict,
    station: str,
) -> None:
    for variable, data in station_results.items():
        forecaster_file_name = f"{FORECASTING_FORECASTERS_PATH}/{station}_{variable}_forecaster.keras"
        data['model'].save(forecaster_file_name)
        scaler_file_name = f"{FORECASTING_SCALERS_PATH}/{station}_{variable}_scaler.joblib"
        joblib.dump(data['scaler'], scaler_file_name)

    metrics_file_name = f"{FORECASTING_METRICS_PATH}/{station}_metrics.json"
    station_metrics = {
        variable: {
            "MAPE": data['mape'],
            "MSE": data['mse']
        }
        for variable, data in station_results.items()
    }
    with open(metrics_file_name, "w") as f:
        json.dump(station_metrics, f, indent=4)
    

def train_and_save_forecasting_data(df: pd.DataFrame):
    results = {}
    stations = df['codigo_estacao'].unique()

    os.makedirs(FORECASTING_FORECASTERS_PATH, exist_ok=True)
    os.makedirs(FORECASTING_SCALERS_PATH, exist_ok=True)
    os.makedirs(FORECASTING_METRICS_PATH, exist_ok=True)
    os.makedirs(FORECASTING_PLOTS_PATH, exist_ok=True)

    for station in tqdm(stations, desc="Training for each station"):
        station_results = {}
        for variable in tqdm(FORECASTING_VARIABLES, desc="Training for each variable"):
            results = train_forecaster(df, station, variable)
            station_results[variable] = results
        save_models_and_metrics(station_results, station)
        plot_results(station_results, station)
