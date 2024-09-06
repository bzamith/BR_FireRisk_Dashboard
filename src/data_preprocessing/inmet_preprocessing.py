import os
import re
from typing import List, Tuple

import numpy as np

import pandas as pd

from tqdm import tqdm

from ..utils import (
    calculate_angstrom_risk, calculate_telicyn_risk,
    get_all_file_paths
)

# Paths for data directories
RAW_DATA_INMET_PATH = "data/raw_data/INMET"
PREPROCESSED_DATA_INMET_PATH = f"data/preprocessed_data/INMET"

# Expected keys and columns
EXPECTED_INMET_INFO_KEYS = [
    'regiao', 'uf', 'estacao', 'codigo_estacao',
    'latitude', 'longitude', 'altitude'
]
INMET_CLIMATE_DATA_COLS = [
    'precipitacao_total', 'pressao_atmosferica', 'temperatura_ar',
    'temperatura_ponto_orvalho', 'umidade_relativa', 'velocidade_vento'
]
EXPECTED_INMET_DATA_COLS = ['data', 'hora'] + INMET_CLIMATE_DATA_COLS


def __extract_inmet_station_metadata(file_path: str) -> dict:
    """Extract metadata information from an INMET file."""
    metadata = pd.read_csv(file_path, encoding='latin1', nrows=9, header=None, delimiter=';', on_bad_lines='skip').T
    metadata.columns = metadata.iloc[0]
    metadata = metadata[1:].reset_index(drop=True)

    if metadata.shape[1] != 8:
        raise Exception(f"File {file_path} has incorrect number of fields: {metadata.shape[1]}")

    file_info_dict = {}
    for col in metadata.columns:
        value = metadata.at[0, col]
        if col in ['REGIAO:', 'REGIÃO:', 'REGI?O:']:
            file_info_dict['regiao'] = value
        elif col == 'UF:':
            file_info_dict['uf'] = value
        elif col in ['ESTACAO:', 'ESTAÇÃO:', 'ESTAC?O:']:
            file_info_dict['estacao'] = value
        elif col in ['CODIGO:', 'CODIGO (WMO):']:
            file_info_dict['codigo_estacao'] = value
        elif col == 'LATITUDE:':
            file_info_dict['latitude'] = float(value.replace(',', '.'))
        elif col == 'LONGITUDE:':
            file_info_dict['longitude'] = float(value.replace(',', '.'))
        elif col == 'ALTITUDE:':
            file_info_dict['altitude'] = float(value.replace(',', '.'))

    if sorted(list(file_info_dict.keys())) != sorted(EXPECTED_INMET_INFO_KEYS):
        raise Exception(f"{os.path.basename(file_path)}: Missing columns INMET station metadata. Found: {metadata.columns}")

    return {key: file_info_dict.get(key, np.nan) for key in EXPECTED_INMET_INFO_KEYS}


def __extract_inmet_data(file_path: str) -> pd.DataFrame:
    """Extract and preprocess data from an INMET file."""
    def __fix_hour(value: str):
        pattern = re.compile(r'^\d{4} UTC$')

        if pattern.match(value):
            time_str = value.replace(' UTC', '')
            time_obj = pd.to_datetime(time_str, format='%H%M')
            return time_obj.strftime('%H:%M')
        else:
            return value
    raw_df = pd.read_csv(file_path, encoding='latin1', skiprows=8, header=0, delimiter=';', on_bad_lines='skip')
    raw_df = raw_df.replace(['-9999', -9999], np.nan)

    df = pd.DataFrame()
    for col in raw_df.columns:
        value = raw_df[col]
        if col in ['DATA (YYYY-MM-DD)', 'Data']:
            df['data'] = value.astype(str).str.replace('/', '-')
        elif col in ['HORA (UTC)', 'Hora UTC']:
            df['hora'] = value.astype(str).apply(__fix_hour)
        elif col == 'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)':
            df['precipitacao_total'] = value.astype(str).str.replace(',', '.').astype(float)
        elif col == 'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)':
            df['pressao_atmosferica'] = value.astype(str).str.replace(',', '.').astype(float)
        elif col == 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)':
            df['temperatura_ar'] = value.astype(str).str.replace(',', '.').astype(float)
        elif col == 'TEMPERATURA DO PONTO DE ORVALHO (°C)':
            df['temperatura_ponto_orvalho'] = value.astype(str).str.replace(',', '.').astype(float)
        elif col == 'UMIDADE RELATIVA DO AR, HORARIA (%)':
            df['umidade_relativa'] = value.astype(str).str.replace(',', '.').astype(float)
        elif col == 'VENTO, VELOCIDADE HORARIA (m/s)':
            df['velocidade_vento'] = value.astype(str).str.replace(',', '.').astype(float)

    if sorted(df.columns) != sorted(EXPECTED_INMET_DATA_COLS):
        raise Exception(f"{os.path.basename(file_path)}: Missing columns INMET data. Found: {list(df.columns)}\n, expected: {sorted(EXPECTED_INMET_DATA_COLS)}")

    return df


def __group_daily_inmet_data(df: pd.DataFrame) -> pd.DataFrame:
    df['hora'] = pd.to_datetime(df['hora'], format='%H:%M').dt.hour

    df_13 = df[df['hora'] == 13]
    df_13 = df_13.rename(
        columns={
            'pressao_atmosferica': 'pressao_atmosferica_13',
            'temperatura_ar': 'temperatura_ar_13',
            'temperatura_ponto_orvalho': 'temperatura_ponto_orvalho_13',
            'umidade_relativa': 'umidade_relativa_13',
            'velocidade_vento': 'velocidade_vento_13',
            'precipitacao_total_interpolacao': 'precipitacao_total_interpolacao_13',
            'pressao_atmosferica_interpolacao': 'pressao_atmosferica_interpolacao_13',
            'temperatura_ar_interpolacao': 'temperatura_ar_interpolacao_13',
            'temperatura_ponto_orvalho_interpolacao': 'temperatura_ponto_orvalho_interpolacao_13',
            'umidade_relativa_interpolacao': 'umidade_relativa_interpolacao_13',
            'velocidade_vento_interpolacao': 'velocidade_vento_interpolacao_13',
        }
    )

    df = df.groupby('data').agg({
        'regiao': 'first',
        'uf': 'first',
        'estacao': 'first',
        'codigo_estacao': 'first',
        'latitude': 'first',
        'longitude': 'first',
        'altitude': 'first',
        'precipitacao_total': 'sum',
        'pressao_atmosferica': 'mean',
        'temperatura_ar': 'mean',
        'temperatura_ponto_orvalho': 'mean',
        'umidade_relativa': 'mean',
        'velocidade_vento': 'mean',
        'precipitacao_total_interpolacao': 'max',
        'pressao_atmosferica_interpolacao': 'max',
        'temperatura_ar_interpolacao': 'max',
        'temperatura_ponto_orvalho_interpolacao': 'max',
        'umidade_relativa_interpolacao': 'max',
        'velocidade_vento_interpolacao': 'max',
    }).reset_index()

    df = df.merge(
        df_13[[
            'data',
            'pressao_atmosferica_13',
            'temperatura_ar_13',
            'temperatura_ponto_orvalho_13',
            'umidade_relativa_13',
            'velocidade_vento_13',
            'precipitacao_total_interpolacao_13',
            'pressao_atmosferica_interpolacao_13',
            'temperatura_ar_interpolacao_13',
            'temperatura_ponto_orvalho_interpolacao_13',
            'umidade_relativa_interpolacao_13',
            'velocidade_vento_interpolacao_13'
        ]],
        on='data',
        how='left'
    )

    return df


def __calculate_days_without_rain_inmet_data(df: pd.DataFrame) -> pd.DataFrame:
    df['rained'] = df['precipitacao_total'] == 0
    df['rain_sequence'] = df['rained'].ne(df['rained'].shift()).cumsum()
    df['dias_sem_chuva'] = df.groupby('rain_sequence').cumcount() + 1
    df = df.drop(columns=['rained', 'rain_sequence'])
    return df


def __preprocess_inmet_data(file_paths: List[str]) -> pd.DataFrame:
    """Read, preprocess, and merge INMET data from files."""
    dfs = []

    for file_path in file_paths:
        file_info = __extract_inmet_station_metadata(file_path)
        data_df = __extract_inmet_data(file_path)
        for key, value in file_info.items():
            data_df[key] = value
        data_df = data_df[EXPECTED_INMET_INFO_KEYS + EXPECTED_INMET_DATA_COLS]
        dfs.append(data_df)

    df = pd.concat(dfs, axis=0)

    if df[EXPECTED_INMET_DATA_COLS].isna().all(axis=0).all():
        raise Exception(f"{os.path.basename(file_path)}: Only null values found.")

    df = df.dropna(subset=['data', 'hora'])
    if df.shape[0] == 0:
        raise Exception(f"{os.path.basename(file_path)}: No rows left after droping the data and hora NaNs.")

    df = df.sort_values(by=['data', 'hora'])
    for info_col in EXPECTED_INMET_INFO_KEYS:
        df[info_col] = df[info_col].ffill().bfill()

    for col in INMET_CLIMATE_DATA_COLS:
        interpolacao_col = f"{col}_interpolacao"
        if interpolacao_col not in df.columns:
            df[interpolacao_col] = False
        missing_before = df[col].isna()
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
        df[interpolacao_col] = df[interpolacao_col] | missing_before & df[col].notna()
        df[col] = df[col]

    df = __group_daily_inmet_data(df)
    df = __calculate_days_without_rain_inmet_data(df)

    df = calculate_angstrom_risk(df)
    df = calculate_telicyn_risk(df)

    return df.round(2)


def extract_inmet_station_codes() -> List[str]:
    """Extract unique INMET station codes from file names."""
    file_paths = get_all_file_paths(RAW_DATA_INMET_PATH, "CSV")
    stations = {re.search(r"_A\d{3}_", os.path.basename(path)).group()[1:5] for path in file_paths if
                re.search(r"_A\d{3}_", os.path.basename(path))}
    return sorted(list(set(stations)))


def process_and_save_inmet_climate_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read, filter, preprocess, and save INMET climate data."""
    stations = extract_inmet_station_codes()
    file_paths = get_all_file_paths(RAW_DATA_INMET_PATH, "CSV")
    total_stations = len(stations)
    
    inmet_df = []

    output_dir = f"{PREPROCESSED_DATA_INMET_PATH}/per_station"
    os.makedirs(output_dir, exist_ok=True)

    for station in tqdm(stations, total=total_stations, desc="Processing INMET"):
        station_file_paths = [f for f in file_paths if f"_{station.upper()}_" in f]
        station_inmet_df = __preprocess_inmet_data(station_file_paths)

        station_inmet_df.to_csv(f"{output_dir}/inmet_{station}.csv", index=False)
        
        inmet_df.append(station_inmet_df)

    print("Saving INMET data...")
    inmet_df = pd.concat(inmet_df, axis=0)
    inmet_df.to_csv(f"{PREPROCESSED_DATA_INMET_PATH}/inmet_data.csv", index=False)

    print("Saving stations info data...")
    stations_info_df = inmet_df[EXPECTED_INMET_INFO_KEYS].drop_duplicates().reset_index(drop=True)
    stations_info_df.to_csv(f"{PREPROCESSED_DATA_INMET_PATH}/inmet_stations_info.csv", index=False)

    return stations_info_df, inmet_df
