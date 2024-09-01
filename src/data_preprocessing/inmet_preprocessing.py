import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd

from ..utils import get_all_file_paths

# Paths for data directories
RAW_DATA_PATH = "data/raw_data/INMET"
PREPROCESSED_DATA_PATH = "data/preprocessed_data/INMET"

# Expected keys and columns
EXPECTED_INMET_INFO_KEYS = [
    'regiao', 'uf', 'estacao', 'codigo_estacao',
    'latitude', 'longitude', 'altitude', 'data_fundacao'
]
INMET_CLIMATE_DATA_COLS = [
    'precipitacao_total', 'pressao_atmosferica', 'temperatura_ar',
    'temperatura_ponto_orvalho', 'umidade_relativa', 'velocidade_vento'
]
EXPECTED_INMET_DATA_COLS = ['data', 'hora'] + INMET_CLIMATE_DATA_COLS


def extract_inmet_station_codes() -> List[str]:
    """Extract unique INMET station codes from file names."""
    file_paths = get_all_file_paths(RAW_DATA_PATH, "CSV")
    stations = {re.search(r"_A\d{3}_", os.path.basename(path)).group()[1:5] for path in file_paths if
                re.search(r"_A\d{3}_", os.path.basename(path))}
    return list(set(stations))


def extract_inmet_metadata(file_path: str) -> dict:
    """Extract metadata information from an INMET file."""
    metadata = pd.read_csv(file_path, encoding='latin1', nrows=9, header=None, delimiter=';', on_bad_lines='skip').T
    metadata.columns = metadata.iloc[0]
    metadata = metadata[1:].reset_index(drop=True)

    if metadata.shape[1] != 8:
        raise Exception(f"File {file_path} has incorrect number of fields: {metadata.shape[1]}")

    column_mapping = {
        'REGIÃO': 'regiao',
        'UF': 'uf',
        'ESTAÇÃO': 'estacao',
        'CODIGO': 'codigo_estacao',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'ALTITUDE': 'altitude',
        'DATA DE FUNDAÇÃO': 'data_fundacao'
    }

    file_info_dict = {}
    for column in metadata.columns:
        column_upper = column.upper()
        key = next((v for k, v in column_mapping.items() if column_upper.startswith(k)), None)
        if key:
            value = metadata.at[0, column].replace(',', '.') if key in {'latitude', 'longitude', 'altitude'} else \
            metadata.at[0, column]
            file_info_dict[key] = float(value) if key in {'latitude', 'longitude', 'altitude'} else value

    return {key: file_info_dict.get(key, np.nan) for key in EXPECTED_INMET_INFO_KEYS}


def extract_inmet_data(file_path: str) -> pd.DataFrame:
    """Extract and preprocess data from an INMET file."""
    raw_data = pd.read_csv(file_path, encoding='latin1', skiprows=8, header=0, delimiter=';', on_bad_lines='skip')
    raw_data = raw_data.replace(['-9999', -9999], np.nan)

    column_mapping = {
        'DATA': 'data',
        'HORA': 'hora',
        'PRECIPITAÇÃO TOTAL': 'precipitacao_total',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO': 'pressao_atmosferica',
        'TEMPERATURA DO AR': 'temperatura_ar',
        'TEMPERATURA DO PONTO DE ORVALHO': 'temperatura_ponto_orvalho',
        'UMIDADE RELATIVA DO AR': 'umidade_relativa',
        'VENTO, VELOCIDADE HORARIA': 'velocidade_vento'
    }

    df = pd.DataFrame()
    for column in raw_data.columns:
        column_upper = column.upper()
        for key, value in column_mapping.items():
            if column_upper.startswith(key):
                df[value] = raw_data[column].astype(str).str.replace(',', '.').astype(
                    float) if 'total' in value or 'pressao' in value or 'temperatura' in value or 'velocidade' in value else \
                raw_data[column]
                break

    if df.shape[1] != 8:
        raise Exception(f"Found {df.shape[1]} columns for {file_path} after preprocessing ({df.columns}). Expected 8.")

    return df


def filter_file_paths(file_paths: List[str], year_filter: int = None, uf_filter: str = None,
                      station_filter: str = None) -> List[str]:
    """Apply filters to the list of file paths."""
    if year_filter:
        file_paths = [f for f in file_paths if f.endswith(f"{year_filter}.CSV")]
    if uf_filter:
        file_paths = [f for f in file_paths if f"_{uf_filter.upper()}_" in f]
    if station_filter:
        file_paths = [f for f in file_paths if f"_{station_filter.upper()}_" in f]
    return file_paths


def preprocess_inmet_data(file_paths: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Read, preprocess, and merge INMET climate data from files."""
    output_dfs = []
    failed_files = []

    for file_path in file_paths:
        try:
            file_info = extract_inmet_data(file_path)
            data_df = extract_inmet_data(file_path)
            for key, value in file_info.items():
                data_df[key] = value
            data_df = data_df[EXPECTED_INMET_INFO_KEYS + EXPECTED_INMET_DATA_COLS]
            output_dfs.append(data_df)
        except Exception as e:
            failed_files.append(f"{os.path.basename(file_path)} - {e}")

    output_df = pd.concat(output_dfs, axis=0)

    if output_df[EXPECTED_INMET_DATA_COLS].isna().all(axis=0).all():
        raise Exception("Only null values found.")

    output_df['data'] = pd.to_datetime(output_df['data'], errors='coerce').dt.strftime('%Y-%m-%d')
    output_df['data_hora'] = pd.to_datetime(output_df['data'] + ' ' + output_df['hora'])

    output_df = output_df.sort_values(by='data_hora')
    for info_col in EXPECTED_INMET_INFO_KEYS:
        output_df[info_col] = output_df[info_col].ffill().bfill()

    for col in INMET_CLIMATE_DATA_COLS:
        interpolacao_col = f"{col}_interpolacao"
        if interpolacao_col not in output_df.columns:
            output_df[interpolacao_col] = False
        missing_before = output_df[col].isna()
        output_df[col] = output_df[col].interpolate(method='linear', limit_direction='both')
        output_df[interpolacao_col] = output_df[interpolacao_col] | missing_before & output_df[col].notna()
        output_df[col] = round(output_df[col], 2)

    return output_df, failed_files


def save_inmet_data(output_df: pd.DataFrame, failed_files: List[str], output_file_name: str = None) -> None:
    """Save preprocessed data and failed file information."""
    output_file_name = output_file_name or "merged_data.csv"
    failed_file_name = f"{os.path.splitext(output_file_name)[0]}_failed_files.txt"

    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)

    output_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, output_file_name), index=False)
    if failed_files:
        with open(os.path.join(PREPROCESSED_DATA_PATH, failed_file_name), 'w') as file:
            file.write('\n'.join(failed_files))


def read_inmet_climate_data(
        year_filter: int = None,
        uf_filter: str = None,
        station_filter: str = None,
        output_file_name: str = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Read, filter, preprocess, and save INMET climate data."""
    file_paths = get_all_file_paths(RAW_DATA_PATH, "CSV")
    file_paths = filter_file_paths(file_paths, year_filter, uf_filter, station_filter)

    output_df, failed_files = preprocess_inmet_data(file_paths)
    save_inmet_data(output_df, failed_files, output_file_name)

    return output_df, failed_files


def extract_inmet_all_stations_info(output_file_name: str = "station_info.csv") -> pd.DataFrame:
    """Extract and save metadata for all stations."""
    file_paths = get_all_file_paths(PREPROCESSED_DATA_PATH, "csv")
    station_info = pd.DataFrame()

    for file_path in file_paths:
        metadata = pd.read_csv(file_path, low_memory=False)
        try:
            station_info = pd.concat([station_info, metadata[EXPECTED_INMET_INFO_KEYS].iloc[0]], axis=0)
        except Exception as e:
            print(file_path)
            raise e

    save_inmet_data(station_info, [], output_file_name)
    return station_info
