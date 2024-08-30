import glob
import os
import re
from typing import List, Tuple

import numpy as np

import pandas as pd


# Paths for data directories
RAW_DATA_PATH = "data/raw_data/INMET"
PREPROCESSED_DATA_PATH = "data/preprocessed_data/INMET"

# Expected keys and columns
EXPECTED_INMET_INFO_KEYS = [
    'region', 'uf', 'station', 'station_code',
    'latitude', 'longitude', 'altitude', 'foundation_date'
]
EXPECTED_INMET_DATA_COLS = [
    'data', 'hora', 'precipitacao_total', 'pressao_atmosferica',
    'temperatura_ar', 'temperatura_ponto_orvalho', 'umidade_relativa', 'velocidade_vento'
]


def get_all_file_paths_from_dir(directory: str, extension: str) -> List[str]:
    """Get all file paths with a given extension from a directory."""
    return glob.glob(os.path.join(directory, f"*.{extension}"))


def get_inmet_stations_list() -> List[str]:
    """Extract and return unique INMET station codes from raw data files."""
    file_paths = get_all_file_paths_from_dir(RAW_DATA_PATH, "CSV")
    file_names = [os.path.basename(path) for path in file_paths]

    stations = {re.search(r"_A\d{3}_", name).group()[1:5] for name in file_names if re.search(r"_A\d{3}_", name)}
    return list(stations)


def extract_inmet_file_info(file_path: str) -> dict:
    """Extract metadata information from an INMET file."""
    file_info_df = pd.read_csv(file_path, encoding='latin1', nrows=9, header=None, delimiter=';', on_bad_lines='skip')
    if file_info_df.shape[1] != 2:
        raise Exception(f"File {file_path} has more than 2 columns: {file_info_df.shape}")

    file_info_df = file_info_df.T
    file_info_df.columns = file_info_df.iloc[0]
    file_info_df = file_info_df[1:].reset_index(drop=True)

    if file_info_df.shape[1] != 8:
        raise Exception(f"File {file_path} has more than 8 fields: {file_info_df.columns}")

    file_info_dict = {}
    column_mapping = {
        'REGIÃO': 'region',
        'UF': 'uf',
        'ESTAÇÃO': 'station',
        'CODIGO': 'station_code',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'ALTITUDE': 'altitude',
        'DATA DE FUNDAÇÃO': 'foundation_date'
    }

    for column in file_info_df.columns:
        column_upper = column.upper()
        key = next((k for k, v in column_mapping.items() if column_upper.startswith(k)), None)
        if key:
            value = file_info_df.at[0, column].replace(',', '.') if key in {'latitude', 'longitude', 'altitude'} else \
            file_info_df.at[0, column]
            file_info_dict[key] = float(value) if key in {'latitude', 'longitude', 'altitude'} else value

    # Fill missing expected keys with NaN
    for key in EXPECTED_INMET_INFO_KEYS:
        file_info_dict.setdefault(key, np.nan)

    return file_info_dict


def extract_inmet_file_data(file_path: str) -> pd.DataFrame:
    """Extract and preprocess data from an INMET file."""
    orig_df = pd.read_csv(file_path, encoding='latin1', skiprows=8, header=0, delimiter=';', on_bad_lines='skip')
    orig_df.replace(['-9999', -9999], np.nan, inplace=True)

    df = pd.DataFrame()
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

    for column in orig_df.columns:
        column_upper = column.upper()
        for key, value in column_mapping.items():
            if column_upper.startswith(key):
                if 'total' in value or 'pressao' in value or 'temperatura' in value or 'velocidade' in value:
                    df[value] = orig_df[column].astype(str).str.replace(',', '.').astype(float)
                else:
                    df[value] = orig_df[column]
                break

    if df.shape[1] != 8:
        raise Exception(f"Found {df.shape[1]} columns for {file_path} after preprocessing ({df.columns}). Expected 8.")

    return df


def read_inmet_climate_data(
        year_filter: int = None,
        uf_filter: str = None,
        station_filter: str = None,
        output_file_name: str = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Read, filter, and merge INMET climate data, then save to file."""
    file_paths = get_all_file_paths_from_dir(RAW_DATA_PATH, "CSV")

    # Apply filters
    if year_filter:
        file_paths = [f for f in file_paths if f.endswith(f"{year_filter}.CSV")]
    if uf_filter:
        file_paths = [f for f in file_paths if f"_{uf_filter.upper()}_" in f]
    if station_filter:
        file_paths = [f for f in file_paths if f"_{station_filter.upper()}_" in f]

    output_dfs = []
    failed_files = []

    for file_path in file_paths:
        try:
            file_info = extract_inmet_file_info(file_path)
            data_df = extract_inmet_file_data(file_path)
            for key, value in file_info.items():
                data_df[key] = value
            data_df = data_df[EXPECTED_INMET_INFO_KEYS + EXPECTED_INMET_DATA_COLS]
            output_dfs.append(data_df)
        except Exception as e:
            failed_files.append(f"{os.path.basename(file_path)} - {e}")

    output_df = pd.concat(output_dfs, axis=0) if output_dfs else pd.DataFrame()
    output_file_name = output_file_name or "merged_data.csv"
    failed_file_name = f"{os.path.splitext(output_file_name)[0]}_failed_files.txt"

    # Ensure output directory exists
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)

    # Save results
    output_df.to_csv(os.path.join(PREPROCESSED_DATA_PATH, output_file_name), index=False)
    if failed_files:
        with open(os.path.join(PREPROCESSED_DATA_PATH, failed_file_name), 'w') as file:
            file.write('\n'.join(failed_files))

    return output_df, failed_files
