import os
import re
from typing import List

from scipy.spatial import cKDTree
from geopy.distance import geodesic

import numpy as np

import pandas as pd

from tqdm import tqdm

from ..utils import get_all_file_paths

# Paths for data directories
RAW_DATA_INPE_HOTSPOTS_DAILY_PATH = "data/raw_data/INPE_hotspots_daily"
PREPROCESSED_DATA_INPE_HOTSPOTS_DAILY_PATH = "data/preprocessed_data/INPE_hotspots_daily"

EXPECTED_INPE_HOTSPOTS_DAILY_DATA_COLS = [
    'latitude', 'longitude', 'data_hora',
    'satelite', 'municipio', 'uf', 'bioma'
]


def __extract_inpe_hotspots_daily_data(file_path: str) -> pd.DataFrame:
    """Extract and preprocess data from an INPE Hotspots Daily file."""
    raw_df = pd.read_csv(file_path)
    raw_df.replace(['-999', -999, -999.0], np.nan, inplace=True)

    uf_mapping = {
        'ACRE': 'AC',
        'ALAGOAS': 'AL',
        'AMAPÁ': 'AP',
        'AMAZONAS': 'AM',
        'BAHIA': 'BA',
        'CEARÁ': 'CE',
        'DISTRITO FEDERAL': 'DF',
        'ESPÍRITO SANTO': 'ES',
        'GOIÁS': 'GO',
        'MARANHÃO': 'MA',
        'MATO GROSSO': 'MT',
        'MATO GROSSO DO SUL': 'MS',
        'MINAS GERAIS': 'MG',
        'PARÁ': 'PA',
        'PARAÍBA': 'PB',
        'PARANÁ': 'PR',
        'PERNAMBUCO': 'PE',
        'PIAUÍ': 'PI',
        'RIO DE JANEIRO': 'RJ',
        'RIO GRANDE DO NORTE': 'RN',
        'RIO GRANDE DO SUL': 'RS',
        'RONDÔNIA': 'RO',
        'RORAIMA': 'RR',
        'SANTA CATARINA': 'SC',
        'SÃO PAULO': 'SP',
        'SERGIPE': 'SE',
        'TOCANTINS': 'TO'
    }

    column_mapping = {
        'lat': 'latitude',
        'lon': 'longitude',
        'data_hora_gmt': 'data_hora',
        'satelite': 'satelite',
        'municipio': 'municipio',
        'estado': 'uf',
        'bioma': 'bioma'
    }

    df = raw_df[list(column_mapping.keys())]
    df = df.rename(columns=column_mapping)
    df['uf'] = df['uf'].map(uf_mapping)

    if sorted(df.columns) != sorted(EXPECTED_INPE_HOTSPOTS_DAILY_DATA_COLS):
        raise Exception(
            f"{os.path.basename(file_path)}: Missing columns INPE Hotspots Daily data. Found: {list(df.columns)}\n, expected: {sorted(EXPECTED_INPE_HOTSPOTS_DAILY_DATA_COLS)}")

    return df


def __extract_inpe_hotspots_daily_months() -> List[str]:
    """Extract unique months from file names."""
    file_paths = get_all_file_paths(RAW_DATA_INPE_HOTSPOTS_DAILY_PATH, "csv")
    months = {
        re.search(r"_20\d{4}", os.path.basename(path)).group(0)[1:]
        for path in file_paths
        if re.search(r"_20\d{4}", os.path.basename(path))
    }
    return sorted(list(months))


def __find_closest_station(inpe_hotspots_daily_df: pd.DataFrame, inmet_stations_info_df: pd.DataFrame, month: str) -> pd.DataFrame:
    def __haversine_distance(lat1, lon1, lat2, lon2):
        return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    # Create a KDTree for fast nearest-neighbor lookup
    station_coords = inmet_stations_info_df[['latitude', 'longitude']].to_numpy()
    station_tree = cKDTree(station_coords)

    # Initialize columns for closest station and distance
    inpe_hotspots_daily_df['codigo_estacao_mais_proxima'] = None
    inpe_hotspots_daily_df['distancia_estacao_mais_proxima'] = None

    # Iterate over hotspots to find closest station
    for i, target_row in tqdm(inpe_hotspots_daily_df.iterrows(), total=inpe_hotspots_daily_df.shape[0], desc=f"Processing hotspots for {month}"):
        target_coords = (target_row['latitude'], target_row['longitude'])
        _, closest_index = station_tree.query(target_coords)

        # Calculate the precise Haversine distance to the closest station
        closest_station_coords = station_coords[closest_index]
        min_distance = __haversine_distance(target_coords[0], target_coords[1], closest_station_coords[0], closest_station_coords[1])

        # Update DataFrame with closest station and distance
        inpe_hotspots_daily_df.at[i, 'codigo_estacao_mais_proxima'] = inmet_stations_info_df.at[closest_index, 'codigo_estacao']
        inpe_hotspots_daily_df.at[i, 'distancia_estacao_mais_proxima'] = min_distance

    return inpe_hotspots_daily_df


def __preprocess_inpe_hotspots_daily_data(file_paths: List[str], inmet_stations_info_df: pd.DataFrame, month: str) -> pd.DataFrame:
    """Read, preprocess, and merge INPE hotspots daily data from files."""
    dfs = []

    for file_path in file_paths:
        data_df = __extract_inpe_hotspots_daily_data(file_path)
        data_df = data_df[EXPECTED_INPE_HOTSPOTS_DAILY_DATA_COLS]
        dfs.append(data_df)

    df = pd.concat(dfs, axis=0)
    df = df.drop_duplicates().reset_index(drop=True)

    if df[EXPECTED_INPE_HOTSPOTS_DAILY_DATA_COLS].isna().all(axis=0).all():
        raise Exception(f"{os.path.basename(file_path)}: Only null values found.")

    df = df.dropna(subset=['data_hora'])
    if df.shape[0] == 0:
        raise Exception(f"{os.path.basename(file_path)}: No rows left after droping the data_hora NaNs.")

    df['data_hora'] = pd.to_datetime(df['data_hora'])
    df['data'] = df['data_hora'].dt.strftime('%Y-%m-%d')
    df = df.drop(columns=['data_hora'])

    df = df.groupby(['data', 'municipio', 'uf', 'bioma']).agg({
        'latitude': 'mean',
        'longitude': 'mean',
    }).reset_index()

    df =__find_closest_station(df, inmet_stations_info_df, month)

    return df.round(2)


def process_and_save_inpe_hotspots_daily_data(inmet_stations_info_df: pd.DataFrame) -> pd.DataFrame:
    """Read, filter, preprocess, and save INPE Hotspots Daily data."""
    months = __extract_inpe_hotspots_daily_months()
    file_paths = get_all_file_paths(RAW_DATA_INPE_HOTSPOTS_DAILY_PATH, "csv")
    total_months = len(months)

    inpe_hotspots_daily_df = []
    output_dir = f"{PREPROCESSED_DATA_INPE_HOTSPOTS_DAILY_PATH}/per_month"
    os.makedirs(output_dir, exist_ok=True)

    for month in tqdm(months, total=total_months, desc="Processing INPE Hotspots Daily"):
        month_file_paths = [f for f in file_paths if f"_{month}." in f]
        month_inpe_hotspots_daily_df = __preprocess_inpe_hotspots_daily_data(month_file_paths, inmet_stations_info_df, month)

        month_inpe_hotspots_daily_df.to_csv(f"{output_dir}/inpe_hotspots_daily_{month}.csv", index=False)

        inpe_hotspots_daily_df.append(month_inpe_hotspots_daily_df)

    print("Saving INPE hotsposts daily data...")
    inpe_hotspots_daily_df = pd.concat(inpe_hotspots_daily_df, axis=0)
    inpe_hotspots_daily_df.to_csv(f"{PREPROCESSED_DATA_INPE_HOTSPOTS_DAILY_PATH}/inpe_hotspot_daily_data.csv", index=False)
    
    return inpe_hotspots_daily_df
