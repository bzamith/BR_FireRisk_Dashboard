import glob
import os
from typing import List

import numpy as np

import pandas as pd


def get_all_file_paths(directory: str, extension: str) -> List[str]:
    """Get all file paths with a given extension from a directory."""
    return glob.glob(os.path.join(directory, f"*.{extension}"))


def calculate_angstrom_risk(df: pd.DataFrame) -> pd.DataFrame:
    def __calculate_angstrom_risk_rate(index: float):
        if index > 4.0:
            return 'Improvável'
        elif index >= 2.5:
            return 'Desfavorável'
        elif index >= 2.0:
            return 'Favorável'
        else:
            return 'Provável'

    df['angstrom_index'] = (df['umidade_relativa_13'] / 20) + ((df['temperatura_ar_13'] - 27) / 10)
    df['angstrom_risk'] = df['angstrom_index'].apply(__calculate_angstrom_risk_rate)
    df['angstrom_interpolacao'] = (df['umidade_relativa_interpolacao_13'] | df['temperatura_ar_interpolacao_13'])

    return df


def calculate_telicyn_risk(df: pd.DataFrame) -> pd.DataFrame:
    def __calculate_telicyn_risk_rate(index: float):
        if index <= 2.0:
            return 'Nenhum'
        elif index <= 3.5:
            return 'Pequeno'
        elif index <= 5.0:
            return 'Medio'
        else:
            return 'Alto'

    df['telicyn_index'] = None
    df['telicyn_risk'] = None
    df['telicyn_interpolacao'] = (df['temperatura_ar_interpolacao_13'] | df['temperatura_ponto_orvalho_13'])
    for i, row in df.iterrows():
        diff = row['temperatura_ar_13'] - row['temperatura_ponto_orvalho_13']
        df.at[i, 'telicyn_index'] = 0.0 if diff <= 0 else np.log10(diff)
        if i - 1 > 0 and row['dias_sem_chuva'] > 0:
            df.at[i, 'telicyn_index'] += df.at[i-1, 'telicyn_index']
            df.at[i, 'telicyn_interpolacao'] |= df.at[i-1, 'telicyn_interpolacao']

    df['telicyn_risk'] = df['telicyn_index'].apply(__calculate_telicyn_risk_rate)
    df['telicyn_index'] = df['telicyn_index'].astype(float)

    return df
