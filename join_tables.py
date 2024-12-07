import numpy as np
import pandas as pd
from src.utils import calculate_angstrom_risk, calculate_telicyn_risk

COLUMNS = [
    'codigo_estacao', 'data', 'precipitacao_total', 'pressao_atmosferica',
    'temperatura_ar', 'umidade_relativa', 'velocidade_vento',
    'latitude', 'longitude', 'bioma', 'regiao', 'uf', 'estacao',
    'temperatura_ponto_orvalho', 'dias_sem_chuva', 'angstrom_index',
    'angstrom_risk', 'telicyn_index', 'telicyn_risk', 'distancia_estacao_mais_proxima',
    'foco_incendio', 'latitude_foco_incendio', 'longitude_foco_incendio',
    'altitude'
]

def correct_relative_humidity(relative_humidity):
    if not (0 <= relative_humidity <= 100):
        return max(0, min(relative_humidity, 100))
    else:
        return relative_humidity

        

def calculate_ts(temperature, relative_humidity_percentage):
    # relative_humidity_percentage = correct_relative_humidity(relative_humidity_percentage)
    dew_point = temperature - ((100 - relative_humidity_percentage) / 5.0)
    return dew_point

def calculate_dias_sem_chuva(precipitation_series):
    dias_sem_chuva = []
    count = 0
    for precip in precipitation_series:
        if precip <= 2.5:
            count += 1
        else:
            count = 0
        dias_sem_chuva.append(count)
    return dias_sem_chuva


data = pd.read_csv("data/preprocessed_data/merged/merged_data.csv")
predictions = pd.read_csv("data/preprocessed_data/predictions_7_days.csv")
df_pivot = predictions.pivot(index=["codigo_estacao", "data"], columns="variavel", values="previsao").reset_index()
codigo_estacao_mapping = data[["codigo_estacao", "latitude", "longitude", "bioma", "regiao", "uf", "estacao"]].drop_duplicates()
merged_df = df_pivot.merge(codigo_estacao_mapping, on="codigo_estacao", how="left")
merged_df["umidade_relativa"] = merged_df["umidade_relativa"].apply(lambda x: correct_relative_humidity(x))
merged_df["temperatura_ponto_orvalho_13"] = merged_df.apply(lambda x: calculate_ts(x.temperatura_ar, x.umidade_relativa), axis=1)
merged_df["dias_sem_chuva"] = calculate_dias_sem_chuva(merged_df["precipitacao_total"])
merged_df = merged_df.rename(
    columns={"umidade_relativa": "umidade_relativa_13", "temperatura_ar": "temperatura_ar_13"}
)
merged_df = calculate_angstrom_risk(merged_df)
merged_df = calculate_telicyn_risk(merged_df)

merged_df = merged_df.rename(
    columns={"umidade_relativa_13": "umidade_relativa", "temperatura_ar_13": "temperatura_ar"}
)
merged_df["is_prediction"] = 1
df_combined = pd.concat([data[COLUMNS], merged_df], ignore_index=True)
df_combined["is_prediction"] = df_combined["is_prediction"].fillna(0)
df_combined.to_csv("data/preprocessed_data/predictions_combined_with_risks.csv", index=False)