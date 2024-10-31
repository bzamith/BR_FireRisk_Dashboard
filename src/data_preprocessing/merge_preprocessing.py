import os

import pandas as pd

PREPROCESSED_DATA_MERGED_PATH = "data/preprocessed_data/merged"

SELECTED_INPE_HOTSPOTS_DAILY_DATA_COLS = [
    'data', 'codigo_estacao_mais_proxima', 'distancia_estacao_mais_proxima', 'latitude', 'longitude', 'bioma'
]


def merge_and_save_data(
    inmet_df: pd.DataFrame,
    inpe_hotspots_daily_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = pd.merge(
        inmet_df,
        inpe_hotspots_daily_df[SELECTED_INPE_HOTSPOTS_DAILY_DATA_COLS].rename(
            columns={
                'codigo_estacao_mais_proxima': 'codigo_estacao',
                'latitude': 'latitude_foco_incendio',
                'longitude': 'longitude_foco_incendio'
            }
        ),
        how='left',
        on=['data', 'codigo_estacao'],
        indicator=True
    )
    merged_df = merged_df.rename(columns={'_merge': 'foco_incendio'})
    merged_df['foco_incendio'] = merged_df['foco_incendio'].map({
        'left_only': False,
        'both': True
    })

    bioma_mapping = (
        merged_df[['codigo_estacao', 'bioma', 'distancia_estacao_mais_proxima']]
        .dropna()
        .sort_values(by='distancia_estacao_mais_proxima')  # Sort by distance
        .drop_duplicates(subset=['codigo_estacao'], keep='first')  # Keep the closest station per 'codigo_estacao'
    )

    # Now map the 'bioma' column in the main dataframe 'df' based on 'codigo_estacao'
    merged_df['bioma'] = merged_df['codigo_estacao'].map(bioma_mapping.set_index('codigo_estacao')['bioma'])

    print("Saving merged data...")
    os.makedirs(PREPROCESSED_DATA_MERGED_PATH, exist_ok=True)
    merged_df.to_csv(f"{PREPROCESSED_DATA_MERGED_PATH}/merged_data.csv", index=False)

    return merged_df
