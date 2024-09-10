from src.data_preprocessing.inmet_preprocessing import process_and_save_inmet_climate_data
from src.data_preprocessing.inpe_hotspots_daily_preprocessing import process_and_save_inpe_hotspots_daily_data
from src.data_preprocessing.merge_preprocessing import merge_and_save_data

import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    # stations_info_df, inmet_df = process_and_save_inmet_climate_data()
    # inpe_hotspots_daily_df = process_and_save_inpe_hotspots_daily_data(stations_info_df)

    stations_info_df = pd.read_csv("data/preprocessed_data/INMET/inmet_stations_info.csv")
    inmet_df = pd.read_csv("data/preprocessed_data/INMET/inmet_data.csv")
    inpe_hotspots_daily_df =  pd.read_csv("data/preprocessed_data/INPE_hotspots_daily/inpe_hotspot_daily_data.csv")

    merge_and_save_data(inmet_df, inpe_hotspots_daily_df)
