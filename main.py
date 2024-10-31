from urllib3.exceptions import NotOpenSSLWarning

from src.data_preprocessing.inmet_preprocessing import process_and_save_inmet_climate_data
from src.data_preprocessing.inpe_hotspots_daily_preprocessing import process_and_save_inpe_hotspots_daily_data
from src.data_preprocessing.merge_preprocessing import merge_and_save_data
from src.forecasting.forecaster import train_and_save_forecasting_data

import pandas as pd

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


if __name__ == "__main__":
    # stations_info_df, inmet_df = process_and_save_inmet_climate_data()
    # inpe_hotspots_daily_df = process_and_save_inpe_hotspots_daily_data(stations_info_df)
    # merged_df = merge_and_save_data(inmet_df, inpe_hotspots_daily_df)

    print("Reading merged_df")
    merged_df = pd.read_csv("data/preprocessed_data/merged/merged_data.csv")

    print("Training models")
    train_and_save_forecasting_data(merged_df)

    