from urllib3.exceptions import NotOpenSSLWarning

from src.data_preprocessing.inmet_preprocessing import process_and_save_inmet_climate_data
from src.data_preprocessing.inpe_hotspots_daily_preprocessing import process_and_save_inpe_hotspots_daily_data
from src.data_preprocessing.merge_preprocessing import merge_and_save_data
from src.forecasting.forecaster import train_and_save_forecasting_data, predict

import pandas as pd

import warnings
import os
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


if __name__ == "__main__":
    print("Reading merged_df")
    merged_df = pd.read_csv("data/preprocessed_data/merged/merged_data.csv")

    print("Making predictions")
    predict_df = predict(merged_df)
    predict_df.to_csv("data/preprocessed_data/predictions_7_days.csv", index=False)
    