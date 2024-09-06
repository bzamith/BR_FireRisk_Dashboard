from src.data_preprocessing.inmet_preprocessing import process_and_save_inmet_climate_data
from src.data_preprocessing.inpe_hotspots_daily_preprocessing import process_and_save_inpe_hotspots_daily_data

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    # stations_info_df, inmet_df = process_and_save_inmet_climate_data()
    import pandas as pd
    stations_info_df = pd.read_csv("data/preprocessed_data/INMET/inmet_stations_info.csv")
    inpe_hotspots_daily_df = process_and_save_inpe_hotspots_daily_data(stations_info_df)
