from src.data_preprocessing.inmet_preprocessing import (
    read_inmet_climate_data
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


if __name__ == "__main__":
    read_inmet_climate_data()
