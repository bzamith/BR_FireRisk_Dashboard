from src.data_preprocessing.inmet_preprocessing import get_inmet_stations_list, read_inmet_climate_data


def preprocess_inmet_per_station():
    """Process INMET climate data for each station and handle errors."""
    errors = []
    stations = get_inmet_stations_list()
    total_stations = len(stations)

    print(f"Processing {total_stations} stations...")

    for i, station in enumerate(stations, start=1):
        try:
            read_inmet_climate_data(
                station_filter=station,
                output_file_name=f"merged_{station}.csv"
            )
        except ValueError:
            errors.append(station)

        if i % 10 == 0:
            print(f"Processed {round(i * 100 / total_stations, 2)}%...")

    print("Errors encountered with the following stations:")
    print(errors)


if __name__ == "__main__":
    preprocess_inmet_per_station()
