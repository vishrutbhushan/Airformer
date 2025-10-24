from config import logger
from station_processing import process_all_stations
from model_preparation import prepare_data_for_model
from spatial_partitioning import create_dartboard_partitions

if __name__ == "__main__":
    print("Preprocessing started.")
    spatial_file = "../SpatialDataGeneration/stations_with_coords.csv"
    print("Preprocessing raw station data")
    process_all_stations("../Data", "../DataPreprocessed", spatial_file)
    print("Preparing model-ready data")
    prepare_data_for_model("../DataPreprocessed", "./Dataset/INDIAN_AIR")
    print("Creating spatial partitions")
    metadata_file = "./Dataset/INDIAN_AIR/metadata.pkl"
    create_dartboard_partitions(metadata_file)
    print("Preprocessing complete.")
