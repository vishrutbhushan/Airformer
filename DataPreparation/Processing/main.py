from config import logger
from station_processing import process_all_stations
from model_preparation import prepare_data_for_model
from spatial_partitioning import create_dartboard_partitions
from wind_bias_generation import create_wind_bias_dataset
import numpy as np

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
    print("Generating wind bias matrices")
    # Load all splits and compute wind bias for each
    for split in ['train', 'val', 'test']:
        data = np.load(f"./Dataset/INDIAN_AIR/{split}.npz")
        create_wind_bias_dataset(data['x'], f"./Dataset/INDIAN_AIR/{split}_wind_bias", wind_speed_idx=12, wind_direction_idx=13)
    print("Preprocessing complete.")
