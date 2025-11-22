import pandas as pd
from config import logger, CORE_FEATURES, BASE_FEATURES, CYCLIC_FEATURES, LIMITS, DROP_COLS, START_DATE, END_DATE
from data_loading import merge_and_standardize_features
from cyclic_features import add_cyclic_features
import numpy as np
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


def _knn_impute_temporal(df, features, cyclic_cols, k=5, max_gap=16):
    """Impute missing values using KNN for small gaps and mean for larger gaps."""
    df = df.copy()
    
    # Filter to columns that exist in dataframe
    cyclic_cols = [col for col in cyclic_cols if col in df.columns]
    imputable_features = [f for f in features if f in df.columns]
    
    if not imputable_features or not cyclic_cols:
        return df
    
    # Find timesteps where all features are missing
    completely_empty = df[imputable_features].isna().all(axis=1)
    empty_idx = np.where(completely_empty.values)[0]
    
    if len(empty_idx) > 0:
        # Find gaps <= max_gap for KNN imputation
        knn_idx = []
        gap_size = 1
        for i in range(len(empty_idx)):
            is_consecutive = i > 0 and empty_idx[i] == empty_idx[i-1] + 1
            gap_size = gap_size + 1 if is_consecutive else 1
            if gap_size <= max_gap:
                knn_idx.append(empty_idx[i])
        
        # KNN imputation for small gaps
        if knn_idx:
            X_cyclic = df[cyclic_cols].values
            n_neighbors = min(k, len(df) - 1)
            knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
            knn.fit(X_cyclic)
            
            for idx in knn_idx:
                _, neighbors = knn.kneighbors([X_cyclic[idx]])
                neighbors = neighbors[0]
                
                for col in imputable_features:
                    valid_values = df.iloc[neighbors][col].dropna().values
                    if len(valid_values) > 0:
                        df.iloc[idx, df.columns.get_loc(col)] = np.nanmean(valid_values)
    
    # Mean imputation for remaining nulls
    for col in imputable_features:
        if df[col].isna().any():
            col_mean = df[col].mean()
            if not np.isnan(col_mean):
                df.loc[df[col].isna(), col] = col_mean
    
    return df

def clean_station(df, station_code=None, spatial_coords=None):
    logger.debug(f"Cleaning station: {station_code}")
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    df = merge_and_standardize_features(df)
    if 'From Date' in df.columns:
        df['datetime'] = pd.to_datetime(df['From Date'], errors='coerce')
    df = df.drop(columns=['From Date', 'To Date'], errors='ignore')
    date_filtered = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)].copy()
    logger.debug(f"Date filtering: {len(df)} -> {len(date_filtered)} records")
    df = date_filtered
    if station_code and spatial_coords and station_code in spatial_coords:
        lat, lon = spatial_coords[station_code]
        df['latitude'] = lat
        df['longitude'] = lon
        logger.debug(f"Added coordinates: ({lat:.4f}, {lon:.4f})")
    for feature in BASE_FEATURES:
        if feature not in df.columns:
            df[feature] = pd.NA
    df = df[['datetime'] + [c for c in ['latitude', 'longitude'] if c in df.columns] + BASE_FEATURES].dropna(subset=['datetime'])
    for col, (min_val, max_val) in LIMITS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(min_val, max_val)
    logger.debug(f"Final station shape: {df.shape}")
    return df

def process_all_stations(input_folder, output_folder, spatial_file):
    import glob, os
    logger.info("Processing stations")
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    from data_loading import load_spatial_coordinates
    spatial_coords = load_spatial_coordinates(spatial_file) if spatial_file and os.path.exists(spatial_file) else {}
    expected_timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq='3h')
    logger.info(f"Expected timestamp grid: {len(expected_timestamps)} 3-hourly intervals")
    logger.info(f"Processing {len(files)} station files")
    station_data = []
    removed_missing_features = 0
    for i, input_path in enumerate(files):
        station_code = os.path.splitext(os.path.basename(input_path))[0]
        try:
            df = pd.read_csv(input_path)
            result = clean_station(df, station_code, spatial_coords)
            result = result.set_index('datetime').sort_index()
            resampled = result.resample('3h').mean()
            resampled = resampled.reindex(expected_timestamps)
            # Ensure static spatial coordinates are present after reindexing
            lat, lon = spatial_coords[station_code]
            resampled['latitude'] = lat
            resampled['longitude'] = lon
            # Log % missing timesteps for this station
            missing_timesteps = resampled[BASE_FEATURES].isna().all(axis=1).sum()
            percent_missing = (missing_timesteps / len(expected_timestamps)) * 100
            percent_available = 100 - percent_missing
            
            if percent_available <= 60:
                removed_missing_features += 1
                # continue
            
            has_all_features = all(resampled[feature].notna().any() if feature in resampled.columns else False for feature in BASE_FEATURES)
            if not has_all_features:
                removed_missing_features += 1
                continue
            logger.info(f"Station {station_code}: {missing_timesteps}/{len(expected_timestamps)} timesteps missing ({percent_missing:.2f}%, {percent_available:.1f}% available) - KEPT")
            resampled.reset_index(inplace=True)
            resampled.rename(columns={'index': 'datetime'}, inplace=True)
            resampled = add_cyclic_features(resampled, datetime_col='datetime')
            cyclic_cols = ['hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
            features_to_impute = [f for f in BASE_FEATURES if f in resampled.columns]
            resampled = _knn_impute_temporal(resampled, features_to_impute, cyclic_cols, k=5, max_gap=16)
            station_data.append((resampled, station_code))
        except Exception as e:
            logger.error(f"Station {station_code}: Processing error - {e}")
    logger.info(f"Successfully processed {len(station_data)} stations")
    logger.info("Saving processed station files")
    for df, station_code in station_data:
        output_path = os.path.join(output_folder, f"{station_code}.csv")
        df.to_csv(output_path, index=False)
    station_dfs = [df for df, _ in station_data]
    total_datapoints = len(station_dfs) * len(expected_timestamps) * len(BASE_FEATURES)
    null_count = sum(df[BASE_FEATURES].isna().sum().sum() for df in station_dfs)
    data_coverage = (1 - null_count / total_datapoints) * 100
    logger.info(f"Stations processed: {len(station_dfs)}/{len(files)} ({len(station_dfs)/len(files)*100:.1f}%)")
    logger.info(f"Stations removed (missing features): {removed_missing_features}")
    logger.info(f"Total datapoints: {total_datapoints:,}")
    logger.info(f"Null values remaining: {null_count:,}")
    logger.info(f"Data coverage: {data_coverage:.2f}%")
    return station_dfs
