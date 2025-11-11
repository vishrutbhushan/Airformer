import pandas as pd
from config import logger, CORE_FEATURES, LIMITS, DROP_COLS, START_DATE, END_DATE
from data_loading import merge_and_standardize_features, interpolate_station_data

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
    for feature in CORE_FEATURES:
        if feature not in df.columns:
            df[feature] = pd.NA
    df, interpolated, capped = interpolate_station_data(df)
    for col, (min_val, max_val) in LIMITS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').clip(min_val, max_val)
    keep_cols = ['datetime'] + [col for col in ['latitude', 'longitude'] if col in df.columns] + CORE_FEATURES
    df = df[keep_cols].dropna(subset=['datetime'])
    logger.debug(f"Final station shape: {df.shape}, interpolated: {interpolated}, capped: {capped}")
    return df, interpolated, capped

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
    total_interpolated = 0
    total_capped = 0
    removed_missing_features = 0
    for i, input_path in enumerate(files):
        station_code = os.path.splitext(os.path.basename(input_path))[0]
        try:
            df = pd.read_csv(input_path)
            result, interpolated, capped = clean_station(df, station_code, spatial_coords)
            result = result.set_index('datetime').sort_index()
            resampled = result.resample('3h').mean()
            resampled = resampled.reindex(expected_timestamps)
            # Ensure static spatial coordinates are present after reindexing
            lat, lon = spatial_coords[station_code]
            resampled['latitude'] = lat
            resampled['longitude'] = lon
            # Log % missing timesteps for this station
            missing_timesteps = resampled[CORE_FEATURES].isna().all(axis=1).sum()
            percent_missing = (missing_timesteps / len(expected_timestamps)) * 100
            has_all_features = all(resampled[feature].notna().any() if feature in resampled.columns else False for feature in CORE_FEATURES)
            if not has_all_features:
                removed_missing_features += 1
                logger.debug(f"Station {station_code}: Removed due to missing core features")
                continue
            logger.info(f"Station {station_code}: {missing_timesteps}/{len(expected_timestamps)} timesteps missing ({percent_missing:.2f}%) after reindexing.")
            for feature in CORE_FEATURES:
                if feature in resampled.columns:
                    resampled[feature] = resampled[feature].interpolate(method='linear', limit=8)
                    resampled[feature] = resampled[feature].ffill(limit=16).bfill(limit=16)
                    if resampled[feature].isna().any():
                        if feature in LIMITS:
                            min_val, max_val = LIMITS[feature]
                            default_val = (min_val + max_val) / 2
                            resampled[feature] = resampled[feature].fillna(default_val)
                        else:
                            resampled[feature] = resampled[feature].fillna(0.0)
            resampled.reset_index(inplace=True)
            resampled.rename(columns={'index': 'datetime'}, inplace=True)
            station_data.append((resampled, station_code))
            total_interpolated += interpolated
            total_capped += capped
        except Exception as e:
            logger.error(f"Station {station_code}: Processing error - {e}")
    logger.info(f"Successfully processed {len(station_data)} stations")
    logger.info("Saving processed station files")
    for df, station_code in station_data:
        output_path = os.path.join(output_folder, f"{station_code}.csv")
        df.to_csv(output_path, index=False)
    station_dfs = [df for df, _ in station_data]
    total_datapoints = len(station_dfs) * len(expected_timestamps) * len(CORE_FEATURES)
    null_count = sum(df[CORE_FEATURES].isna().sum().sum() for df in station_dfs)
    data_coverage = (1 - null_count / total_datapoints) * 100
    logger.info(f"Stations processed: {len(station_dfs)}/{len(files)} ({len(station_dfs)/len(files)*100:.1f}%)")
    logger.info(f"Stations removed (missing features): {removed_missing_features}")
    logger.info(f"Total datapoints: {total_datapoints:,}")
    logger.info(f"Null values remaining: {null_count:,}")
    logger.info(f"Data coverage: {data_coverage:.2f}%")
    logger.info(f"Total interpolated values: {total_interpolated:,}")
    logger.info(f"Total capped values: {total_capped:,}")
    return station_dfs
