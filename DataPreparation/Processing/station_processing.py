import pandas as pd
from config import logger, BASE_FEATURES, CORE_FEATURES, CYCLIC_FEATURES, LIMITS, DROP_COLS, START_DATE, END_DATE, FEATURE_MAPPING
from cyclic_features import add_cyclic_features, get_cyclic_feature_columns
from null_imputation import impute_station_features

def load_spatial_coordinates(spatial_file):
    """Load spatial coordinates from CSV."""
    logger.info(f"Loading spatial coordinates from: {spatial_file}")
    df = pd.read_csv(spatial_file)
    coords_dict = {row['file_name']: (row['latitude'], row['longitude']) for _, row in df.iterrows()}
    logger.info(f"Loaded coordinates for {len(coords_dict)} stations")
    return coords_dict

def merge_and_standardize_features(df):
    """Merge and standardize feature columns based on FEATURE_MAPPING."""
    new_cols = {}
    merged_count = 0
    for std_name, raw_names in FEATURE_MAPPING.items():
        present = [col for col in raw_names if col in df.columns]
        if present:
            vals = df[present].apply(pd.to_numeric, errors='coerce')
            new_cols[std_name] = vals.mean(axis=1, skipna=True)
            if len(present) > 1:
                merged_count += 1
    out_df = pd.DataFrame(new_cols)
    for col in ['From Date', 'To Date']:
        if col in df.columns:
            out_df[col] = df[col]
    if merged_count > 0:
        logger.debug(f"Merged {merged_count} feature groups with multiple columns")
    return out_df

def load_and_standardize_station(input_path, station_code):
    """
    Step 1: Load raw CSV, standardize features, filter by date range, and check ALL core features present.
    Returns df, or None if station is missing core features.
    """
    logger.debug(f"Step 1: Loading station {station_code}")
    df = pd.read_csv(input_path)
    
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors='ignore')
    df = merge_and_standardize_features(df)
    
    if 'From Date' in df.columns:
        df['datetime'] = pd.to_datetime(df['From Date'], errors='coerce')
    df = df.drop(columns=['From Date', 'To Date'], errors='ignore')
    
    # Filter to date range
    df = df[(df['datetime'] >= START_DATE) & (df['datetime'] <= END_DATE)].copy()
    logger.debug(f"Station {station_code}: Date filtered to {len(df)} records")
    
    # Check if ALL BASE features are present (cyclic features are added later)
    missing = [f for f in BASE_FEATURES if f not in df.columns]
    if missing:
        logger.warning(f"Station {station_code}: Skipping because missing required base features: {missing}")
        return None
    
    return df

def add_spatial_features(df, station_code, spatial_coords):
    """
    Step 5 (formerly Step 2): Add spatial coordinates (latitude, longitude) AFTER all processing.
    This is done at the end since coordinates are static and don't need resampling/imputation.
    """
    logger.debug(f"Step 5: Adding spatial coords for {station_code}")
    
    if station_code in spatial_coords:
        lat, lon = spatial_coords[station_code]
        df['latitude'] = lat
        df['longitude'] = lon
        logger.debug(f"Station {station_code}: Added coordinates ({lat:.4f}, {lon:.4f})")
    else:
        logger.warning(f"Station {station_code}: No spatial coordinates found in database")
    
    return df

def resample_to_3h(df, station_code, expected_timestamps):
    """
    Step 2: Set datetime index, resample to 3-hourly, reindex to expected grid.
    Creates rows for ALL expected timestamps (including missing ones).
    Returns df, or None if data coverage is less than 60%.
    """
    logger.debug(f"Step 2: Resampling to 3-hourly for {station_code}")
    
    df = df.set_index('datetime').sort_index()
    df = df.resample('3h').mean()
    
    # Reindex to full 4-year grid (creates NaN rows for missing timesteps)
    df = df.reindex(expected_timestamps)
    logger.debug(f"Station {station_code}: Reindexed to {len(expected_timestamps)} timesteps")
    
    # Count how many timesteps are completely empty after reindexing
    empty_timesteps = df.isna().all(axis=1).sum()
    present_timesteps = len(expected_timestamps) - empty_timesteps
    data_coverage_pct = (present_timesteps / len(expected_timestamps)) * 100
    
    logger.info(f"Station {station_code}: After resampling, {empty_timesteps} out of {len(expected_timestamps)} timesteps have no data (coverage: {data_coverage_pct:.1f}%)")
    
    # Skip station if data coverage is less than 60%
    if data_coverage_pct < 60:
        logger.warning(f"Station {station_code}: Skipping due to insufficient data coverage ({data_coverage_pct:.1f}% < 60%)")
        return None
    
    # Check if individual base features have ANY data before proceeding
    # (cyclic features haven't been added yet, so only check BASE_FEATURES)
    completely_empty_features = []
    for feature in BASE_FEATURES:
        if feature in df.columns:
            if df[feature].notna().sum() == 0:  # Entire feature is null
                completely_empty_features.append(feature)
    
    if completely_empty_features:
        logger.warning(f"Station {station_code}: Skipping because following base features are completely empty: {completely_empty_features}")
        return None
    
    return df

def add_cyclic_features_step(df, station_code):
    """
    Step 3: Add cyclic features (hour, day_of_week, month).
    Must be done BEFORE imputation so Kalman can use temporal context.
    """
    logger.debug(f"Step 3: Adding cyclic features for {station_code}")
    df = df.reset_index()
    df = df.rename(columns={'index': 'datetime'})
    df = add_cyclic_features(df, datetime_col='datetime')
    logger.debug(f"Station {station_code}: Added 6 cyclic features")
    return df

def apply_imputation(df, station_code):
    """
    Step 4: Apply comprehensive imputation (Kalman filter).
    Handles missing individual features AND missing entire timesteps.
    Now has cyclic features to help predict missing values.
    """
    logger.debug(f"Step 5: Comprehensive imputation for {station_code}")
    df = impute_station_features(df, station_code)
    return df

def process_all_stations(input_folder, output_folder, spatial_file):

    import glob, os
    
    logger.info("="*60)
    logger.info("STARTING STATION PROCESSING PIPELINE")
    logger.info("="*60)
    
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.csv"))
    spatial_coords = load_spatial_coordinates(spatial_file) if spatial_file and os.path.exists(spatial_file) else {}
    expected_timestamps = pd.date_range(start=START_DATE, end=END_DATE, freq='3h')
    
    logger.info(f"Found {len(files)} station files")
    logger.info(f"Expected timestamp grid: {len(expected_timestamps)} 3-hourly intervals")
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Core features: {len(CORE_FEATURES)}")
    
    station_data = []
    skipped_no_features = 0
    skipped_low_coverage = 0
    
    for i, input_path in enumerate(files):
        station_code = os.path.splitext(os.path.basename(input_path))[0]
        
        try:
            # STEP 1: Load, standardize, filter dates, check ALL base features
            df = load_and_standardize_station(input_path, station_code)
            if df is None:
                skipped_no_features += 1
                continue
            
            # STEP 2: Resample to 3h (also checks data coverage >= 60%)
            df = resample_to_3h(df, station_code, expected_timestamps)
            if df is None:
                skipped_low_coverage += 1
                continue
            
            # STEP 3: Add cyclic features (BEFORE imputation for Kalman context)
            df = add_cyclic_features_step(df, station_code)
            
            # STEP 4: Comprehensive imputation (with cyclic features as context)
            df = apply_imputation(df, station_code)
            
            # STEP 5: Add spatial coordinates (last, since they're static and don't need processing)
            df = add_spatial_features(df, station_code, spatial_coords)
            
            # STEP 6: Save processed station file immediately
            output_path = os.path.join(output_folder, f"{station_code}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Station {station_code}: Successfully processed and saved ({i+1}/{len(files)})")
            logger.debug(f"Saved to: {output_path}")
            
            station_data.append((df, station_code))
            
        except Exception as e:
            logger.error(f"Station {station_code} processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    # Summary statistics
    station_dfs = [df for df, _ in station_data]
    total_datapoints = len(station_dfs) * len(expected_timestamps) * len(CORE_FEATURES)
    null_count = sum(df[CORE_FEATURES].isna().sum().sum() for df in station_dfs)
    data_coverage = (1 - null_count / total_datapoints) * 100 if total_datapoints > 0 else 0
    
    logger.info("="*60)
    logger.info("FINAL STATISTICS")
    logger.info("="*60)
    logger.info(f"Total stations in input: {len(files)}")
    logger.info(f"Successfully processed: {len(station_dfs)}")
    logger.info(f"Skipped (missing features): {skipped_no_features}")
    logger.info(f"Skipped (low data coverage <60%): {skipped_low_coverage}")
    logger.info(f"Data coverage: {data_coverage:.2f}%")
    logger.info(f"Total datapoints: {total_datapoints:,}")
    logger.info(f"Null values remaining: {null_count:,}")
    logger.info("="*60)
    
    return station_dfs
