import pandas as pd
from config import logger, CORE_FEATURES, LIMITS, FEATURE_MAPPING, DROP_COLS, START_DATE, END_DATE

def load_spatial_coordinates(spatial_file):
    logger.info(f"Loading spatial coordinates from: {spatial_file}")
    df = pd.read_csv(spatial_file)
    coords_dict = {row['file_name']: (row['latitude'], row['longitude']) for _, row in df.iterrows()}
    logger.info(f"Loaded coordinates for {len(coords_dict)} stations")
    return coords_dict

def merge_and_standardize_features(df):
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

def interpolate_station_data(df):
    feature_cols = [c for c in CORE_FEATURES if c in df.columns]
    original_nulls = df[feature_cols].isna().sum().sum()
    capped_count = 0
    for col in feature_cols:
        if col in LIMITS:
            min_val, max_val = LIMITS[col]
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            out_of_bounds = ((numeric_col < min_val) | (numeric_col > max_val)).sum()
            capped_count += out_of_bounds
    df = df.sort_values('datetime').reset_index(drop=True)
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].notna().any():
            df[col] = df[col].interpolate(method='linear', limit_direction='both')
            df[col] = df[col].ffill().bfill()
    remaining_nulls = df[feature_cols].isna().sum().sum()
    interpolated_count = original_nulls - remaining_nulls
    return df, interpolated_count, capped_count
