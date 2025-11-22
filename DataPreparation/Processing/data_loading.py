import pandas as pd
from config import logger, BASE_FEATURES, FEATURE_MAPPING, START_DATE, END_DATE

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
