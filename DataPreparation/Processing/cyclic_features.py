import pandas as pd
import numpy as np
from config import logger


def add_cyclic_features(df, datetime_col='datetime'):

    df = df.copy()
    
    if datetime_col not in df.columns:
        logger.warning(f"Column '{datetime_col}' not found in dataframe")
        return df
    
    # Ensure datetime column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Hour of day (0-23)
    hour = df[datetime_col].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Day of week (0-6, where 0 is Monday)
    day_of_week = df[datetime_col].dt.dayofweek
    df['day_of_week_sin'] = np.sin(2 * np.pi * day_of_week / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * day_of_week / 7)
    
    # Month of year (1-12)
    month = df[datetime_col].dt.month
    df['month_sin'] = np.sin(2 * np.pi * (month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (month - 1) / 12)
    
    logger.debug(f"Added cyclic features. Shape: {df.shape}")
    return df


def get_cyclic_feature_columns():
    """Return list of cyclic feature column names."""
    return [
        'hour_sin', 'hour_cos',
        'day_of_week_sin', 'day_of_week_cos',
        'month_sin', 'month_cos'
    ]
