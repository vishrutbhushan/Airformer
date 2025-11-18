"""Simplified imputation: splines for small gaps, KNN for large/complete gaps."""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from scipy.interpolate import CubicSpline
from sklearn.neighbors import NearestNeighbors

from config import logger, CORE_FEATURES, LIMITS
from cyclic_features import get_cyclic_feature_columns

def spline_imputation(series, max_gap=3):
    """
    Fill gaps of size <= max_gap using cubic spline interpolation.
    Returns series with small gaps filled, larger gaps still NaN.
    """
    result = series.copy()
    mask = result.notna()
    
    if mask.sum() < 2:
        return result
    
    try:
        # Get indices and values of non-null points
        valid_idx = np.where(mask.values)[0]
        valid_vals = result.iloc[valid_idx].values
        
        # Create spline from valid points
        cs = CubicSpline(valid_idx, valid_vals, extrapolate='linear')
        
        # Identify gaps to fill
        current_gap_start = None
        for i in range(len(result)):
            if not mask.iloc[i]:
                if current_gap_start is None:
                    current_gap_start = i
            else:
                if current_gap_start is not None:
                    gap_size = i - current_gap_start
                    if gap_size <= max_gap:
                        # Fill this gap with spline
                        gap_indices = np.arange(current_gap_start, i)
                        result.iloc[current_gap_start:i] = cs(gap_indices)
                    current_gap_start = None
        
        logger.debug(f"Spline imputation: {result.isna().sum()} nulls remaining")
    except Exception as e:
        logger.debug(f"Spline imputation failed: {e}, keeping gaps")
    
    return result


def knn_impute_timestep(df, timestep_idx, knn_model, cyclic_cols, k=5):
    """
    Impute missing values in a timestep using pre-fitted KNN model.
    Uses whatever features are available to fill missing ones.
    
    Args:
        df: Full station dataframe with cyclic features
        timestep_idx: Index of timestep to impute
        knn_model: Pre-fitted NearestNeighbors model
        cyclic_cols: List of cyclic feature column names
        k: Number of nearest neighbors
    
    Returns:
        Tuple of (imputed series, count of features where all k neighbors are null)
    """
    if not cyclic_cols:
        return df.iloc[timestep_idx].copy(), 0
    
    knn_count = 0
    mean_count = 0
    missed_count = 0
    try:
        # Get cyclic features for this timestep
        X_cyclic = df[cyclic_cols].values
        # Find k nearest neighbors
        distances, neighbor_indices = knn_model.kneighbors([X_cyclic[timestep_idx]])
        neighbor_indices = neighbor_indices[0]
        # Get imputable features (exclude cyclic features which are deterministic)
        imputable_cols = [col for col in df.columns if col not in cyclic_cols and col != 'datetime']
        # Fill missing values from neighbors
        imputed = df.iloc[timestep_idx].copy()
        for col in imputable_cols:
            if pd.isna(imputed[col]):
                neighbor_vals = df.iloc[neighbor_indices][col].values
                valid_vals = neighbor_vals[~np.isnan(neighbor_vals)]
                if len(valid_vals) > 0:
                    imputed[col] = np.nanmean(valid_vals)
                    knn_count += 1
                else:
                    # Fallback: mean imputation for this feature (excluding current row)
                    feature_mean = df[col].drop(index=df.index[timestep_idx]).mean()
                    if not np.isnan(feature_mean):
                        imputed[col] = feature_mean
                        mean_count += 1
                    else:
                        missed_count += 1
        # logger.debug(f"KNN imputation timestep {timestep_idx}: filled from {len(neighbor_indices)} neighbors")
    except Exception as e:
        logger.debug(f"KNN imputation failed for timestep {timestep_idx}: {e}")
        imputed = df.iloc[timestep_idx].copy()
    return imputed, knn_count, mean_count, missed_count


def impute_station_features(df, station_code, features=None):
    """
    Simplified two-stage imputation:
    1. Spline interpolation for small gaps (2-3 timesteps = 6-9 hours)
    2. KNN for larger/complete gaps using cyclic features
    
    Args:
        df: Station dataframe with datetime index (or datetime column)
        station_code: Station identifier for logging
        features: List of features to impute (default: CORE_FEATURES)
    
    Returns:
        Fully imputed dataframe with no nulls
    """
    if features is None:
        features = [f for f in CORE_FEATURES if f in df.columns]
    
    df = df.copy()
    
    # Ensure datetime is set as index for easier timestep operations
    if 'datetime' in df.columns:
        df = df.set_index('datetime')
    
    cyclic_cols = get_cyclic_feature_columns()
    cyclic_cols = [col for col in cyclic_cols if col in df.columns]
    imputable_features = [f for f in features if f not in cyclic_cols]
    
    logger.info(f"Station {station_code}: Imputing {len(imputable_features)} features across {len(df)} timesteps")
    
    # Log statistics BEFORE imputation
    total_nulls_before = df[imputable_features].isna().sum().sum()
    empty_timesteps_before = (df[imputable_features].isna().sum(axis=1) == len(imputable_features)).sum()
    logger.debug(f"  Before: {total_nulls_before} nulls, {empty_timesteps_before} empty timesteps")
    
    # ========== STAGE 1: Spline interpolation for small gaps (2-3 timesteps) ========== 
    logger.debug(f"  Stage 1: Spline interpolation for small gaps (≤3 timesteps)")
    spline_count = 0
    for feature in imputable_features:
        if feature not in df.columns:
            continue
        n_nulls = df[feature].isna().sum()
        if n_nulls == 0:
            continue
        # Count how many will be filled by spline (gaps <= 3)
        before = df[feature].isna().sum()
        df[feature] = spline_imputation(df[feature], max_gap=3)
        after = df[feature].isna().sum()
        spline_count += before - after
        # Apply physical constraints
        if feature in LIMITS:
            min_val, max_val = LIMITS[feature]
            df[feature] = df[feature].clip(min_val, max_val)
    
    # ========== STAGE 2: KNN for remaining gaps using cyclic features ========== 
    logger.debug(f"  Stage 2: KNN imputation for remaining gaps")
    # Fit KNN model once upfront
    X_cyclic = df[cyclic_cols].values
    k = min(5, len(df) - 1)
    knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    knn_model.fit(X_cyclic)
    logger.debug(f"    KNN model fitted with k={k} neighbors")
    # Find all timesteps with any missing values
    has_nulls_mask = df[imputable_features].isna().any(axis=1)
    null_indices = np.where(has_nulls_mask.values)[0]
    knn_count = 0
    mean_count = 0
    missed_count = 0
    if len(null_indices) > 0:
        logger.debug(f"    Imputing {len(null_indices)} timesteps with missing values via KNN")
        for idx in null_indices:
            imputed_row, knn_c, mean_c, missed_c = knn_impute_timestep(df, idx, knn_model, cyclic_cols, k=k)
            df.iloc[idx] = imputed_row
            knn_count += knn_c
            mean_count += mean_c
            missed_count += missed_c
    
    # Log final statistics
    total_nulls_after = df[imputable_features].isna().sum().sum()
    empty_timesteps_after = (df[imputable_features].isna().sum(axis=1) == len(imputable_features)).sum()
    logger.info(f"Station {station_code}: Imputation complete")
    logger.info(f"  - Nulls: {total_nulls_before} → {total_nulls_after}")
    logger.info(f"  - Empty timesteps: {empty_timesteps_before} → {empty_timesteps_after}")
    logger.info(f"  - Spline-imputed: {spline_count}")
    logger.info(f"  - KNN-imputed: {knn_count}")
    logger.info(f"  - Mean-imputed (KNN fallback): {mean_count}")
    logger.info(f"  - Missed (still null): {missed_count}")
    # Reset datetime index if it was originally a column
    if 'datetime' not in df.columns and df.index.name == 'datetime':
        df = df.reset_index()
    # Return imputation stats for aggregation
    df._impute_stats = dict(spline=spline_count, knn=knn_count, mean=mean_count, missed=missed_count)
    return df

