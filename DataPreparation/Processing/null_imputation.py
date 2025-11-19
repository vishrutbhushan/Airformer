"""Simplified imputation: KNN for complete missing timesteps (≤3), Mean for rest."""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.neighbors import NearestNeighbors

from config import logger, CORE_FEATURES, LIMITS
from cyclic_features import get_cyclic_feature_columns


def impute_station_features(df, station_code, features=None):
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
    
    # ========== STAGE 1: KNN for completely empty timesteps (gap ≤3) ========== 
    logger.debug(f"  Stage 1: KNN imputation for completely empty timesteps (gap ≤3)")
    
    knn_count = 0
    # Find completely empty timesteps
    completely_empty_mask = df[imputable_features].isna().all(axis=1)
    empty_indices = np.where(completely_empty_mask.values)[0]
    
    if len(empty_indices) > 0:
        # Simple approach: count consecutive empty timesteps
        # Only use KNN for gaps of ≤3 timesteps
        knn_indices = []
        consecutive_count = 1
        
        for i in range(len(empty_indices)):
            if i > 0 and empty_indices[i] == empty_indices[i-1] + 1:
                # Consecutive empty timestep
                consecutive_count += 1
                if consecutive_count <= 3:
                    knn_indices.append(empty_indices[i])
            else:
                # Start of new gap
                consecutive_count = 1
                if consecutive_count <= 3:
                    knn_indices.append(empty_indices[i])
        
        if len(knn_indices) > 0 and len(cyclic_cols) > 0:
            # Fit KNN model
            X_cyclic = df[cyclic_cols].values
            k = min(5, len(df) - 1)
            knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
            knn_model.fit(X_cyclic)
            logger.debug(f"    KNN model fitted with k={k} neighbors for {len(knn_indices)} empty timesteps")
            
            # Impute each completely empty timestep using KNN
            for idx in knn_indices:
                distances, neighbor_indices = knn_model.kneighbors([X_cyclic[idx]])
                neighbor_indices = neighbor_indices[0]
                
                for col in imputable_features:
                    neighbor_vals = df.iloc[neighbor_indices][col].values
                    valid_vals = neighbor_vals[~np.isnan(neighbor_vals)]
                    
                    if len(valid_vals) > 0:
                        df.at[df.index[idx], col] = np.nanmean(valid_vals)
                        knn_count += 1
    
    # ========== STAGE 2: Mean imputation for all remaining nulls ========== 
    logger.debug(f"  Stage 2: Mean imputation for remaining nulls")
    mean_count = 0
    missed_count = 0
    
    for col in imputable_features:
        null_mask = df[col].isna()
        n_nulls = null_mask.sum()
        
        if n_nulls > 0:
            feature_mean = df[col].mean()
            if not np.isnan(feature_mean):
                df.loc[null_mask, col] = feature_mean
                mean_count += n_nulls
            else:
                missed_count += n_nulls
        
        # Apply physical constraints
        if col in LIMITS:
            min_val, max_val = LIMITS[col]
            df[col] = df[col].clip(min_val, max_val)
    
    # Log final statistics
    total_nulls_after = df[imputable_features].isna().sum().sum()
    empty_timesteps_after = (df[imputable_features].isna().sum(axis=1) == len(imputable_features)).sum()
    logger.info(f"Station {station_code}: Imputation complete")
    logger.info(f"  - Nulls: {total_nulls_before} → {total_nulls_after}")
    logger.info(f"  - Empty timesteps: {empty_timesteps_before} → {empty_timesteps_after}")
    logger.info(f"  - KNN-imputed (complete gaps ≤3): {knn_count}")
    logger.info(f"  - Mean-imputed: {mean_count}")
    logger.info(f"  - Missed (still null): {missed_count}")
    
    # Reset datetime index if it was originally a column
    if 'datetime' not in df.columns and df.index.name == 'datetime':
        df = df.reset_index()
    
    # Return imputation stats for aggregation
    df._impute_stats = dict(knn=knn_count, mean=mean_count, missed=missed_count)
    return df

