"""Kalman filter-based time series imputation with seasonal decomposition."""

import pandas as pd
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')
from scipy.signal import savgol_filter

from config import logger, CORE_FEATURES, LIMITS


def kalman_filter_imputation(series, seasonal_period=8):
    """
    Bidirectional Kalman filter imputation for time series.
    
    Two-pass algorithm handles ALL gaps uniformly:
    - Forward pass: learns from first observation onward
    - Backward pass: learns from last observation backward
    - Blends both estimates for optimal fill
    
    Works for:
    - Large gaps at start (no prior data)
    - Large gaps at end (no future data)
    - Large gaps in middle
    - Small gaps anywhere
    
    Pure filtering, no backups, no arbitrary rules.
    """
    result = series.copy()
    mask = result.notna()
    n_nulls = result.isna().sum()
    
    if n_nulls == 0 or mask.sum() < 3:
        return result
    
    try:
        # Extract observed values and their positions
        observed_indices = np.where(mask.values)[0]
        observed_vals = result.iloc[observed_indices].values
        
        if len(observed_vals) == 0:
            return result
        
        # Estimate initial state from observed data
        mean_val = np.nanmean(observed_vals)
        trend_est = 0
        if len(observed_vals) >= 2:
            diffs = np.diff(observed_vals)
            trend_est = np.median(diffs[~np.isnan(diffs)]) if len(diffs) > 0 else 0
        
        # Extract seasonal pattern from observed data
        seasonal_pattern = np.zeros(seasonal_period)
        seasonal_counts = np.zeros(seasonal_period)
        for idx, val in zip(observed_indices, observed_vals):
            if not np.isnan(val):
                seasonal_idx = idx % seasonal_period
                seasonal_pattern[seasonal_idx] += val
                seasonal_counts[seasonal_idx] += 1
        
        # Average the seasonal patterns
        for i in range(seasonal_period):
            if seasonal_counts[i] > 0:
                seasonal_pattern[i] /= seasonal_counts[i]
            else:
                seasonal_pattern[i] = mean_val
        
        # Kalman filter parameters
        process_variance = np.var(observed_vals) if len(observed_vals) > 1 else 1.0
        measurement_variance = process_variance * 0.1  # Trust observations more than model
        
        # State transition matrix (how state evolves from one step to next)
        F = np.array([[1, 1], [0, 1]])  # value += trend
        Q = np.array([[process_variance * 0.1, 0], [0, process_variance * 0.01]])
        
        # Adaptive process noise based on gap size (longer gaps → more flexible filter)
        gap_ratio = np.clip(n_nulls / len(series), 0.5, 3.0)
        Q = Q * gap_ratio
        
        # Measurement matrix (observe only the value, not trend)
        H = np.array([[1, 0]])
        R = np.array([[measurement_variance]])
        
        # ========== FORWARD PASS ==========
        forward_estimates = np.full(len(result), np.nan)
        state = np.array([mean_val, trend_est])
        state_covariance = np.array([[process_variance, 0], [0, process_variance * 0.01]])
        
        for i in range(len(result)):
            seasonal_component = seasonal_pattern[i % seasonal_period]
            
            # Predict step
            state_pred = F @ state
            cov_pred = F @ state_covariance @ F.T + Q
            
            # Update step
            if mask.iloc[i]:  # We have an observation
                obs = result.iloc[i]
                deseasonalized_obs = obs - seasonal_component
                innovation = deseasonalized_obs - H @ state_pred
                innovation_cov = H @ cov_pred @ H.T + R
                kalman_gain = cov_pred @ H.T @ np.linalg.inv(innovation_cov)
                
                state = state_pred + kalman_gain @ np.array([innovation])
                state_covariance = (np.eye(2) - kalman_gain @ H) @ cov_pred
            else:
                # Missing observation, use prediction
                state = state_pred
                state_covariance = cov_pred
            
            # Store forward estimate
            forward_estimates[i] = state[0] + seasonal_component
        
        # ========== BACKWARD PASS ==========
        backward_estimates = np.full(len(result), np.nan)
        
        # Find first observation going backward (rightmost observation)
        last_obs_idx = None
        for i in range(len(result) - 1, -1, -1):
            if mask.iloc[i]:
                last_obs_idx = i
                break
        
        # Start state from the last (rightmost) observation
        if last_obs_idx is not None:
            state = np.array([result.iloc[last_obs_idx], trend_est])  # Use actual observation, not mean
        else:
            state = np.array([mean_val, trend_est])
        
        state_covariance = np.array([[process_variance, 0], [0, process_variance * 0.01]])
        
        for i in range(len(result) - 1, -1, -1):
            seasonal_component = seasonal_pattern[i % seasonal_period]
            
            # Predict step (going backward in time)
            state_pred = F @ state
            cov_pred = F @ state_covariance @ F.T + Q
            
            # Update step
            if mask.iloc[i]:  # We have an observation
                obs = result.iloc[i]
                deseasonalized_obs = obs - seasonal_component
                innovation = deseasonalized_obs - H @ state_pred
                innovation_cov = H @ cov_pred @ H.T + R
                kalman_gain = cov_pred @ H.T @ np.linalg.inv(innovation_cov)
                
                state = state_pred + kalman_gain @ np.array([innovation])
                state_covariance = (np.eye(2) - kalman_gain @ H) @ cov_pred
            else:
                # Missing observation, use prediction
                state = state_pred
                state_covariance = cov_pred
            
            # Store backward estimate
            backward_estimates[i] = state[0] + seasonal_component
        
        # ========== BLEND ESTIMATES ==========
        # For observed values, use observation
        # For gaps: average forward and backward estimates (both have context)
        for i in range(len(result)):
            if not mask.iloc[i]:
                # Blend forward and backward estimates
                if np.isnan(forward_estimates[i]) or np.isnan(backward_estimates[i]):
                    # One-sided estimate
                    estimate = forward_estimates[i] if not np.isnan(forward_estimates[i]) else backward_estimates[i]
                else:
                    # Both estimates available - take average for smooth transition
                    estimate = (forward_estimates[i] + backward_estimates[i]) / 2.0
                
                result.iloc[i] = estimate
        
        # ========== POST-SMOOTHING ==========
        # Apply Savitzky-Golay filter to smooth remaining artifacts
        # Only smooth the imputed values, preserve observations
        if len(result) >= 9:
            try:
                smoothed = pd.Series(
                    savgol_filter(result.values, window_length=9, polyorder=2),
                    index=result.index
                )
                # Only replace imputed values with smoothed version, keep observations
                result[~mask] = smoothed[~mask]
            except:
                pass  # If smoothing fails, keep as-is
        
        logger.debug(f"Kalman imputation: {n_nulls} → {result.isna().sum()} nulls")
        
    except Exception as e:
        logger.debug(f"Kalman filter failed: {e}")
        # Last resort: forward fill then backward fill
        result = result.fillna(method='ffill').fillna(method='bfill')
    
    return result


def impute_station_features(df, station_code, features=None):

    if features is None:
        features = [f for f in CORE_FEATURES if f in df.columns]
    
    df = df.copy()
    logger.info(f"Station {station_code}: Starting imputation ({len(df)} timesteps, {len(features)} features)")
    
    # Log null statistics BEFORE imputation
    null_stats_before = df[features].isna().sum()
    total_nulls_before = null_stats_before.sum()
    completely_empty_rows_before = (df[features].isna().sum(axis=1) == len(features)).sum()
    
    logger.debug(f"Station {station_code}: Before imputation:")
    logger.debug(f"  - Total null values: {total_nulls_before}")
    logger.debug(f"  - Completely empty rows: {completely_empty_rows_before}")
    
    # Single pass: Apply Kalman filter imputation to each feature
    logger.debug(f"Station {station_code}: Applying Kalman Filter imputation to all {len(features)} features")
    
    for feature in features:
        if feature not in df.columns:
            continue
        
        n_nulls = df[feature].isna().sum()
        if n_nulls == 0:
            logger.debug(f"  {feature}: 0 nulls, skipping")
            continue
        
        logger.debug(f"  {feature}: Imputing {n_nulls} null values using Kalman Filter")
        
        # Apply Kalman filter imputation
        series = kalman_filter_imputation(df[feature])
        
        # Apply physical constraints
        if feature in LIMITS:
            min_val, max_val = LIMITS[feature]
            series = series.clip(min_val, max_val)
            logger.debug(f"  {feature}: Applied limits [{min_val}, {max_val}]")
        
        df[feature] = series
    
    # Log final statistics AFTER imputation
    null_stats_after = df[features].isna().sum()
    total_nulls_after = null_stats_after.sum()
    completely_empty_rows_after = (df[features].isna().sum(axis=1) == len(features)).sum()
    
    logger.info(f"Station {station_code}: Imputation complete:")
    logger.info(f"  - Null values reduced from {total_nulls_before} to {total_nulls_after}")
    logger.info(f"  - Empty rows reduced from {completely_empty_rows_before} to {completely_empty_rows_after}")
    
    return df

