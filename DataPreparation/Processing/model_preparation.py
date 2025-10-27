import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from config import logger, CORE_FEATURES, START_DATE, END_DATE

def prepare_data_for_model(data_folder, output_dir):
    logger.info("Preparing tensors")
    station_files = sorted(glob.glob(os.path.join(data_folder, "*.csv")))
    logger.info(f"Found {len(station_files)} preprocessed station files")
    # Use a global, consistent time index based on config START_DATE / END_DATE
    time_idx = pd.date_range(start=START_DATE, end=END_DATE, freq='3h')
    n_times = len(time_idx)
    logger.info(f"Time range: {time_idx[0]} to {time_idx[-1]}")
    logger.info(f"Total timesteps: {n_times} (3-hourly intervals)")
    valid_stations = []
    for fpath in station_files:
        station_code = os.path.splitext(os.path.basename(fpath))[0]
        df = pd.read_csv(fpath)
        if 'datetime' not in df.columns:
            logger.warning(f"Skipping {station_code}: no datetime column")
            continue
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Verify latitude/longitude exist and are valid numbers (not NaN)
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            logger.warning(f"Skipping {station_code}: missing spatial coordinates columns")
            continue
        try:
            lat_val = float(df['latitude'].iloc[0])
            lon_val = float(df['longitude'].iloc[0])
        except Exception:
            logger.warning(f"Skipping {station_code}: invalid spatial coordinate values ({df.get('latitude').iloc[0] if 'latitude' in df else 'NA'}, {df.get('longitude').iloc[0] if 'longitude' in df else 'NA'})")
            continue
        # Skip if coordinates are NaN
        if pd.isna(lat_val) or pd.isna(lon_val):
            logger.warning(f"Skipping {station_code}: NaN spatial coordinates")
            continue
        valid_stations.append({
            'station_code': station_code,
            'latitude': lat_val,
            'longitude': lon_val,
            'data': df
        })
    logger.info(f"Valid stations loaded: {len(valid_stations)}")
    n_stations = len(valid_stations)
    n_features = len(CORE_FEATURES)
    data = np.zeros((n_times, n_stations, n_features), dtype=np.float32)
    logger.info(f"Creating data matrix: {data.shape} (time, stations, features)")
    stations_info = []
    for i, station in enumerate(valid_stations):
        stations_info.append({
            'station_code': station['station_code'],
            'latitude': station['latitude'],
            'longitude': station['longitude']
        })
        # Ensure station data is indexed by the global time index
        df = station['data'].copy()
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime').sort_index()
        df = df.reindex(time_idx)
        for j, feature in enumerate(CORE_FEATURES):
            if feature in df.columns:
                vals = df[feature].values
                if len(vals) != n_times:
                    logger.warning(f"Station {station['station_code']} feature {feature} length {len(vals)} != expected {n_times}; reindexing filled with NaN/zeros")
                    # ensure length matches by reindexing above; fill NaN with 0.0
                    vals = pd.Series(vals).reindex(range(n_times)).fillna(0.0).values
                # convert to float and fill NaNs
                arr = np.array(pd.to_numeric(vals, errors='coerce')).astype(np.float32)
                arr = np.nan_to_num(arr, nan=0.0)
                data[:, i, j] = arr
            else:
                logger.warning(f"Feature {feature} missing in {station['station_code']}")
                data[:, i, j] = 0.0
    logger.info(f"Data matrix filled successfully: {data.shape}")
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.1)
    train = data[:train_size].copy()
    val = data[train_size:train_size+val_size].copy()
    test = data[train_size+val_size:].copy()
    logger.info(f"Data split - Train: {len(train)}, Validation: {len(val)}, Test: {len(test)}")
    logger.info("Applying feature-wise standardization...")
    scalers = []
    for i in range(n_features):
        scaler = StandardScaler()
        train_flat = train[..., i].reshape(-1, 1)
        val_flat = val[..., i].reshape(-1, 1)
        test_flat = test[..., i].reshape(-1, 1)
        scaler.fit(train_flat)
        train[..., i] = scaler.transform(train_flat).reshape(train[..., i].shape)
        val[..., i] = scaler.transform(val_flat).reshape(val[..., i].shape)
        test[..., i] = scaler.transform(test_flat).reshape(test[..., i].shape)
        scalers.append(scaler)
        logger.info(f"Feature {CORE_FEATURES[i]}: mean={scaler.mean_[0]:.3f}, std={scaler.scale_[0]:.3f}")
    logger.info(f"Standardization complete with {len(scalers)} feature scalers")
    seq_len_hours = 72
    horizon_hours = 72
    seq_len = seq_len_hours // 3
    horizon = horizon_hours // 3
    logger.info(f"Sequence configuration:")
    logger.info(f"  Input: {seq_len_hours}h : {seq_len} timesteps")
    logger.info(f"  Forecast: {horizon_hours}h : {horizon} timesteps")
    def make_sequences(data):
        n_samples = len(data) - seq_len - horizon + 1
        X = np.zeros((n_samples, seq_len, data.shape[1], data.shape[2]), dtype=np.float32)
        y = np.zeros((n_samples, horizon, data.shape[1], 1), dtype=np.float32)
        for i in range(n_samples):
            X[i] = data[i:i+seq_len]
            y[i] = data[i+seq_len:i+seq_len+horizon, :, 0:1]
        return X, y
    logger.info("Creating temporal sequences")
    X_train, y_train = make_sequences(train)
    X_val, y_val = make_sequences(val)
    X_test, y_test = make_sequences(test)
    logger.info(f"Sequence shapes:")
    logger.info(f"  Train: X{X_train.shape}, y{y_train.shape}")
    logger.info(f"  Validation: X{X_val.shape}, y{y_val.shape}")
    logger.info(f"  Test: X{X_test.shape}, y{y_test.shape}")
    logger.info("Saving model-ready data...")
    os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(f'{output_dir}/train.npz', x=X_train, y=y_train)
    np.savez_compressed(f'{output_dir}/val.npz', x=X_val, y=y_val)
    np.savez_compressed(f'{output_dir}/test.npz', x=X_test, y=y_test)
    with open(f'{output_dir}/scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    stations_df = pd.DataFrame(stations_info)
    metadata = {
        'n_stations': n_stations,
        'seq_len_timesteps': seq_len,
        'horizon_timesteps': horizon,
        'seq_len_hours': seq_len_hours,
        'horizon_hours': horizon_hours,
        'timestep_interval_hours': 3,
        'features': CORE_FEATURES,
        'stations': stations_df.to_dict('records'),
        'time_range': f"{time_idx[0]} to {time_idx[-1]}",
        'n_timesteps': n_times,
        'data_splits': {
            'train_size': len(X_train),
            'val_size': len(X_val),
            'test_size': len(X_test)
        }
    }
    with open(f'{output_dir}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
    logger.info("Complete")
    logger.info(f"Stations: {n_stations}")
    logger.info(f"Features: {len(CORE_FEATURES)}")
    logger.info(f"Timesteps: {n_times}")
    logger.info(f"Output directory: {output_dir}")
