import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class StandardScaler:
    """
    Standard scaler for normalizing data.
    Normalizes using: (data - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_dataloader(datapath, batch_size, num_workers=2, pin_memory=False, use_wind_bias=False):
    """
    Load preprocessed dataset for training.
    
    Expected data shape per sample:
    - x: (seq_len, num_nodes, num_features)
      where num_features = 21 (15 base + 6 cyclic temporal features)
    - y: (horizon, num_nodes, 1)
    - wind_bias (optional): (seq_len, num_nodes, num_regions)
    
    Features breakdown:
    - Base features (15): PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, Benzene,
                         Temp, Humidity, Wind Speed, Wind Direction, Rainfall
    - Cyclic features (6): hour_sin, hour_cos, day_of_week_sin, day_of_week_cos,
                          month_sin, month_cos
    """
    import pickle
    data = {}
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
        if use_wind_bias and 'wind_bias' in cat_data.files:
            data['wind_bias_' + category] = cat_data['wind_bias']
            print(f"Loaded wind bias for {category}: {cat_data['wind_bias'].shape}")
    
    scaler_path = os.path.join(datapath, 'scalers.pkl')
    print(f"Loading scalers from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        original_scalers = pickle.load(f)
        
    pm25_scaler = StandardScaler(
        mean=original_scalers[0].mean_[0],
        std=original_scalers[0].scale_[0]
    )
    print(f"PM2.5 Scaler - Mean: {pm25_scaler.mean:.2f}, Std: {pm25_scaler.std:.2f}")
    
    # Log feature information
    sample_x_shape = data['x_train'].shape
    print(f"Sample X shape: {sample_x_shape}")
    print(f"  Sequence length: {sample_x_shape[0]}, Stations: {sample_x_shape[1]}, Features: {sample_x_shape[2]}")
    print(f"  (Features include: 15 base measurements + 6 cyclic temporal features = 21 total)")
    
    datasets = {}
    for category in ['train', 'val', 'test']:
        x_data = data['x_' + category]
        # Clip outliers: keep values within [-5, 5] std (handles negative wind after standardization)
        x_data = np.clip(x_data, -5.0, 5.0)
        x = torch.Tensor(x_data)
        y = torch.Tensor(data['y_' + category])
        
        if use_wind_bias and f'wind_bias_{category}' in data:
            wind_bias = torch.Tensor(data['wind_bias_' + category])
            datasets[category] = TensorDataset(x, y, wind_bias)
        else:
            datasets[category] = TensorDataset(x, y)
    
    results = {
        'train_loader': DataLoader(
            datasets['train'], 
            batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        ),
        'val_loader': DataLoader(
            datasets['val'], 
            batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        ),
        'test_loader': DataLoader(
            datasets['test'], 
            batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0
        ),
        'scaler': pm25_scaler,
    }
    
    print(f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    return results


def check_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")
