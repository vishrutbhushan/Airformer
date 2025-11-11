import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def get_dataloader(datapath, batch_size, num_workers=2, pin_memory=False):
    import pickle
    data = {}
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    scaler_path = os.path.join(datapath, 'scalers.pkl')
    print(f"Loading scalers from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        original_scalers = pickle.load(f)
        
    pm25_scaler = StandardScaler(
        mean=original_scalers[0].mean_[0],
        std=original_scalers[0].scale_[0]
    )
    print(f"PM2.5 Scaler - Mean: {pm25_scaler.mean:.2f}, Std: {pm25_scaler.std:.2f}")
    
    # Extract scalers for other features (wind speed, wind direction, etc.)
    # original_scalers is a list of StandardScaler objects, one for each feature
    scalers_dict = {}
    if len(original_scalers) > 12:
        # Index 12 is wind speed
        scalers_dict['wind_speed_scaler'] = StandardScaler(
            mean=original_scalers[12].mean_[0],
            std=original_scalers[12].scale_[0]
        )
        print(f"Wind Speed Scaler - Mean: {scalers_dict['wind_speed_scaler'].mean:.4f}, Std: {scalers_dict['wind_speed_scaler'].std:.4f}")
    
    if len(original_scalers) > 13:
        # Index 13 is wind direction
        scalers_dict['wind_direction_scaler'] = StandardScaler(
            mean=original_scalers[13].mean_[0],
            std=original_scalers[13].scale_[0]
        )
        print(f"Wind Direction Scaler - Mean: {scalers_dict['wind_direction_scaler'].mean:.4f}, Std: {scalers_dict['wind_direction_scaler'].std:.4f}")
    
    datasets = {}
    for category in ['train', 'val', 'test']:
        x = torch.Tensor(data['x_' + category])
        y = torch.Tensor(data['y_' + category])
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
        'scalers': scalers_dict,  # All scalers for wind and other features
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
