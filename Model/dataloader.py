import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class WindBiasDataset(Dataset):
    """
    Dataset that includes precomputed wind bias matrices.
    Extends TensorDataset to also load wind bias if available.
    """
    def __init__(self, x, y, wind_bias=None):
        self.x = torch.Tensor(x) if not isinstance(x, torch.Tensor) else x
        self.y = torch.Tensor(y) if not isinstance(y, torch.Tensor) else y
        self.wind_bias = torch.Tensor(wind_bias) if wind_bias is not None and not isinstance(wind_bias, torch.Tensor) else wind_bias
        
        assert len(self.x) == len(self.y), "x and y must have same length"
        if self.wind_bias is not None:
            assert len(self.wind_bias) == len(self.x), "wind_bias must have same length as x"
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if self.wind_bias is not None:
            return self.x[idx], self.y[idx], self.wind_bias[idx]
        else:
            return self.x[idx], self.y[idx], None


def collate_with_wind_bias(batch):
    """
    Custom collate function for batches that may include wind bias.
    Returns ((x, y), wind_bias) where wind_bias is None if not available.
    """
    x_list, y_list, wb_list = zip(*batch)
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    
    # Only stack wind_bias if all items are not None
    if all(wb is not None for wb in wb_list):
        wind_bias = torch.stack(wb_list)
    else:
        wind_bias = None
    
    return (x, y), wind_bias


def get_dataloader(datapath, batch_size, num_workers=2, pin_memory=False):
    import pickle
    data = {}
    wind_bias_data = {}
    
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(datapath, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']
    
    # Try to load precomputed wind bias for each split
    for category in ['train', 'val', 'test']:
        wind_bias_path = os.path.join(datapath, f'{category}_wind_bias', 'wind_bias.npy')
        if os.path.exists(wind_bias_path):
            logger.info(f"Loading precomputed wind bias for {category} from {wind_bias_path}")
            wind_bias_data[category] = np.load(wind_bias_path)
            logger.info(f"  {category} wind bias shape: {wind_bias_data[category].shape}")
        else:
            logger.warning(f"Wind bias file not found for {category}: {wind_bias_path}. Will compute dynamically during training.")
            wind_bias_data[category] = None
    
    scaler_path = os.path.join(datapath, 'scalers.pkl')
    print(f"Loading scalers from {scaler_path}")
    with open(scaler_path, 'rb') as f:
        original_scalers = pickle.load(f)
        
    pm25_scaler = StandardScaler(
        mean=original_scalers[0].mean_[0],
        std=original_scalers[0].scale_[0]
    )
    print(f"PM2.5 Scaler - Mean: {pm25_scaler.mean:.2f}, Std: {pm25_scaler.std:.2f}")
    
    # Note: Wind scalers are no longer needed since wind bias is precomputed
    
    datasets = {}
    for category in ['train', 'val', 'test']:
        x = data['x_' + category]
        y = data['y_' + category]
        wb = wind_bias_data.get(category, None)
        datasets[category] = WindBiasDataset(x, y, wb)
    
    results = {
        'train_loader': DataLoader(
            datasets['train'], 
            batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=collate_with_wind_bias
        ),
        'val_loader': DataLoader(
            datasets['val'], 
            batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=collate_with_wind_bias
        ),
        'test_loader': DataLoader(
            datasets['test'], 
            batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
            collate_fn=collate_with_wind_bias
        ),
        'scaler': pm25_scaler,
        'wind_bias_available': bool(wind_bias_data),
    }
    
    print(f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    print(f"Precomputed wind bias available: {results['wind_bias_available']}")
    return results


def check_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    else:
        print("Using CPU")
        return torch.device("cpu")
