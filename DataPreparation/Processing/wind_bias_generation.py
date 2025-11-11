"""
Wind Bias Calculation and Storage
Generates wind-aware attention bias matrices based on wind speed and direction.
Similar to spatial_partitioning.py - creates separate files for wind bias data.
"""

import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_wind_bias_matrix(wind_speed, wind_direction, num_nodes, num_sectors=8, n_rings=3, radii=[50, 200, 500]):
    """
    Compute wind bias for a single time step.
    
    Args:
        wind_speed: (num_nodes,) - wind speed for each node
        wind_direction: (num_nodes,) - wind direction for each node (0-360 degrees)
        num_nodes: number of stations
        num_sectors: number of dartboard sectors (typically 8)
        n_rings: number of dartboard rings
        radii: ring radii in km
        
    Returns:
        bias: (num_nodes, 1 + n_rings*num_sectors) - wind bias matrix
    """
    n_regions = 1 + n_rings * num_sectors
    bias = np.zeros((num_nodes, n_regions), dtype=np.float32)
    
    for node_idx in range(num_nodes):
        wind_spd = float(wind_speed[node_idx])
        wind_dir = float(wind_direction[node_idx])
        
        # Determine upwind sector (where wind comes FROM)
        sector_size = 360.0 / num_sectors
        upwind_sector = int((wind_dir + sector_size / 2) % 360 / sector_size) % num_sectors
        
        # Determine which rings to boost based on wind speed
        # Stronger winds transport pollution further
        if wind_spd < 3.0:
            rings_to_boost = [1]
        elif wind_spd < 6.0:
            rings_to_boost = [1, 2]
        else:
            rings_to_boost = [1, 2, 3]
        
        # Apply bias to upwind regions
        for ring_idx in rings_to_boost:
            region_idx = 1 + ring_idx * num_sectors + upwind_sector
            if region_idx < n_regions:
                bias[node_idx, region_idx] = 2.0
    
    return bias


def create_wind_bias_dataset(data_X, output_path, wind_speed_idx=12, wind_direction_idx=13, 
                             num_sectors=8, n_rings=3, radii=[50, 200, 500]):
    """
    Compute and save wind bias for entire dataset.
    
    Args:
        data_X: (num_samples, seq_len, num_nodes, num_features)
        output_path: directory to save wind bias matrices
        wind_speed_idx: index of wind speed in features
        wind_direction_idx: index of wind direction in features
        num_sectors: number of dartboard sectors
        n_rings: number of dartboard rings
        radii: ring radii
        
    Returns:
        wind_bias: (num_samples, seq_len, num_nodes, 1 + n_rings*num_sectors)
    """
    num_samples, seq_len, num_nodes, num_features = data_X.shape
    n_regions = 1 + n_rings * num_sectors
    
    logger.info(f"Computing wind bias...")
    logger.info(f"  Data shape: {data_X.shape}")
    logger.info(f"  Output regions: {n_regions} (1 center + {n_rings} rings × {num_sectors} sectors)")
    
    wind_bias = np.zeros((num_samples, seq_len, num_nodes, n_regions), dtype=np.float32)
    
    for sample_idx in range(num_samples):
        if (sample_idx + 1) % max(1, num_samples // 10) == 0:
            logger.info(f"  Processing sample {sample_idx + 1}/{num_samples}")
        
        for t in range(seq_len):
            wind_speed = data_X[sample_idx, t, :, wind_speed_idx]
            wind_direction = data_X[sample_idx, t, :, wind_direction_idx]
            
            wind_bias[sample_idx, t, :, :] = compute_wind_bias_matrix(
                wind_speed, wind_direction, num_nodes,
                num_sectors=num_sectors, n_rings=n_rings, radii=radii
            )
    
    logger.info(f"Wind bias computation complete. Shape: {wind_bias.shape}")
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'wind_bias.npy')
    np.save(output_file, wind_bias)
    logger.info(f"Saved wind bias to: {output_file}")
    
    return wind_bias


def load_wind_bias(data_path, category='train'):
    """
    Load precomputed wind bias.
    
    Args:
        data_path: directory containing wind_bias.npy
        category: 'train', 'val', or 'test' (unused, wind_bias.npy contains all)
        
    Returns:
        wind_bias: (num_samples, seq_len, num_nodes, n_regions)
    """
    bias_file = os.path.join(data_path, 'wind_bias.npy')
    if not os.path.exists(bias_file):
        logger.warning(f"Wind bias file not found: {bias_file}")
        return None
    
    wind_bias = np.load(bias_file)
    logger.info(f"Loaded wind bias from {bias_file}. Shape: {wind_bias.shape}")
    return wind_bias
