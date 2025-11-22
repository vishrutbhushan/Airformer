import os
import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_wind_bias_matrix(wind_speed, wind_direction, num_nodes, num_sectors=8, n_rings=3, radii=[50, 200, 500]):
    n_regions = 1 + n_rings * num_sectors
    bias = np.zeros((num_nodes, n_regions), dtype=np.float32)
    
    for node_idx in range(num_nodes):
        wind_spd = float(wind_speed[node_idx])
        wind_dir = float(wind_direction[node_idx])
        
        if np.isnan(wind_spd) or np.isnan(wind_dir):
            continue
        
        sector_size = 360.0 / num_sectors
        upwind_sector = int((wind_dir + sector_size / 2) % 360 / sector_size) % num_sectors
        
        if wind_spd < 3.0:
            rings_to_boost = [0]
        elif wind_spd < 5.0:
            rings_to_boost = [0, 1]
        else:
            rings_to_boost = [0, 1, 2]
        
        for ring_idx in rings_to_boost:
            region_idx = 1 + ring_idx * num_sectors + upwind_sector
            if region_idx < n_regions:
                bias_strength = max(0.5, min(2.5, 1.0 + wind_spd / 5.0))
                bias[node_idx, region_idx] = bias_strength
    
    return bias


def create_wind_bias_dataset(data_X, output_path=None, wind_speed_idx=12, wind_direction_idx=13, 
                             num_sectors=8, n_rings=3, radii=[50, 200, 500]):
    """Compute wind bias for entire dataset. Returns bias array, optionally saves to file."""
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
    
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, 'wind_bias.npy')
        np.save(output_file, wind_bias)
        logger.info(f"Saved wind bias to: {output_file}")
    
    return wind_bias
