import os
import sys
import torch
import logging

# Setup logging to both console and file
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'training.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='a'),  # File handler
            logging.StreamHandler(sys.stdout)          # Console handler
        ]
    )

logger = logging.getLogger(__name__)

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

from model import AirFormer
from dataloader import get_dataloader, check_device
from trainer import Trainer
from config import get_config, print_config

"""
AirFormer Training Script

Data Format:
- Input features: 21 total
  - 15 base measurements: PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, O3, Benzene,
                         Temperature, Humidity, Wind Speed, Wind Direction, Rainfall
  - 6 cyclic temporal features: hour_sin, hour_cos, day_of_week_sin, day_of_week_cos,
                               month_sin, month_cos

- Input shape: (batch_size, seq_len=24, num_nodes, 21 features)
- Output shape: (batch_size, horizon=24, num_nodes, 1)  [PM2.5 predictions]

Cyclic Features (added during preprocessing):
- Encoded using sine/cosine transformations to capture daily/weekly/yearly patterns
- Naturally normalized to [-1, 1] range
- No missing values (derived deterministically from datetime)
"""


def main():
    # Load configuration
    config = get_config()
    
    # Setup logging first
    setup_logging(config['log_dir'])
    
    logger.info("Starting AirFormer training")
    print_config(config)
    
    # Select device: use GPU if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device for model/data (cuda: GPU, cpu: CPU)
    
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.95, 0)  # Limit GPU memory usage to 95% for this process
        torch.backends.cudnn.benchmark = True                # Enable cuDNN auto-tuner for best performance
        torch.backends.cuda.matmul.allow_tf32 = True          # Allow TensorFloat-32 for faster matmul on Ampere GPUs
        torch.backends.cudnn.allow_tf32 = True                # Allow TensorFloat-32 for cuDNN convolutions
        torch.backends.cudnn.deterministic = False            # Use non-deterministic algorithms (faster, less reproducible)
        torch.backends.cudnn.enabled = True                   # Enable cuDNN backend for GPU acceleration
        torch.cuda.empty_cache()                              # Release unused GPU memory back to the system
        
        logger.info("CUDA optimizations enabled")
    
    logger.info("Loading data...")
    data = get_dataloader(
        config['data_path'],              # Path to preprocessed data
        config['batch_size'],             # Batch size for training
        num_workers=config['num_workers'],# Number of worker processes for data loading
        pin_memory=config['pin_memory'] and device.type == 'cuda', # Use pinned memory for faster GPU transfer
        use_wind_bias=config.get('use_wind_bias', False) # Load wind bias if enabled
    )
    
    # Get actual number of stations from data
    sample_x = next(iter(data['train_loader']))[0]
    n_stations = sample_x.shape[2]  # Shape: (batch, seq_len, stations, features)
    logger.info(f"Actual stations from data: {n_stations}")
    logger.info(f"Sample input shape: {sample_x.shape}")
    
    logger.info("Creating model...")
    model = AirFormer(
        num_nodes=n_stations,              # Number of stations (nodes)
        input_dim=config['input_dim'],     # Number of input features per station
        output_dim=config['output_dim'],   # Number of output features (1 PM2.5)
        seq_len=config['seq_len'],         # Input sequence length (timesteps)
        horizon=config['horizon'],         # Output forecast horizon (timesteps)
        hidden_channels=config['hidden_dim'], # Model hidden dimension
        end_channels=config['hidden_dim'] * config['end_channels_mult'], # Final layer dimension
        blocks=config['blocks'],           # Number of transformer blocks
        num_heads=config['num_heads'],     # Number of attention heads
        dropout=config['dropout'],         # Dropout rate for regularization
        spatial_flag=config['use_spatial'],# Use spatial attention blocks
        stochastic_flag=config['use_stochastic'], # Use stochastic output (uncertainty)
        dartboard=config['dartboard'],     # Precomputed spatial assignment matrix
        dartboard_path=config['dartboard_path'], # Path to dartboard file
        local_windows=config.get('local_windows', None), # Local window config (optional)
        device=device,                     # Device to run model (cpu/cuda)
        use_kan=config.get('use_kan', False), # Use KAN for feedforward layers
        use_wind_bias=config.get('use_wind_bias', False) # Use precomputed wind bias
    ).to(device)                          # Move model to device (GPU/CPU)
    
    total_params = sum(p.numel() for p in model.parameters()) # Total number of model parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # Parameters updated during training
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Pass the data dict and config values to Trainer as expected
    trainer = Trainer(
        model,
        data,
        config['max_epochs'],
        config['learning_rate'],
        config['patience'],
        config['log_dir'],
        device,
        grad_accum_steps=config.get('grad_accum_steps', 2),
        weight_decay=config.get('weight_decay', 1e-5),
        grad_clip=config.get('grad_clip', 5.0),
        scheduler_patience=config.get('scheduler_patience', 10),
        scheduler_factor=config.get('scheduler_factor', 0.5),
        scheduler_threshold=config.get('scheduler_threshold', 1e-4),
        use_amp=config.get('use_amp', True)
    )

    trainer.train()
    trainer.test()
    logger.info("Done!")


if __name__ == "__main__":
    main()
