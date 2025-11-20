import logging

def print_config(config):
    logger = logging.getLogger(__name__)
    logger.info("Configuration:")
    for k, v in config.items():
        logger.info(f"  {k}: {v}")
def get_config():
    config = {
        "batch_size": 30,              # Number of samples per training batch
        "output_dim": 1,               # Number of output features (1 PM2.5)
        "grad_accum_steps": 1,         # Gradient accumulation steps
        "hidden_dim": 34,              # Model hidden dimension size
        "end_channels_mult": 8,        # Multiplier for end channels
        "num_heads": 2,                # Number of attention heads
        "blocks": 4,                   # Number of transformer blocks
        "dropout": 0.2,                # Dropout rate for regularization
        "learning_rate": 3e-4,         # Initial learning rate
        "max_epochs": 300,             # Maximum number of training epochs
        "patience": 30,                # Early stopping patience
        "use_amp": True,               # Use automatic mixed precision
        "pin_memory": True,            # Pin memory for DataLoader
        "num_workers": 0,              # Number of DataLoader workers
        "seq_len": 24,                # Input sequence length (7 days × 24 hours at 3-hourly) 
        "horizon": 24,                 # Forecast horizon (3 days at 3-hourly = 24 timesteps)
        "input_dim": 17,               # Number of input features (before cyclic: 15 core + 6 cyclic)
        "use_spatial": True,           # Enable spatial attention (DS-MSA)
        "use_stochastic": True,        # Enable stochastic latent variables
        "dartboard": 1,                # Dartboard partition type (50-200-500 km rings)
        "local_windows": [3, 6, 12, 24], # Local window sizes for CT-MSA
        "weight_decay": 5e-6,          # Weight decay for optimizer
        "grad_clip": 5.0,              # Gradient clipping value
        "scheduler_patience": 10,      # LR scheduler patience
        "scheduler_factor": 0.5,       # LR scheduler reduction factor
        "scheduler_threshold": 1e-4,   # LR scheduler improvement threshold
        "data_path": "../DataPreparation/Processing/Dataset/INDIAN_AIR",
        "dartboard_path": "../DataPreparation/Processing/Dataset/INDIAN_AIR/local_partition/50-200-500",
        "log_dir": "./logs",
        "use_wind_bias": False,         # Enable wind-aware attention bias
        "use_kan": False,              # Enable Kolmogorov-Arnold Networks in LatentLayers
    }
    return config
