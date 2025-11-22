import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer for AirFormer model.
    
    Handles training loop, validation, early stopping, and model checkpointing.
    
    Input data format:
    - x: (batch, seq_len, num_nodes, 21 features)
      - Features: 15 base measurements + 6 cyclic temporal features
    - y: (batch, horizon, num_nodes, 1)
    """
    def __init__(self, model, data, max_epochs, learning_rate, patience, log_dir, device, 
                 grad_accum_steps=2, weight_decay=1e-5, grad_clip=5.0, 
                 scheduler_patience=10, scheduler_factor=0.5, scheduler_threshold=1e-4,
                 use_amp=True):
        self.model = model
        self.device = device
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_dir = log_dir
        
        self.train_loader = data['train_loader']
        self.val_loader = data['val_loader']
        self.test_loader = data['test_loader']
        self.scaler = data['scaler']
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',                    # Minimize validation loss
            factor=scheduler_factor,       # Reduce LR by this factor (0.5 = halve)
            patience=scheduler_patience,   # Wait this many epochs before reducing
            threshold=scheduler_threshold, # Minimum change to qualify as improvement
            min_lr=1e-7                   # Don't reduce below this
        )
        
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler_amp = torch.amp.GradScaler('cuda') if self.use_amp else None
        
        self.accumulation_steps = grad_accum_steps
        self.grad_clip = grad_clip
        
        self.best_val_loss = float('inf')
        self.wait = 0
        
        if self.use_amp:
            logger.info("Mixed precision training enabled")
        logger.info(f"Gradient accumulation: {self.accumulation_steps} steps")
        logger.info(f"Gradient clipping: {self.grad_clip}")
        logger.info(f"Weight decay: {weight_decay}")
        logger.info(f"LR scheduler: ReduceLROnPlateau(patience={scheduler_patience}, factor={scheduler_factor})")
        
        # Log training data information
        logger.info(f"Training data: {len(self.train_loader)} batches × {self.train_loader.batch_size} batch_size = {len(self.train_loader) * self.train_loader.batch_size} total sequences")
        logger.info(f"Validation data: {len(self.val_loader)} batches × {self.val_loader.batch_size} batch_size = {len(self.val_loader) * self.val_loader.batch_size} total sequences")

        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            if len(batch_data) == 3:
                x, y, wind_bias = batch_data
                x, y, wind_bias = x.to(self.device), y.to(self.device), wind_bias.to(self.device)
            else:
                x, y = batch_data
                x, y = x.to(self.device), y.to(self.device)
                wind_bias = None
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    if self.model.stochastic_flag:
                        y_pred, x_rec, kl_loss = self.model(x, wind_bias=wind_bias)
                        pred_loss = nn.functional.l1_loss(y_pred, y)
                        rec_loss = nn.functional.l1_loss(x_rec, x)
                        loss = pred_loss + rec_loss + kl_loss
                    else:
                        y_pred = self.model(x, wind_bias=wind_bias)
                        loss = nn.functional.l1_loss(y_pred, y)
                    
                    loss = loss / self.accumulation_steps
                
                self.scaler_amp.scale(loss).backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    self.scaler_amp.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler_amp.step(self.optimizer)
                    self.scaler_amp.update()
                    self.optimizer.zero_grad()
            else:
                if self.model.stochastic_flag:
                    y_pred, x_rec, kl_loss = self.model(x, wind_bias=wind_bias)
                    pred_loss = nn.functional.l1_loss(y_pred, y)
                    rec_loss = nn.functional.l1_loss(x_rec, x)  # FIX: should be x not y
                    loss = pred_loss + rec_loss + kl_loss
                else:
                    y_pred = self.model(x, wind_bias=wind_bias)
                    loss = nn.functional.l1_loss(y_pred, y)
                
                loss = loss / self.accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_data in self.val_loader:
                if len(batch_data) == 3:
                    x, y, wind_bias = batch_data
                    x, y, wind_bias = x.to(self.device), y.to(self.device), wind_bias.to(self.device)
                else:
                    x, y = batch_data
                    x, y = x.to(self.device), y.to(self.device)
                    wind_bias = None
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.model.stochastic_flag:
                            y_pred, _, _ = self.model(x, wind_bias=wind_bias)
                        else:
                            y_pred = self.model(x, wind_bias=wind_bias)
                        loss = nn.functional.l1_loss(y_pred, y)
                else:
                    if self.model.stochastic_flag:
                        y_pred, _, _ = self.model(x, wind_bias=wind_bias)
                    else:
                        y_pred = self.model(x, wind_bias=wind_bias)
                    loss = nn.functional.l1_loss(y_pred, y)
                
                total_loss += loss.item()
        
        # Clean cache only after all validation batches
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        import time
        logger.info("Training...")
        
        for epoch in range(self.max_epochs):
            epoch_start = time.time()
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            # ReduceLROnPlateau uses validation loss
            self.scheduler.step(val_loss)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch+1:3d}/{self.max_epochs} | "
                       f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                       f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                       f"Time: {epoch_time:.1f}s")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
                torch.save(self.model.state_dict(), f'{self.log_dir}/best_model.pth')
                logger.info(f"  [SAVED] (val_loss: {val_loss:.4f})")
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        logger.info("Training complete")
    
    def test(self):
        logger.info("Testing Best Model")
       
        self.model.load_state_dict(torch.load(f'{self.log_dir}/best_model.pth'))
        self.model.eval()
        
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                if len(batch_data) == 3:
                    x, y, wind_bias = batch_data
                    x, y, wind_bias = x.to(self.device), y.to(self.device), wind_bias.to(self.device)
                else:
                    x, y = batch_data
                    x, y = x.to(self.device), y.to(self.device)
                    wind_bias = None
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
                        if self.model.stochastic_flag:
                            y_pred, _, _ = self.model(x, wind_bias=wind_bias)
                        else:
                            y_pred = self.model(x, wind_bias=wind_bias)
                else:
                    if self.model.stochastic_flag:
                        y_pred, _, _ = self.model(x, wind_bias=wind_bias)
                    else:
                        y_pred = self.model(x, wind_bias=wind_bias)
                
                predictions.append(y_pred.cpu().numpy())
                targets.append(y.cpu().numpy())
        
        # Clean cache only after all test batches
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Inverse transform to real PM2.5 values
        pred_real = self.scaler.inverse_transform(predictions)
        target_real = self.scaler.inverse_transform(targets)
        
        # Overall metrics
        overall_mae = np.abs(pred_real - target_real).mean()
        overall_rmse = np.sqrt(((pred_real - target_real) ** 2).mean())
        
        logger.info(f"\nOverall Test Metrics:")
        logger.info(f"  MAE:  {overall_mae:.4f} µg/m³")
        logger.info(f"  RMSE: {overall_rmse:.4f} µg/m³")
        
        # Time window metrics - AirFormer style breakdown for 72-hour horizon
        logger.info(f"\nTime Window Analysis (3-hourly intervals):")
        horizon = pred_real.shape[1]  # Get actual horizon (should be 24)
        logger.info(f"Prediction horizon: {horizon} timesteps ({horizon * 3} hours)")
        
        if horizon == 24:  # 72 hours total (24 × 3-hour intervals)
            # Break down into three 8-timestep windows (24 hours each)
            windows = [
                (0, 8, '1-24h'),      # First 24 hours (timesteps 0-7)
                (8, 16, '25-48h'),    # Second 24 hours (timesteps 8-15)  
                (16, 24, '49-72h')    # Third 24 hours (timesteps 16-23)
            ]
            
            logger.info(f"Breaking down {horizon} timesteps into 3 windows of 8 timesteps each:")
            for start, end, name in windows:
                pred_window = pred_real[:, start:end]
                target_window = target_real[:, start:end]
                
                mae = np.abs(pred_window - target_window).mean()
                rmse = np.sqrt(((pred_window - target_window) ** 2).mean())
                
                logger.info(f"  {name:8s} (steps {start:2d}-{end-1:2d}): MAE={mae:6.2f}, RMSE={rmse:6.2f} µg/m³")
        else:
            logger.info(f"  Full horizon ({horizon} timesteps): Single window analysis")
            mae = np.abs(pred_real - target_real).mean() 
            rmse = np.sqrt(((pred_real - target_real) ** 2).mean())
            logger.info(f"  Full horizon: MAE={mae:6.2f}, RMSE={rmse:6.2f} µg/m³")
        
        # Severe pollution events (PM2.5 > 75 µg/m³)
        severe_mask = target_real > 75
        if severe_mask.any():
            severe_mae = np.abs(pred_real[severe_mask] - target_real[severe_mask]).mean()
            severe_rmse = np.sqrt(((pred_real[severe_mask] - target_real[severe_mask]) ** 2).mean())
            severe_pct = (severe_mask.sum() / severe_mask.size) * 100
            
            logger.info(f"\nSevere Pollution Events (PM2.5 > 75):")
            logger.info(f"  Frequency: {severe_pct:.1f}% of data")
            logger.info(f"  MAE:  {severe_mae:.4f} µg/m³")
            logger.info(f"  RMSE: {severe_rmse:.4f} µg/m³")
        
        logger.info("="*70 + "\n")
        
        return overall_mae, overall_rmse
