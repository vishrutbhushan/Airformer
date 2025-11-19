import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KANLayer(nn.Module):
    """
    Kolmogorov–Arnold Network (KAN) Layer.
    Uses learnable sine basis functions to expand input dimensionality, providing
    a powerful non-linear transformation that can learn complex patterns efficiently.
    
    Reference: Kolmogorov-Arnold Networks (2024)
    """
    def __init__(self, in_dim, out_dim, num_basis=8, grid_size=5, grid_min=-1.0, grid_max=1.0):
        """
        Initializes the KAN layer.
        
        Args:
            in_dim (int): Input feature dimension.
            out_dim (int): Output feature dimension.
            num_basis (int): Number of basis functions per input dimension.
            grid_size (int): Number of grid points for spline knots.
            grid_min (float): Minimum value for grid initialization.
            grid_max (float): Maximum value for grid initialization.
        """
        super().__init__()
        
        # Handle in_dim as int or list/tuple
        if isinstance(in_dim, (list, tuple)):
            if len(in_dim) == 0:
                raise ValueError("in_dim must be a positive integer or non-empty list/tuple")
            in_dim = int(in_dim[0])
        
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.num_basis = int(num_basis)
        self.grid_size = int(grid_size)
        
        # Learnable coefficients for basis expansion: (out_dim, in_dim, num_basis)
        self.coeff = nn.Parameter(torch.randn(self.out_dim, self.in_dim, self.num_basis) * 0.1)
        
        # Per-channel frequencies for greater expressivity: (in_dim, num_basis)
        self.freq = nn.Parameter(torch.randn(self.in_dim, self.num_basis) * 0.1)
        
        # Phase shifts for basis functions: (num_basis,)
        self.phase = nn.Parameter(torch.zeros(self.num_basis))
        
        # Linear skip connection for residual learning
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        
        # Optional learnable activation scales per dimension
        self.activation_scale = nn.Parameter(torch.ones(self.in_dim, 1))

    def forward(self, x):
        """
        Forward pass applying KAN transformation.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., in_dim)
            
        Returns:
            torch.Tensor: Output tensor of shape (..., out_dim)
        """
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"KANLayer expected input last-dim {self.in_dim}, got {x.shape[-1]}")
        
        # Apply activation scaling and create basis functions
        x_scaled = x * self.activation_scale.t()
        
        # Generate sine basis functions: (..., in_dim, num_basis)
        basis = torch.sin(x_scaled.unsqueeze(-1) * self.freq.unsqueeze(0) + self.phase.view(1, 1, -1))
        
        # Contract over input dimensions: (..., in_dim, num_basis) @ (out_dim, in_dim, num_basis)
        # Using einsum for efficient batched operation
        y = torch.einsum('...in,oin->...o', basis, self.coeff)
        
        # Add linear residual connection for better gradient flow
        return self.linear(x) + y




# WindAwareBias class removed - wind bias is now precomputed in preprocessing
# and loaded directly from wind_bias.npy files. See wind_bias_generation.py


class LatentLayer(nn.Module):
    """
    Encodes input features to produce the parameters (mu and sigma) of a latent variable distribution.
    This layer is a core component of the stochastic model, enabling the capture of uncertainty.
    
    Enhanced with KAN layers for improved non-linear feature transformation before
    computing distribution parameters.
    """
    def __init__(self, dm_dim, latent_dim_in, latent_dim_out, hidden_dim, num_layers=2, use_kan=True, num_basis=8):
        """
        Initializes the LatentLayer.

        Args:
            dm_dim (int): Dimension of the deterministic memory input.
            latent_dim_in (int): Input latent dimension.
            latent_dim_out (int): Output latent dimension.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of hidden layers in the encoder.
            use_kan (bool): Whether to use KAN layers for hidden transformations.
            num_basis (int): Number of basis functions for KAN layers.
        """
        super().__init__()
        self.num_layers = num_layers
        self.use_kan = use_kan
        self.hidden_dim = hidden_dim
        
        # Initial projection: Conv2d for spatial-temporal structure preservation
        self.enc_in = nn.Conv2d(dm_dim + latent_dim_in, hidden_dim, 1)
        
        # Build hidden layers with optional KAN
        if use_kan:
            # KAN-based hidden layers for better non-linear modeling
            self.enc_hidden = nn.ModuleList()
            for i in range(num_layers):
                self.enc_hidden.append(
                    nn.Sequential(
                        nn.Conv2d(hidden_dim, hidden_dim, 1),  # Depthwise convolution
                        nn.BatchNorm2d(hidden_dim),
                    )
                )
            # Separate KAN projection layers for each position (applied after conv)
            self.kan_layers = nn.ModuleList([
                KANLayer(hidden_dim, hidden_dim, num_basis=num_basis)
                for _ in range(num_layers)
            ])
        else:
            # Standard Conv2d with ReLU
            layers = []
            for _ in range(num_layers):
                layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
                layers.append(nn.ReLU(inplace=True))
            self.enc_hidden = nn.Sequential(*layers)
        
        # Output projections for distribution parameters
        self.enc_out_1 = nn.Conv2d(hidden_dim, latent_dim_out, 1)  # mu
        self.enc_out_2 = nn.Conv2d(hidden_dim, latent_dim_out, 1)  # log_sigma
        
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the LatentLayer.

        Args:
            x (torch.Tensor): Input tensor combining deterministic memory and latent variables from the layer above.
                              Shape: (batch_size, dm_dim + latent_dim_in, num_nodes, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mu (torch.Tensor): The mean of the latent variable distribution.
                                     Shape: (batch_size, latent_dim_out, num_nodes, seq_len)
                - sigma (torch.Tensor): The standard deviation of the latent variable distribution.
                                        Shape: (batch_size, latent_dim_out, num_nodes, seq_len)
        """
        h = self.enc_in(x)
        
        if self.use_kan:
            # Apply KAN-enhanced layers
            for i, (conv_layer, kan_layer) in enumerate(zip(self.enc_hidden, self.kan_layers)):
                h = conv_layer(h)  # Conv2d + BatchNorm
                
                # Apply KAN layer on spatial-temporal dimensions
                b, c, n, t = h.shape
                h_reshaped = h.permute(0, 2, 3, 1).reshape(-1, c)  # Flatten spatial-temporal
                h_kan = kan_layer(h_reshaped)
                h = h_kan.reshape(b, n, t, c).permute(0, 3, 1, 2)  # Reshape back
                
                h = self.activation(h)
        else:
            # Standard processing
            for i in range(self.num_layers):
                h = self.enc_hidden[i](h)
        
        # Generate distribution parameters with clamping for stability
        mu = torch.clamp(self.enc_out_1(h), min=-10, max=10)
        log_sigma = torch.clamp(self.enc_out_2(h), min=-10, max=10)
        sigma = torch.exp(log_sigma) + 1e-6  # Add small epsilon for numerical stability
        
        return mu, sigma


class StochasticModel(nn.Module):
    """
    Manages a stack of LatentLayers to create a hierarchical stochastic model.
    This allows for capturing dependencies and uncertainties at multiple levels of abstraction.
    
    Enhanced with KAN layers for improved latent variable modeling.
    """
    def __init__(self, dm_dim, latent_dim, num_blocks=4, use_kan=True, num_basis=8):
        """
        Initializes the StochasticModel.

        Args:
            dm_dim (int): Dimension of the deterministic memory input.
            latent_dim (int): Dimension of the latent variables.
            num_blocks (int): Number of hierarchical blocks (and thus LatentLayers).
            use_kan (bool): Whether to use KAN layers in LatentLayers.
            num_basis (int): Number of basis functions for KAN layers.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Bottom n-1 layers receive deterministic memory and latent variables from the layer below.
        for _ in range(num_blocks-1):
            self.layers.append(LatentLayer(dm_dim, latent_dim, latent_dim, latent_dim, 2, 
                                          use_kan=use_kan, num_basis=num_basis))
        
        # The top layer only receives deterministic memory.
        self.layers.append(LatentLayer(dm_dim, 0, latent_dim, latent_dim, 2, 
                                      use_kan=use_kan, num_basis=num_basis))

    def reparameterize(self, mu, sigma):
        """
        Applies the reparameterization trick to sample from the latent distribution
        in a way that allows for backpropagation.

        Args:
            mu (torch.Tensor): The mean of the distribution.
            sigma (torch.Tensor): The standard deviation of the distribution.

        Returns:
            torch.Tensor: A sample from the latent distribution.
        """
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps * sigma

    def forward(self, d):
        """
        Forward pass for hierarchical latent variable inference.

        Args:
            d (torch.Tensor): A list of deterministic memory tensors from each block of the main model.
                              Shape: (num_blocks, batch_size, dm_dim, num_nodes, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - z (torch.Tensor): Stacked latent variable samples from all layers.
                                    Shape: (num_blocks, batch_size, latent_dim, num_nodes, seq_len)
                - mus (torch.Tensor): Stacked means from all layers.
                                      Shape: (num_blocks, batch_size, latent_dim, num_nodes, seq_len)
                - sigmas (torch.Tensor): Stacked standard deviations from all layers.
                                         Shape: (num_blocks, batch_size, latent_dim, num_nodes, seq_len)
        """
        # d: [num_blocks, b, c, n, t]
        _mu, _sigma = self.layers[-1](d[-1])
        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]

        for i in reversed(range(len(self.layers)-1)):
            _mu, _sigma = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))

        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return z, mus, sigmas

class SpatialAttention(nn.Module):
    """
    Computes multi-head self-attention across spatial nodes (e.g., air quality monitoring stations).
    This allows the model to learn spatial dependencies by attending to different sectors or regions.
    Enhanced with wind-aware dynamic bias for physically-grounded attention.
    """
    def __init__(self, dim, heads=4, qkv_bias=False, qk_scale=None, dropout=0., 
                 num_sectors=17, assignment=None, mask=None, use_wind_bias=False,
                 wind_speed_idx=12, wind_direction_idx=13, n_rings=3, radii=[50, 200, 500],
                 num_wind_sectors=8):
        """
        Initializes the SpatialAttention module.

        Args:
            dim (int): Input feature dimension.
            heads (int): Number of attention heads.
            qkv_bias (bool): Whether to include bias in the query, key, and value projections.
            qk_scale (float, optional): Scaling factor for the query-key dot product. Defaults to head_dim ** -0.5.
            dropout (float): Dropout rate.
            num_sectors (int): Number of spatial sectors.
            assignment (torch.Tensor): Assignment matrix mapping nodes to sectors.
            mask (torch.Tensor): Mask to prevent attention to certain sectors.
            use_wind_bias (bool): Whether to use precomputed wind-aware bias.
            wind_speed_idx (int): Unused - kept for compatibility.
            wind_direction_idx (int): Unused - kept for compatibility.
            n_rings (int): Unused - kept for compatibility.
            radii (list): Unused - kept for compatibility.
        """
        super().__init__()
        assert dim % heads == 0
        
        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sector = num_sectors
        self.assignment = assignment
        self.mask = mask
        self.use_wind_bias = use_wind_bias

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Wind-aware bias is precomputed in preprocessing, no module needed here

    def forward(self, x, wind_data=None):
        """
        Forward pass for spatial attention with optional precomputed wind bias.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size * seq_len, num_nodes, channels)
            wind_data (dict, optional): Dictionary containing 'wind_bias_precomputed' key with
                                       precomputed wind bias tensor.

        Returns:
            torch.Tensor: Output tensor with spatially attended features.
                          Shape: (batch_size * seq_len, num_nodes, channels)
        """
        B, N, C = x.shape
        
        pre_kv = torch.einsum('bnc,mnr->bmrc', x, self.assignment)
        pre_kv = pre_kv.reshape(-1, self.num_sector, C)
        pre_q = x.reshape(-1, 1, C)

        q = self.q_linear(pre_q).reshape(B*N, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_linear(pre_kv).reshape(B*N, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.reshape(B, N, self.num_heads, 1, self.num_sector) + self.relative_bias
        
        # Add precomputed wind bias if available
        if self.use_wind_bias and wind_data is not None and isinstance(wind_data, dict):
            wind_bias_precomputed = wind_data.get('wind_bias_precomputed', None)
            
            if wind_bias_precomputed is not None:
                # wind_bias_precomputed shape: (batch_size, seq_len, num_nodes, n_regions)
                # Reshape to (B, N, 1, 1, num_regions) where B = batch_size * seq_len
                wind_bias = wind_bias_precomputed.reshape(B, N, 1, 1, -1).float().to(attn.device)
                
                # Expand to match attention head dimension
                wind_bias = wind_bias.expand(-1, -1, self.num_heads, -1, -1)  # (B, N, heads, 1, num_regions)
                
                # Add to attention (broadcasting will align dimensions)
                attn = attn + wind_bias
        
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)
        attn = attn.masked_fill_(mask, float("-inf")).reshape(B * N, self.num_heads, 1, self.num_sector).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalAttention(nn.Module):
    """
    Computes multi-head self-attention across the temporal dimension (timesteps).
    This allows the model to learn temporal patterns and dependencies.
    """
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, 
                 dropout=0., causal=True, device=None):
        """
        Initializes the TemporalAttention module.

        Args:
            dim (int): Input feature dimension.
            heads (int): Number of attention heads.
            window_size (int): Local window size for attention. If > 0, attention is computed within windows.
            qkv_bias (bool): Whether to include bias in the query, key, and value projections.
            qk_scale (float, optional): Scaling factor for the query-key dot product. Defaults to head_dim ** -0.5.
            dropout (float): Dropout rate.
            causal (bool): Whether to use causal masking to prevent attending to future timesteps.
            device (torch.device): Device for the mask tensor.
        """
        super().__init__()
        assert dim % heads == 0
        
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        self.mask = torch.tril(torch.ones(window_size, window_size)).to(device)

    def forward(self, x):
        """
        Forward pass for temporal attention.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size * num_nodes, seq_len, channels)

        Returns:
            torch.Tensor: Output tensor with temporally attended features.
                          Shape: (batch_size * num_nodes, seq_len, channels)
        """
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape

        # Dynamically adjust the mask size to match the sequence length
        if self.mask.size(0) != T or self.mask.size(1) != T:
            self.mask = torch.tril(torch.ones(T, T)).to(x.device)

        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.causal:
            attn = attn.masked_fill_(self.mask == 0, float("-inf"))

        x = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class PreNorm(nn.Module):
    """
    Applies Layer Normalization before passing the input to a function (e.g., an attention or feedforward layer).
    This helps stabilize training.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Standard FeedForward layer for transformer blocks.
    Uses simple 2-layer MLP with GELU activation.
    KAN is not used here as it adds unnecessary complexity to already-simple layers.
    """
    def __init__(self, dim, hidden_dim, dropout=0., use_kan=False, num_basis=8):
        """
        Initializes the FeedForward layer.
        
        Args:
            dim (int): Input/output dimension.
            hidden_dim (int): Hidden dimension.
            dropout (float): Dropout rate.
            use_kan (bool): Unused - kept for compatibility. FF always uses standard MLP.
            num_basis (int): Unused - kept for compatibility.
        """
        super().__init__()
        # Standard MLP: simple, efficient, proven effective in transformers
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class DS_MSA(nn.Module):
    """
    Deterministic Spatial Multi-Head Self-Attention block.
    This block consists of a spatial attention layer followed by a feedforward network.
    Enhanced with wind-aware dynamic bias for physically-grounded spatial attention.
    """
    def __init__(self, dim, depth, heads, mlp_dim, assignment, mask, dropout=0.,
                 use_wind_bias=False, wind_speed_idx=12, wind_direction_idx=13,
                 n_rings=3, radii=[50, 200, 500], use_kan=False, num_basis=8):
        """
        Initializes the DS_MSA block.

        Args:
            dim (int): Input feature dimension.
            depth (int): Number of attention/feedforward layers to stack.
            heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension for the MLP.
            assignment (torch.Tensor): Assignment matrix for sectors.
            mask (torch.Tensor): Mask for attention.
            dropout (float): Dropout rate.
            use_wind_bias (bool): Whether to use wind-aware dynamic bias.
            wind_speed_idx (int): Index of wind speed feature.
            wind_direction_idx (int): Index of wind direction feature.
            n_rings (int): Number of dartboard rings.
            radii (list): Radii of dartboard rings.
            use_kan (bool): Unused - kept for compatibility.
            num_basis (int): Unused - kept for compatibility.
        """
        super().__init__()
        self.use_wind_bias = use_wind_bias
        self.wind_speed_idx = wind_speed_idx
        self.wind_direction_idx = wind_direction_idx
        
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim, heads=heads, dropout=dropout,
                               assignment=assignment, mask=mask,
                               num_sectors=assignment.shape[-1],
                               use_wind_bias=use_wind_bias,
                               wind_speed_idx=wind_speed_idx,
                               wind_direction_idx=wind_direction_idx,
                               n_rings=n_rings,
                               radii=radii,
                               num_wind_sectors=8),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, wind_data=None):
        """
        Forward pass for the DS_MSA block.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, channels, num_nodes, seq_len)
            wind_data (dict, optional): Dictionary with 'wind_bias_precomputed' tensor

        Returns:
            torch.Tensor: Output tensor with spatially attended features.
                          Shape: (batch_size, channels, num_nodes, seq_len)
        """
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)
        
        # Reshape wind_bias if present
        wind_data_reshaped = None
        if wind_data is not None and 'wind_bias_precomputed' in wind_data:
            wind_bias_precomputed = wind_data['wind_bias_precomputed']
            # wind_bias_precomputed shape: (batch, seq_len, num_nodes, num_regions)
            # Reshape to (batch*seq_len, num_nodes, num_regions) for processing
            wind_bias_reshaped = wind_bias_precomputed.reshape(b*t, n, -1)
            wind_data_reshaped = {'wind_bias_precomputed': wind_bias_reshaped}
        
        for attn, ff in self.layers:
            x = attn(x, wind_data_reshaped) + x
            x = ff(x) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


class CT_MSA(nn.Module):
    """
    Causal Temporal Multi-Head Self-Attention block.
    This block consists of a temporal attention layer followed by a feedforward network.
    """
    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0., device=None, 
                 use_kan=False, num_basis=8):
        """
        Initializes the CT_MSA block.

        Args:
            dim (int): Input feature dimension.
            depth (int): Number of attention/feedforward layers to stack.
            heads (int): Number of attention heads.
            window_size (int): Local window size for temporal attention.
            mlp_dim (int): Hidden dimension for the MLP.
            num_time (int): Number of time steps for positional embedding.
            dropout (float): Dropout rate.
            device (torch.device): Device for mask tensor.
            use_kan (bool): Unused - kept for compatibility.
            num_basis (int): Unused - kept for compatibility.
        """
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads, window_size=window_size,
                                dropout=dropout, device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Forward pass for the CT_MSA block.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, channels, num_nodes, seq_len)

        Returns:
            torch.Tensor: Output tensor with temporally attended features.
                          Shape: (batch_size, channels, num_nodes, seq_len)
        """
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b*n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class AirFormer(nn.Module):
    """
    The main AirFormer model for air quality forecasting.
    It combines spatial, temporal, and stochastic modules to capture complex spatiotemporal dependencies and uncertainties.
    Enhanced with wind-aware dynamic attention bias for improved pollution transport modeling.
    """
    def __init__(self, 
                 # Data
                 num_nodes, input_dim, output_dim, seq_len, horizon,
                 # Architecture
                 hidden_channels=32, end_channels=512, blocks=4, 
                 num_heads=2, mlp_expansion=2, dropout=0.3,
                 # Stages
                 spatial_flag=True, stochastic_flag=True,
                 # Dartboard
                 dartboard_path=None, dartboard=0,
                 # Local windows for CT-MSA
                 local_windows=None,
                 # Wind-aware bias
                 use_wind_bias=False, wind_speed_idx=12, wind_direction_idx=13,
                 n_wind_rings=3, wind_radii=[50, 200, 500],
                 # KAN-based enhancements
                 use_kan=False, kan_basis=4,
                 # Other
                 device=None):
        """
        Initializes the AirFormer model.

        Args:
            num_nodes (int): Number of spatial nodes (e.g., monitoring stations).
            input_dim (int): Number of input features per node.
            output_dim (int): Number of output features to predict.
            seq_len (int): Length of the input sequence.
            horizon (int): Length of the output forecast horizon.
            hidden_channels (int): Number of hidden channels in the model.
            end_channels (int): Number of channels in the final layers.
            blocks (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            mlp_expansion (int): Expansion factor for the hidden dimension of the MLP.
            dropout (float): Dropout rate.
            spatial_flag (bool): Whether to use the spatial attention module.
            stochastic_flag (bool): Whether to use the stochastic latent variable model.
            dartboard_path (str, optional): Path to the dartboard partition files.
            dartboard (int): Type of dartboard partition to use.
            local_windows (list, optional): List of local window sizes for CT-MSA in each block.
            use_wind_bias (bool): Whether to use wind-aware dynamic attention bias.
            wind_speed_idx (int): Feature index for wind speed.
            wind_direction_idx (int): Feature index for wind direction.
            n_wind_rings (int): Number of dartboard rings for wind bias.
            wind_radii (list): Radii for dartboard rings in wind bias.
            use_kan (bool): Whether to use Kolmogorov-Arnold Networks in LatentLayers only.
                           Not used in feedforward layers (simple MLP is more efficient).
            kan_basis (int): Number of basis functions for KAN layers (when used).
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.blocks = blocks
        self.spatial_flag = spatial_flag
        self.stochastic_flag = stochastic_flag
        self.device = device
        self.alpha = 10  # Coefficient for the KL divergence loss.
        self.use_wind_bias = use_wind_bias
        self.wind_speed_idx = wind_speed_idx
        self.wind_direction_idx = wind_direction_idx
        self.use_kan = use_kan
        self.kan_basis = kan_basis
        
        # Load dartboard partitioning for spatial attention
        dartboard_map = {0: '50-200', 1: '50-200-500', 2: '50'}
        if dartboard_path is None:
            dartboard_path = f'../DataFitting/Dataset/INDIAN_AIR/local_partition/{dartboard_map[dartboard]}'
        
        self.assignment = torch.from_numpy(np.load(f'{dartboard_path}/assignment.npy')).float().to(device)
        self.mask = torch.from_numpy(np.load(f'{dartboard_path}/mask.npy')).bool().to(device)
        
        # Input projection layer
        self.start_conv = nn.Conv2d(input_dim, hidden_channels, kernel_size=(1, 1))
        
        # Encoder blocks
        self.t_modules = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        for b in range(blocks):
            # Determine window size for temporal attention
            if local_windows is not None and b < len(local_windows):
                window_size = local_windows[b]
            else:
                window_size = seq_len // 2 ** (blocks - b - 1)
            
            # Temporal attention module with KAN-enhanced feedforward
            self.t_modules.append(CT_MSA(hidden_channels, depth=1, heads=num_heads,
                                        window_size=window_size, 
                                        mlp_dim=hidden_channels*mlp_expansion,
                                        num_time=seq_len, dropout=dropout, device=device,
                                        use_kan=use_kan, num_basis=kan_basis))
            
            # Spatial attention module with optional wind-aware bias and KAN-enhanced feedforward
            if spatial_flag:
                self.s_modules.append(DS_MSA(hidden_channels, depth=1, heads=num_heads,
                                            mlp_dim=hidden_channels*mlp_expansion,
                                            assignment=self.assignment, mask=self.mask,
                                            dropout=dropout,
                                            use_wind_bias=use_wind_bias,
                                            wind_speed_idx=wind_speed_idx,
                                            wind_direction_idx=wind_direction_idx,
                                            n_rings=n_wind_rings,
                                            radii=wind_radii,
                                            use_kan=use_kan, num_basis=kan_basis))
            
            self.bn.append(nn.BatchNorm2d(hidden_channels))
        
        # Stochastic models for capturing uncertainty with optional KAN-enhanced layers
        if stochastic_flag:
            self.generative_model = StochasticModel(hidden_channels, hidden_channels, blocks, 
                                                   use_kan=use_kan, num_basis=kan_basis)
            self.inference_model = StochasticModel(hidden_channels, hidden_channels, blocks,
                                                  use_kan=use_kan, num_basis=kan_basis)
            self.reconstruction_model = nn.Sequential(
                nn.Conv2d(hidden_channels*blocks, end_channels, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(end_channels, input_dim, kernel_size=(1, 1))
            )
        
        # Decoder to produce the final forecast
        in_channels = hidden_channels*blocks*2 if stochastic_flag else hidden_channels*blocks
        self.end_conv_1 = nn.Conv2d(in_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, horizon*output_dim, kernel_size=(1, 1))

    def forward(self, x, supports=None, wind_data=None):
        """
        Forward pass for the AirFormer model.

        Args:
            x (torch.Tensor): Input tensor. Shape: (batch_size, seq_len, num_nodes, input_dim)
            supports: Not used in this model, but kept for compatibility with other frameworks.
            wind_data (dict, optional): Dictionary containing 'wind_bias_precomputed' tensor.
                                       Shape: (batch_size, seq_len, num_nodes, n_regions)

        Returns:
            If stochastic_flag is True:
                Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                    - x_hat (torch.Tensor): The final forecast.
                                            Shape: (batch_size, horizon * output_dim, num_nodes, 1)
                    - x_rec (torch.Tensor): The reconstructed input.
                                            Shape: (batch_size, seq_len, num_nodes, input_dim)
                    - kl_loss (torch.Tensor): The KL divergence loss.
            If stochastic_flag is False:
                torch.Tensor: The final forecast.
                              Shape: (batch_size, horizon * output_dim, num_nodes, 1)
        """
        # x: [b, t, n, c]
        x = x.permute(0, 3, 2, 1)  # [b, c, n, t]
        x = self.start_conv(x)
        
        # Encoder
        d = []
        for i in range(self.blocks):
            if self.spatial_flag:
                # Pass entire wind_data dict (contains wind_bias_precomputed)
                x = self.s_modules[i](x, wind_data=wind_data)
            x = self.t_modules[i](x)
            x = self.bn[i](x)
            d.append(x)
        
        d = torch.stack(d)  # [num_blocks, b, c, n, t]
        
        # Stochastic modeling
        if self.stochastic_flag:
            # Shift deterministic memory for the generative model (prior)
            d_shift = [nn.functional.pad(d[i], pad=(1, 0))[..., :-1] for i in range(len(d))]
            d_shift = torch.stack(d_shift)
            
            z_p, mu_p, sigma_p = self.generative_model(d_shift)
            z_q, mu_q, sigma_q = self.inference_model(d)
            
            # Calculate KL divergence between prior and posterior
            p = torch.distributions.Normal(mu_p, sigma_p)
            q = torch.distributions.Normal(mu_q, sigma_q)
            kl_loss = torch.distributions.kl_divergence(q, p).mean() * self.alpha
            
            # Reshape latent variables
            num_blocks, B, C, N, T = d.shape
            z_p = z_p.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            z_q = z_q.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            
            # Reconstruct input from prior latent variables
            x_rec = self.reconstruction_model(z_p).permute(0, 3, 2, 1)
            
            # Generate prediction from deterministic memory and posterior latent variables
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = torch.cat([d[..., -1:], z_q[..., -1:]], dim=1)
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat, x_rec, kl_loss
        
        else:
            # Deterministic prediction
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = F.relu(d[..., -1:])
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat
