import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LatentLayer(nn.Module):
    # Step 1: LatentLayer encodes input features and produces latent variables (mu, sigma)
    def __init__(self, dm_dim, latent_dim_in, latent_dim_out, hidden_dim, num_layers=2):
        """
        Args:
            dm_dim: Dimension of deterministic memory input
            latent_dim_in: Input latent dimension
            latent_dim_out: Output latent dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super().__init__()
        self.num_layers = num_layers
        self.enc_in = nn.Conv2d(dm_dim + latent_dim_in, hidden_dim, 1)
        
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        self.enc_hidden = nn.Sequential(*layers)
        
        self.enc_out_1 = nn.Conv2d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv2d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        """
        Step 1.1: Forward pass for LatentLayer
        - Encodes input x and outputs mu, sigma for latent variable sampling
        """
        h = self.enc_in(x)
        for i in range(self.num_layers):
            h = self.enc_hidden[i](h)
        mu = torch.minimum(self.enc_out_1(h), torch.ones_like(h)*10)
        sigma = torch.minimum(self.enc_out_2(h), torch.ones_like(h)*10)
        return mu, sigma


class StochasticModel(nn.Module):
    # Step 2: StochasticModel manages a stack of LatentLayers for hierarchical latent variable modeling
    def __init__(self, dm_dim, latent_dim, num_blocks=4):
        """
        Args:
            dm_dim: Dimension of deterministic memory input
            latent_dim: Latent variable dimension
            num_blocks: Number of hierarchical blocks
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Bottom n-1 layers
        for _ in range(num_blocks-1):
            self.layers.append(LatentLayer(dm_dim, latent_dim, latent_dim, latent_dim, 2))
        
        # Top layer
        self.layers.append(LatentLayer(dm_dim, 0, latent_dim, latent_dim, 2))

    def reparameterize(self, mu, sigma):
        """
        Step 2.1: Reparameterization trick for sampling latent variables
        """
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps*sigma

    def forward(self, d):
        """
        Step 2.2: Forward pass for hierarchical latent variable inference
        - d: List of deterministic memory tensors from each block
        - Returns stacked latent variables, mus, and sigmas
        """
        # d: [num_blocks, b, c, n, t]
        _mu, _logsigma = self.layers[-1](d[-1])
        _sigma = torch.exp(_logsigma) + 1e-3
        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]

        for i in reversed(range(len(self.layers)-1)):
            _mu, _logsigma = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            _sigma = torch.exp(_logsigma) + 1e-3
            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))

        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return z, mus, sigmas

class SpatialAttention(nn.Module):
    # Step 3: SpatialAttention computes attention across spatial nodes (stations/sectors)
    def __init__(self, dim, heads=4, qkv_bias=False, qk_scale=None, dropout=0., 
                 num_sectors=17, assignment=None, mask=None):
        """
        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            num_sectors: Number of spatial sectors
            assignment: Assignment matrix for sector mapping
            mask: Mask for attention
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

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Step 3.1: Forward pass for spatial attention
        - x: Input tensor of shape (B, N, C)
        - Returns attended features across spatial sectors
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
        mask = self.mask.reshape(1, N, 1, 1, self.num_sector)
        attn = attn.masked_fill_(mask, float("-inf")).reshape(B * N, self.num_heads, 1, self.num_sector).softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TemporalAttention(nn.Module):
    # Step 4: TemporalAttention computes attention across temporal dimension (timesteps)
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False, qk_scale=None, 
                 dropout=0., causal=True, device=None):
        """
        Args:
            dim: Input feature dimension
            heads: Number of attention heads
            window_size: Local window size for attention
            causal: Whether to use causal masking (for autoregressive)
            device: Device for mask tensor
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
        Step 4.1: Forward pass for temporal attention
        - x: Input tensor of shape (B, T, C)
        - Returns attended features across time
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
    # Step 5: PreNorm applies LayerNorm before a given function (e.g., attention or feedforward)
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    # Step 6: FeedForward applies a two-layer MLP with GELU activation and dropout
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
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
    # Step 7: DS_MSA (Deterministic Spatial Multi-Head Self-Attention) stacks spatial attention and feedforward layers
    def __init__(self, dim, depth, heads, mlp_dim, assignment, mask, dropout=0.):
        """
        Args:
            dim: Input feature dimension
            depth: Number of attention/feedforward layers
            heads: Number of attention heads
            mlp_dim: Hidden dimension for MLP
            assignment: Assignment matrix for sectors
            mask: Mask for attention
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim, heads=heads, dropout=dropout,
                               assignment=assignment, mask=mask,
                               num_sectors=assignment.shape[-1]),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        """
        Step 7.1: Forward pass for DS_MSA
        - x: Input tensor of shape (B, C, N, T)
        - Returns spatially attended features
        """
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b*t, n, c)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x


class CT_MSA(nn.Module):
    # Step 8: CT_MSA (Causal Temporal Multi-Head Self-Attention) stacks temporal attention and feedforward layers
    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0., device=None):
        """
        Args:
            dim: Input feature dimension
            depth: Number of attention/feedforward layers
            heads: Number of attention heads
            window_size: Local window size for temporal attention
            mlp_dim: Hidden dimension for MLP
            num_time: Number of time steps
            device: Device for mask tensor
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
        Step 8.1: Forward pass for CT_MSA
        - x: Input tensor of shape (B, C, N, T)
        - Returns temporally attended features
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
    # Step 9: AirFormer is the main model class combining spatial, temporal, and stochastic modules for air quality forecasting
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
                 # Other
                 device=None):
        """
        Step 9.1: Initialize AirFormer model
        - Combines spatial, temporal, and stochastic modules for spatiotemporal forecasting
        Args:
            num_nodes: Number of spatial nodes (stations)
            input_dim: Number of input features per node
            output_dim: Number of output features
            seq_len: Input sequence length
            horizon: Output forecast horizon
            hidden_channels: Hidden dimension size
            end_channels: Final layer dimension
            blocks: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_expansion: Expansion factor for MLP
            dropout: Dropout rate
            spatial_flag: Enable spatial attention
            stochastic_flag: Enable stochastic latent variables
            dartboard_path: Path to dartboard partition files
            dartboard: Dartboard partition type
            local_windows: Local window sizes for CT-MSA
            device: Device for computation
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
        self.alpha = 10  # KL loss coefficient
        
        # Load dartboard
        dartboard_map = {0: '50-200', 1: '50-200-500', 2: '50'}
        if dartboard_path is None:
            dartboard_path = f'../DataFitting/Dataset/INDIAN_AIR/local_partition/{dartboard_map[dartboard]}'
        
        self.assignment = torch.from_numpy(np.load(f'{dartboard_path}/assignment.npy')).float().to(device)
        self.mask = torch.from_numpy(np.load(f'{dartboard_path}/mask.npy')).bool().to(device)
        
        # Input projection
        self.start_conv = nn.Conv2d(input_dim, hidden_channels, kernel_size=(1, 1))
        
        # Encoder blocks
        self.t_modules = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.bn = nn.ModuleList()
        
        for b in range(blocks):
            # Use paper's local windows if provided, else calculate dynamically
            if local_windows is not None and b < len(local_windows):
                window_size = local_windows[b]
            else:
                window_size = seq_len // 2 ** (blocks - b - 1)
            
            # Temporal
            self.t_modules.append(CT_MSA(hidden_channels, depth=1, heads=num_heads,
                                        window_size=window_size, 
                                        mlp_dim=hidden_channels*mlp_expansion,
                                        num_time=seq_len, dropout=dropout, device=device))
            
            # Spatial
            if spatial_flag:
                self.s_modules.append(DS_MSA(hidden_channels, depth=1, heads=num_heads,
                                            mlp_dim=hidden_channels*mlp_expansion,
                                            assignment=self.assignment, mask=self.mask,
                                            dropout=dropout))
            
            self.bn.append(nn.BatchNorm2d(hidden_channels))
        
        # Stochastic models
        if stochastic_flag:
            self.generative_model = StochasticModel(hidden_channels, hidden_channels, blocks)
            self.inference_model = StochasticModel(hidden_channels, hidden_channels, blocks)
            self.reconstruction_model = nn.Sequential(
                nn.Conv2d(hidden_channels*blocks, end_channels, kernel_size=(1, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(end_channels, input_dim, kernel_size=(1, 1))
            )
        
        # Decoder
        in_channels = hidden_channels*blocks*2 if stochastic_flag else hidden_channels*blocks
        self.end_conv_1 = nn.Conv2d(in_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, horizon*output_dim, kernel_size=(1, 1))

    def forward(self, x, supports=None):
        """
        Step 9.2: Forward pass for AirFormer
        - x: Input tensor of shape (B, seq_len, N, input_dim)
        - Returns forecasted air quality values
        """
        # x: [b, t, n, c]
        x = x.permute(0, 3, 2, 1)  # [b, c, n, t]
        x = self.start_conv(x)
        
        # Encoder
        d = []
        for i in range(self.blocks):
            if self.spatial_flag:
                x = self.s_modules[i](x)
            x = self.t_modules[i](x)
            x = self.bn[i](x)
            d.append(x)
        
        d = torch.stack(d)  # [num_blocks, b, c, n, t]
        
        # Stochastic encoding
        if self.stochastic_flag:
            # Shift for generative model
            d_shift = [nn.functional.pad(d[i], pad=(1, 0))[..., :-1] for i in range(len(d))]
            d_shift = torch.stack(d_shift)
            
            z_p, mu_p, sigma_p = self.generative_model(d_shift)
            z_q, mu_q, sigma_q = self.inference_model(d)
            
            # KL divergence
            p = torch.distributions.Normal(mu_p, sigma_p)
            q = torch.distributions.Normal(mu_q, sigma_q)
            kl_loss = torch.distributions.kl_divergence(q, p).mean() * self.alpha
            
            # Reshape
            num_blocks, B, C, N, T = d.shape
            z_p = z_p.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            z_q = z_q.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            
            # Reconstruction
            x_rec = self.reconstruction_model(z_p).permute(0, 3, 2, 1)
            
            # Prediction
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = torch.cat([d[..., -1:], z_q[..., -1:]], dim=1)
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat, x_rec, kl_loss
        
        else:
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = F.relu(d[..., -1:])
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
            return x_hat
