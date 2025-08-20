import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor
import os
import logging
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class ResidualAdd(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return  x + self.f(x)
    
class EEGProjectLayer(nn.Module):
    def __init__(self, z_dim, c_num, timesteps, drop_proj=0.3):
        super(EEGProjectLayer, self).__init__()
        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.input_dim = self.c_num * (self.timesteps[1]-self.timesteps[0])
        proj_dim = z_dim

        self.model = nn.Sequential(nn.Linear(self.input_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        x = x.view(x.shape[0], self.input_dim)
        x = self.model(x)
        return x

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        return x
    
class BaseModel(nn.Module):
    def __init__(self,  z_dim, c_num, timesteps, embedding_dim = 1440):
        super(BaseModel, self).__init__()

        self.backbone = None
        self.project = nn.Sequential(
            FlattenHead(),
            nn.Linear(embedding_dim, z_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(z_dim, z_dim),
                nn.Dropout(0.5))),
            nn.LayerNorm(z_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.project(x)
        return x

class Shallownet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.Dropout(0.5),
            )
    
class Deepnet(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps,embedding_dim = 1400)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 25, (1, 10), (1, 1)),
                nn.Conv2d(25, 25, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(25),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(25, 50, (1, 10), (1, 1)),
                nn.BatchNorm2d(50),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(50, 100, (1, 10), (1, 1)),
                nn.BatchNorm2d(100),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),

                nn.Conv2d(100, 200, (1, 10), (1, 1)),
                nn.BatchNorm2d(200),
                nn.ELU(),
                nn.MaxPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
            )
        
class EEGnet(BaseModel):
    def __init__(self,  z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps, embedding_dim = 1248)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 8, (1, 64), (1, 1)),
                nn.BatchNorm2d(8),
                nn.Conv2d(8, 16, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout(0.5),
                nn.Conv2d(16, 16, (1, 16), (1, 1)),
                nn.BatchNorm2d(16), 
                nn.ELU(),
                # nn.AvgPool2d((1, 2), (1, 2)),
                nn.Dropout2d(0.5)
            )
        
class TSconv(BaseModel):
    def __init__(self, z_dim, c_num, timesteps):
        super().__init__(z_dim, c_num, timesteps)
        self.backbone = nn.Sequential(
                nn.Conv2d(1, 40, (1, 25), (1, 1)),
                nn.AvgPool2d((1, 51), (1, 5)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Conv2d(40, 40, (c_num, 1), (1, 1)),
                nn.BatchNorm2d(40),
                nn.ELU(),
                nn.Dropout(0.5),
            )
    
class EEGTransformerProjector(nn.Module):
    """
    Transformer encoder for EEG sequences.

    - Treats EEG as a sequence over time with vectors of channel readings.
    - Projects per-time-step channel vector C -> d_model.
    - Adds a learnable [CLS] token and learnable positional embeddings.
    - Runs a TransformerEncoder and uses the [CLS] representation.
    - Projects to `z_dim` with a small residual MLP head and LayerNorm.

    Expected input: tensor of shape [batch_size, num_channels, num_timesteps].
    Output: tensor of shape [batch_size, z_dim].
    """

    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: list,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.c_num = c_num
        self.timesteps = timesteps

        self.seq_len = int(self.timesteps[1] - self.timesteps[0])
        self.d_model = int(d_model)

        if dim_feedforward is None:
            dim_feedforward = self.d_model * 4

        # 1) Per-time-step projection: C -> d_model
        self.input_projection = nn.Sequential(
            nn.Linear(self.c_num, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2) Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.positional_embedding = nn.Parameter(
            torch.randn(1, self.seq_len + 1, self.d_model) * 0.02
        )
        self.input_layernorm = nn.LayerNorm(self.d_model)

        # 3) Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4) Output projection to z_dim with residual MLP and LayerNorm
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.z_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(self.z_dim, self.z_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(self.z_dim),
        )

        # For contrastive loss scaling (kept for compatibility with existing training loop)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

        # Initialize parameters
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        # Xavier init for linear layers in projections
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [batch, C, T]
        returns: [batch, z_dim]
        """
        batch_size = x.shape[0]

        # Rearrange to [batch, T, C]
        x = x.transpose(1, 2).contiguous()

        # Ensure time dimension matches configured window
        if x.size(1) != self.seq_len:
            # If input time dimension differs, center-crop or pad to match
            # Here we simple slice or pad zeros at the end for safety
            if x.size(1) > self.seq_len:
                x = x[:, : self.seq_len, :]
            else:
                pad_len = self.seq_len - x.size(1)
                pad = x.new_zeros(batch_size, pad_len, x.size(2))
                x = torch.cat([x, pad], dim=1)

        # Project per time step: [B, T, C] -> [B, T, d_model]
        x = self.input_projection(x)
        x = self.input_layernorm(x)

        # Prepend [CLS]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, T+1, d_model]

        # Add positional embeddings
        x = x + self.positional_embedding

        # Transformer encoding
        x = self.transformer(x)

        # Take [CLS] token
        cls_representation = x[:, 0, :]  # [B, d_model]

        # Project to z_dim
        z = self.output_projection(cls_representation)
        return z


class TemporalConvTransformerEEG(nn.Module):
    """
    Temporal feature extractor for EEG.

    Pipeline:
    - 1D temporal convolutions to downsample and extract local features
    - TransformerEncoder over the temporal dimension
    - Global average pooling over time
    - Projection MLP to `z_dim` (default aligns with vision embedding dim)

    Expected input: [batch_size, num_channels, num_timesteps]
    Output: [batch_size, z_dim]
    """

    def __init__(
        self,
        z_dim: int,
        c_num: int,
        timesteps: list,
        hidden_dim: int = 256,
        proj_dim: int | None = None,
        nhead: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.z_dim = int(z_dim)
        self.c_num = int(c_num)
        self.timesteps = timesteps
        if proj_dim is None:
            proj_dim = self.z_dim

        # Temporal conv backbone
        self.conv = nn.Sequential(
            nn.Conv1d(self.c_num, 64, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(128, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.GELU(),
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, proj_dim),
        )

        # For contrastive loss scaling (kept for compatibility with existing training loop)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.softplus = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        """
        x: [batch, channels, time]
        returns: [batch, z_dim]
        """
        h = self.conv(x)                 # [B, hidden_dim, T']
        h = h.permute(0, 2, 1)           # [B, T', hidden_dim]
        h = self.transformer(h)          # [B, T', hidden_dim]
        h = h.mean(dim=1)                # [B, hidden_dim]
        z = self.proj(h)                 # [B, z_dim]
        # Normalize; training loop also normalizes, but this is harmless
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-6)
        return z
