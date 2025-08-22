import torch
import torch.nn as nn


class EEGToIPAdapterProjection(nn.Module):
    """
    Map EEG encoder embeddings to IP-Adapter ViT-H image embedding space.
    Assumes input shape [batch, eeg_dim] and outputs [batch, ip_dim].
    """

    def __init__(
        self,
        eeg_dim: int,
        ip_adapter_dim: int = 1024,
        hidden_dim: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_dim = eeg_dim
        for _ in range(num_layers - 1):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, ip_adapter_dim)]
        self.mlp = nn.Sequential(*layers)

        # Optional learnable scale to match cosine-normed spaces
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, eeg_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.mlp(eeg_embeddings)
        # Normalize to unit sphere to better match CLIP/IP embedding geometry
        x = nn.functional.normalize(x, dim=-1) * self.output_scale
        return x
