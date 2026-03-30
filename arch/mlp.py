import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links and LayerNorm."""

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(hidden_dim)

        if input_dim != hidden_dim:
            self.residual_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[..., D] -> [..., hidden_dim]"""
        residual = self.residual_proj(x)
        out = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.norm(out + residual)
