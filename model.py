import torch
import torch.nn as nn
from torch import Tensor

class GRUPredictor(nn.Module):
    """
    Simple 1-layer GRU for next-step time-series prediction.

    input: (B, T, 1)
    output: y_pred (B, 1), h_all (B, T, H)
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: Tensor):
        """
        x: (B, T, 1)
        returns:
            y: (B, 1)
            h_all: (B, T, H)
        """
        out, h_T = self.gru(x)
        last_h = out[:, -1, :]  # (B, H)
        y = self.fc(last_h)     # (B, 1)
        return y, out