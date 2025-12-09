import torch
import torch.nn as nn

lstm = nn.LSTM(
    input_size=1,
    hidden_size=32,
    num_layers=1,
    batch_first=True,
)

x = torch.randn(32, 20, 1)
out, (h_T, c_T) = lstm(x)