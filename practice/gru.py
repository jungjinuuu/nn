import torch
import torch.nn as nn

gru = nn.GRU(
    input_size=1,
    hidden_size=32,
    batch_first=True
)

x = torch.randn(32, 20, 1)
out, h_T = gru(x)

print(out.shape, h_T.shape)