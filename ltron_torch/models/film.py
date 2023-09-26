import torch.nn as nn

class FILMLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dim=-1):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels*2)
        self.out_channels = out_channels
        self.dim = dim
    
    def forward(self, x1, x2):
        scale_bias = self.linear(x1)
        scale = scale_bias[...,:self.out_channels]
        bias = scale_bias[...,self.out_channels:]
        b = x2.shape[0]
        shape = [1 for _ in x2.shape]
        shape[0] = b
        shape[self.dim] = self.out_channels
        shape = tuple(shape)
        return x2 * scale.view(shape) + bias.view(shape)
