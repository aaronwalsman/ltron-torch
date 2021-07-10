import math

import torch

def positional_encoding(channels, *shape, dtype=torch.float):
    encoding = torch.zeros((*shape, channels), dtype=dtype)
    dims = len(shape)
    assert channels % dims == 0
    c = channels // dims
    
    wavelength = (
        torch.exp(-math.log(10000.) * torch.arange(0, c, 2, dtype=dtype) / c))
    
    for i, s in enumerate(shape):
        t = torch.arange(0, s, dtype=dtype).unsqueeze(1) * wavelength
        t_shape = [1] * dims + [c//2]
        t_shape[i] = s
        t = t.view(t_shape)
        encoding[..., i*c:(i+1)*c:2] = torch.sin(t)
        encoding[..., i*c+1:(i+1)*c:2] = torch.cos(t)
    
    return encoding
