import math

import torch

'''
def positional_encoding_1d(dim, max_len):
    encoding = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    encoding[:, 0::2] = torch.sin(position * div_term)
    encoding[:, 1::2] = torch.cos(position * div_term)
    
    return encoding
'''
# weird constant that everybody uses that's passed around everywhere

#def positional_encoding_1d(max_1, channels):
#    encoding = torch.zeros(max_1, channels)
#    d = k * torch.arange(0, c, 2).float() / channels
#    d = torch.exp(d)
#    
#    p = torch.arange(0, max_1, dtype=torch.float).unsqueeze(1) * d
#    
#    encoding[:,:,0:channels:2] = torch.sin(p)
#    encoding[:,:,1:channels:2] = torch.cos(p)
#    
#    return encoding
#
#def positional_encoding_2d(max_1, max_2, channels):
#    encoding = torch.zeros(max_1, max_2, channels)
#    cc = channels//2
#    d = k * torch.arange(0, cc, 2).float() / cc
#    d = torch.exp(d)
#    
#    p1 = torch.arange(0, max_1, dtype=torch.float).unsqueeze(1).unsqueeze(2) * d
#    p2 = torch.arange(0, max_2, dtype=torch.float).unsqueeze(0).unsqueeze(2) * d
#    
#    encoding[:,:,0:cc:2] = torch.sin(p1)
#    encoding[:,:,1:cc:2] = torch.cos(p1)
#    encoding[:,:,cc::2] = torch.sin(p2)
#    encoding[:,:,cc+1::2] = torch.cos(p2)
#    
#    return encoding

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

'''
class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        encoding = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        #encoding = encoding.unsqueeze(0).transpose(0, 1)
        encoding = encoding.unsqueeze(1)
        self.register_buffer('encoding', encoding)
    
    def forward(self, x):
        x = x + self.encoding[:x.shape[0], :]
        return self.dropout(x)

class PositionalEncoding2D(torch.nn.Module):
    def __init__(self, dim, dropout=0.1, max_len1=5000, max_len2=64):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        encoding = torch.zeros(max_len1, max_len2, dim)
        half_dim = dim//2
        position1 = torch.arange(0, max_len1, dtype=torch.float).view(
            max_len1, 1, 1)
        position2 = torch.arange(0, max_len2, dtype=torch.float).view(
            1, max_len2, 1)
        div_term = torch.exp(
            torch.arange(0, half_dim, 2).float() *
            (-math.log(10000.0) / half_dim)
        )
        encoding[:, :, 0:half_dim:2] =  torch.sin(position1 * div_term)
        encoding[:, :, 1:half_dim:2] =  torch.cos(position1 * div_term)
        encoding[:, :, half_dim::2] =   torch.sin(position2 * div_term)
        encoding[:, :, half_dim+1::2] = torch.cos(position2 * div_term)
        encoding = encoding.unsqueeze(2) # batch_size dimension
        self.register_buffer('encoding', encoding)
    
    def forward(self, x):
        x = x + self.encoding[:x.shape[0], :x.shape[1], :]
        return self.dropout(x)

class SingleEncoding(torch.nn.Module):
    def __init__(self, dim, dropout=0.1):
        super(SingleEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        encoding = torch.zeros(1,1,dim).uniform_(-1,1)
        self.register_parameter('encoding', torch.nn.Parameter(encoding))
    
    def forward(self, n):
        dim = self.encoding.shape[-1]
        encoding = self.encoding.expand(1,n,dim)
        return self.dropout(encoding)
'''
