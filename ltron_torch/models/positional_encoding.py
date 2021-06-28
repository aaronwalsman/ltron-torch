import math

import torch

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
