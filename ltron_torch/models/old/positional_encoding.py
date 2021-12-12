import math

import torch
from torch.nn import Module, Linear, Dropout, ParameterList, ModuleList

from ltron_torch.models.parameter import NoWeightDecayParameter


def sinusoid_positional_encoding(channels, *shape, dtype=torch.float):
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

def compressed_causal_mask(i, pad, causal_dim):
    i = i[:,:,causal_dim]
    n, b = i.shape
    causal_mask = i.view(n, 1, b) < i.view(1, n, b)
    square = torch.max(torch.arange(n).view(n,1), torch.arange(n).view(1,n))
    square = square.to(pad.device)
    padding_mask = square.unsqueeze(-1) >= pad.view(1,1,b)
    causal_mask = causal_mask | padding_mask
    
    # make the diagonal False to avoid nan
    causal_mask[torch.arange(n), torch.arange(n), :] = False
    
    return causal_mask

class FactoredLearnedRelativePositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims):
        super(FactoredLearnedRelativePositionalEncoding, self).__init__()
        encoding_shape = tuple(
            d if (i in causal_dims) else d*2-1
            for i, d in enumerate(data_shape)
        )
        self.register_buffer('center_offset', torch.LongTensor([
            0 if (i in causal_dims) else d-1
            for i,d in enumerate(data_shape)
        ]))
        self.dim_encodings = ParameterList([
            NoWeightDecayParameter(torch.zeros(d, channels))
            for d in encoding_shape
        ])

    def forward(self, i):
        n, b, d = i.shape
        r = i.view(n,1,b,d) - i.view(1,n,b,d)
        r = r + self.center_offset.view(1,1,1,d)
        #cm = torch.max(r < 0, dim=-1)[0]
        r = torch.clamp(r, min=0)
        
        pe = sum(p[r[:,:,:,j]] for j,p in enumerate(self.dim_encodings))
        
        # TMP
        #cm = torch.zeros((n, n, b), dtype=torch.bool, device=i.device)
        cm = compressed_causal_mask(i)
        
        return pe, cm

'''
class NoBatchedFactoredPositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims, learned):
        super(FactoredPositionalEncoding, self).__init__()
        self.learned = learned
        if self.learned:
            self.dim_encodings = ParameterList([
                NoWeightDecayParameter(torch.zeros(d, channels))
                for d in data_shape
            ])
        else:
            positional_linears = []
            for i, d in enumerate(data_shape):
                self.register_buffer(
                    'pe_%i'%i, sinusoid_positional_encoding(channels, d))
                positional_linears.append(Linear(channels, channels))
            self.positional_linears = ModuleList(positional_linears)
            
        self.causal_dims = causal_dims
        self.d = len(data_shape)

    def forward(self, i):
        n, d = i.shape
        
        if self.learned:
            pe = sum(p[i[:,j]] for j,p in enumerate(self.dim_encodings))
        else:
            pe = sum(
                self.positional_linears[j](getattr(self, 'pe_%i'%j)[i[:,j]])
                for j in range(self.d)
            )
        
        # TMP
        cm = torch.zeros((n, n), dtype=torch.bool, device=i.device)
        
        return pe, cm
'''

class FactoredPositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dim, learned):
        super(FactoredPositionalEncoding, self).__init__()
        self.learned = learned
        if self.learned:
            self.dim_encodings = ParameterList([
                NoWeightDecayParameter(torch.zeros(d+1, channels))
                for d in data_shape
            ])
        else:
            positional_linears = []
            self.register_buffer(
                'pe', sinusoid_positional_encoding(channels, max(data_shape)+1))
            for i, d in enumerate(data_shape):
                positional_linears.append(Linear(channels, channels))
            self.positional_linears = ModuleList(positional_linears)
            
        self.causal_dim = causal_dim
        self.d = len(data_shape)

    def forward(self, i, pad_lengths):
        n, b, d = i.shape
        
        if self.learned:
            pe = sum(p[i[:,:,j]+1] for j,p in enumerate(self.dim_encodings))
        else:
            pe = sum(
                self.positional_linears[j](getattr(self, 'pe')[i[:,:,j]+1])
                for j in range(self.d)
            )

        cm = compressed_causal_mask(i, pad_lengths, causal_dim=self.causal_dim)
        
        return pe, cm

class PositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims, learned):
        super(PositionalEncoding, self).__init__()
        self.learned = learned
        if self.learned:
            self.pe = NoWeightDecayParameter(torch.zeros(*data_shape, channels))
        else:
            self.register_buffer(
                'pe', sinusoid_positional_encoding(channels, data_shape))
        
        self.causal_dims = causal_dims
        self.d = len(data_shape)
    
    def forward(self, i):
        n, b, d = i.shape
        
        dimension_index = tuple([i[:,:,j].view(-1) for j in range(self.d)])
        pe = self.pe[dimension_index].view(n, b, -1)
        
        # TMP
        cm = torch.zeros((n, n, b), dtype=torch.bool, device=i.device)
        
        return pe, cm

'''
class FactoredLearnedPositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims):
        super(FactoredLearnedPositionalEncoding, self).__init__()
        self.dim_encodings = ParameterList([
            NoWeightDecayParameter(torch.zeros(d, channels))
            for d in data_shape
        ])
        self.causal_dims = causal_dims

    def forward(self, i):
        n, b, d = i.shape
        
        pe = sum(p[i[:,:,j]] for j,p in enumerate(self.dim_encodings))

        # TMP
        cm = torch.zeros((n, n, b), dtype=torch.bool, device=i.device)

        return pe, cm

class FactoredSinusoidPositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims):
        super(FactoredSinusoidPositionalEncoding, self).__init__()
        positional_linears = []
        for i, d in enumerate(data_shape):
            self.register_buffer(
                'pe_%i'%i, sinusoid_positional_encoding(channels, d))
            positional_linears.append(Linear(channels, channels))
        self.positional_linears = ModuleList(positional_linears)

        self.causal_dims = causal_dims
        self.d = len(data_shape)

    def forward(self, i):
        n, b, d = i.shape
        pe = sum(
            self.positional_linears[j](getattr(self, 'pe_%i'%j)[i[:,:,j]])
            for j in range(self.d)
        )

        # TMP
        cm = torch.zeros((n, n, b), dtype=torch.bool, device=i.device)

        return pe, cm
'''
