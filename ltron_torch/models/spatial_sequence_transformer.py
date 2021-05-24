import math

import torch

from ltron_torch.models.positional_encoding import (
    PositionalEncoding1D,
    PositionalEncoding2D,
    SingleEncoding,
)

class SpatialSequenceTransformer(torch.nn.Module):
    def __init__(self,
        channels,
        hidden_channels=256,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dropout=0.5,
        max_seq_len=256,
        max_spatial_len=64,
        compute_single=False,
    ):
        super(SpatialSequenceTransformer, self).__init__()
        self.channels = channels
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        self.compute_single = compute_single
        
        self.positional_encoding = PositionalEncoding2D(
            channels, max_len1=max_seq_len, max_len2=max_spatial_len)
        if compute_single:
            self.single_encoding = SingleEncoding(self.channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            self.transformer_heads,
            channels,
            self.transformer_dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_layer, self.transformer_layers)
    
    def forward(self, x, seq_mask=None, padding_mask=None):
        s,b,c,h,w = x.shape
        x = x.view(s,b,c,h*w).permute(0,3,1,2)
        x = self.positional_encoding(x)
        x = x.reshape(s*h*w,b,c)
        if seq_mask is not None:
            seq_mask = seq_mask.unsqueeze(1).expand(s,h*w,s)
            seq_mask = seq_mask.unsqueeze(3).expand(s,h*w,s,h*w)
            seq_mask = seq_mask.reshape(s*h*w, s*h*w)
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(2).expand(b,s,h*w).reshape(
                b,s*h*w)
        x = self.transformer_encoder(
            x,
            mask=seq_mask,
            src_key_padding_mask=padding_mask,
        )
        x = x.view(s,h*w,b,c).permute(0,2,3,1).reshape(s,b,c,h,w)
        return x


class SequenceTransformer(torch.nn.Module):
    def __init__(self,
        channels,
        hidden_channels=256,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dropout=0.5,
        max_seq_len=256,
    ):
        super(SequenceTransformer, self).__init__()
        self.channels = channels
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        
        self.positional_encoding = PositionalEncoding1D(
            channels, max_len=max_seq_len)
        self.single_encoding = SingleEncoding(self.channels)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            self.transformer_heads,
            channels,
            self.transformer_dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_layer, self.transformer_layers)
        
        stdv = 1./math.sqrt(channels)
        self.global_token = torch.nn.Parameter(
            torch.zeros(1,1,channels).uniform_(-stdv, stdv)
        )
    
    def forward(self, x, seq_mask=None, padding_mask=None):
        s,b,c = x.shape
        g = self.global_token.expand(1,b,c)
        x = torch.cat((g, x), dim=0)
        #x = x.view(s,b,c,h*w).permute(0,3,1,2)
        x = self.positional_encoding(x)
        #x = x.reshape(s*h*w,b,c)
        #if seq_mask is not None:
        #    seq_mask = seq_mask.unsqueeze(1).expand(s,h*w,s)
        #    seq_mask = seq_mask.unsqueeze(3).expand(s,h*w,s,h*w)
        #    seq_mask = seq_mask.reshape(s*h*w, s*h*w)
        #if padding_mask is not None:
        #    padding_mask = padding_mask.unsqueeze(2).expand(b,s,h*w).reshape(
        #        b,s*h*w)
        x = self.transformer_encoder(
            x,
            mask=seq_mask,
            src_key_padding_mask=padding_mask,
        )
        #x = x.view(s,h*w,b,c).permute(0,2,3,1).reshape(s,b,c,h,w)
        g = x[0]
        x = x[1:]
        return x, g
