import math

import torch

from ltron_torch.models.positional_encoding import (
    PositionalEncoding1D,
    SingleEncoding,
)

class SpatialSequenceTransformer(torch.nn.Module):
    def __init__(self,
        channels,
        hidden_channels=256,
        transformer_layers=6,
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
    
    def forward(self, x):
        s,b,c,h,w = x.shape
        x = x.view(s,b,c,h*w).permute(0,3,1,2)
        x = self.positional_encoding(x)
        x = x.view(s*h*w,b,c)
        x = self.transformer_encoder(x)
        x = x.view(s,h*w,b,c).permute(0,2,3,1).view(s,b,c,h,w)
        return x
