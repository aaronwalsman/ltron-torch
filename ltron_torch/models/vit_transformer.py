import math

import torch

from ltron_torch.models.positional_encoding import (
    PositionalEncoding1D,
    SingleEncoding,
)

class VITTransformer(torch.nn.Module):
    def __init__(self,
        channels=256,
        image_block_size=16,
        transformer_layers=6,
        transformer_heads=4,
        transformer_dropout=0.5,
        compute_single=False,
    ):
        super(VITTransformer, self).__init__()
        self.channels = channels
        self.image_block_size = image_block_size
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        self.compute_single = compute_single
        
        self.positional_encoding = PositionalEncoding1D(self.channels)
        if compute_single:
            self.single_encoding = SingleEncoding(self.channels)
        
        self.block_encoder = torch.nn.Conv2d(
            3, channels, self.image_block_size, self.image_block_size)
        
        transformer_layer = torch.nn.TransformerEncoderLayer(
            channels,
            self.transformer_heads,
            channels,
            self.transformer_dropout
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_layer, self.transformer_layers)
    
    def forward(self, x):
        x = self.block_encoder(x)
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(2,0,1)
        if self.compute_single:
            single_embedding = self.single_encoding(b)
            x = torch.cat((single_embedding, x), dim=0)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        
        if self.compute_single:
            x_dense = x[1:].permute(1,2,0).view(b, -1, h, w)
            x_single = x[0]
        else:
            x_dense = x.permute(1,2,0).view(b, -1, h, w)
            x_single = None
        
        return x_dense, x_single
