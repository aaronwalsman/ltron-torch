import torch
from torch.nn import Module, Identity

from ltron_torch.models.transformer import (
    TransformerConfig, TokenMapSequenceEncoder)

class Transformoencoder(Module):
    def __init__(self, encoder_config, decoder_config):
        super(Transformoencoder, self).__init__()
        
        self.encoder = TokenMapSequenceEncoder(encoder_config)
        self.decoder = TokenMapSequenceEncoder(decoder_config)
        
        self.decoder.embedding.embedding = Identity()
    
    
    def forward(self, x, pause=False):
        x = self.encoder(x)
        s, b, c = x.shape
        x = x.view(s, 1, b, c)
        x = self.decoder(x)
        
        return x
