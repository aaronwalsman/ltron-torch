import torch
from torch.nn import Module, Identity, Dropout

from ltron_torch.models.transformer import (
    TransformerConfig, Transformer)

class Transformoencoder(Module):
    def __init__(self, encoder_config, decoder_config):
        super(Transformoencoder, self).__init__()
        self.hidden_tokens = encoder_config.decoder_tokens
        
        self.encoder = Transformer(encoder_config)
        self.decoder = Transformer(decoder_config)
        self.decoder.embedding.embedding = Identity()
        
        #self.dropout = Dropout(0.)
    
    def forward(self, x, pause=False):
        #print('==============================================')
        #print('enc')
        x = self.encoder(x)
        
        #s = x[:self.hidden_tokens]
        #d = x[self.hidden_tokens:]
        #d = self.dropout(d)
        #x = torch.cat((s, d), dim=0)
        
        t, b, c = x.shape
        
        x = x.view(t, 1, b, c)
        x = self.decoder(x)
        
        return x
