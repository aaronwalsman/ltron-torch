from torch.nn import (
    Module, ModuleList, Sequential, Linear, Dropout, ReLU, GELU,
    Embedding, LayerNorm,
)

from ltron_torch.config import Config
from ltron_torch.models.memory_attention import MemoryAttention
from ltron_torch.models.mask import padded_causal_mask

class TransformerConfig(Config):
    nonlinearity = 'gelu'
    num_blocks = 12
    channels = 768
    residual_channels = None
    num_heads = 12
    
    residual_dropout = 0.1
    attention_dropout = 0.1
    content_dropout = 0.1
    
    init_weights = True
    
    def set_dependents(self):
        if self.residual_channels is None:
            self.residual_channels = 4 * self.channels

def make_nonlinearity(config):
    if config.nonlinearity == 'relu':
        return ReLU()
    elif config.nonlinearity == 'gelu':
        return GELU()

class TransformerBlock(Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        
        self.attention_norm = LayerNorm(config.channels)
        self.attention = MemoryAttention(
            config.channels,
            config.num_heads,
            attention_dropout = config.attention_dropout,
            content_dropout = config.content_dropout,
        )
        
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            make_nonlinearity(config),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x, pad, mask=None, use_memory=None):
        x = x + self.attention(self.attention_norm(x), pad, mask, use_memory)
        x = x + self.projection_residual(x)
        
        return x

class Transformer(Module):
    '''
    This class implements a stack of multiple transformer blocks without the
    token embedding or any output decoders.
    '''
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.blocks = ModuleList(
            [TransformerBlock(config) for _ in range(config.num_blocks)])
        
        if config.init_weights:
            self.apply(init_weights)
    
    def forward(self, x, t, pad, use_memory=None):
        mask = padded_causal_self_mask(t, pad)
        
        for block in self.blocks:
            x = block(x, pad, mask, use_memory)
        
        return x
    
def init_weights(module):
    if isinstance(module, (Linear, Embedding)):
        module.weight.data.normal_(mean=0., std=0.02)
        if isinstance(module, Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.)
