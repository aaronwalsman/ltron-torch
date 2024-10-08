import torch
from torch.nn import (
    Module, ModuleList, Sequential, Linear, Dropout, ReLU, GELU,
    Embedding, LayerNorm, init,
)

from ltron.config import Config

from ltron_torch.models.mlp import make_nonlinearity
from ltron_torch.models.attention import Attention, CausalAttention
from ltron_torch.models.mask import padded_causal_mask

class TransformerConfig(Config):
    nonlinearity = 'gelu'
    blocks = 12
    channels = 768
    residual_channels = None
    heads = 12
    
    residual_dropout = 0.1
    attention_dropout = 0.1
    content_dropout = 0.1
    
    init_weights = True
    
    def set_dependents(self):
        if self.residual_channels is None:
            self.residual_channels = 4 * self.channels

class TransformerBlock(Module):
    def __init__(self, config, AttentionModule):
        super(TransformerBlock, self).__init__()
        
        self.attention_norm = LayerNorm(config.channels)
        self.attention = AttentionModule(
            config.channels,
            config.heads,
            attention_dropout = config.attention_dropout,
            content_dropout = config.content_dropout,
        )
        
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            make_nonlinearity(config.nonlinearity),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x, xk=None, xv=None, *args, **kwargs):
        xq = self.attention_norm(x)
        if xk is not None:
            xk = self.attention_norm(xk)
        if xv is not None:
            xv = self.attention_norm(xv)
        x = x + self.attention(xq, xk=xk, xv=xv, *args, **kwargs)
        x = x + self.projection_residual(x)
        
        return x
    
    #def forward(self,
    #    x, pad, xk=None, mask=None, use_memory=None
    #):
    #    xq = self.attention_norm(x)
    #    if xk is not None:
    #        xk = self.attention_norm(xk)
    #    x = x + self.attention(xq, pad, xk=xk, mask=mask, use_memory=use_memory)
    #    x = x + self.projection_residual(x)
    #    
    #    return x

class Transformer(Module):
    '''
    This class implements a stack of multiple transformer blocks without the
    token embedding or any output decoders.
    '''
    
    AttentionModule = Attention
    
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        self.blocks = ModuleList([
            TransformerBlock(config, self.AttentionModule)
            for _ in range(config.blocks)
        ])
        
        #if config.init_weights:
        #    self.apply(init_weights)
    
    def forward(self, x, output_layers=None, **kwargs):
        if output_layers is None:
            output_layers = set()
        
        #x_norm = torch.norm(x, dim=-1)
        #print('Transformer input min norm: %.04f'%(x_norm.min()))
        #print('Transformer input mean norm: %.04f'%(x_norm.mean()))
        #print('Transformer input max norm: %.04f'%(x_norm.max()))
        
        x_out = {}
        
        for i, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            if i in output_layers:
                x_out[i] = x
            
            #x_norm = torch.norm(x, dim=-1)
            #print('Block %i min norm: %.04f'%(i, x_norm.min()))
            #print('Block %i mean norm: %.04f'%(i, x_norm.mean()))
            #print('Block %i max norm: %.04f'%(i, x_norm.max()))
        
        x_out[-1] = x
        
        return x_out
    
    #def forward(self, x, t, pad, use_memory=None, output_layers=None):
    #    mask = padded_causal_mask(t, pad)
    #    
    #    if output_layers is None:
    #        output_layers = set()
    #    
    #    xs = {}
    #    xs['mask'] = mask
    #    
    #    for i, block in enumerate(self.blocks):
    #        x = block(x, pad, mask=mask, use_memory=use_memory)
    #        if i in output_layers:
    #            xs[i] = x
    #    
    #    xs[-1] = x
    #    
    #    return xs
    

class CausalTransformer(Transformer):
    
    AttentionModule = CausalAttention
    
    def forward(self, x, t, pad, output_layers=None, **kwargs):
        mask = padded_causal_mask(t, pad)
        x_out = super().forward(
            x, output_layers=output_layers, pad=pad, **kwargs)
        x_out['mask'] = mask
        
        return x_out
    
    def zero_all_memory(self):
        for block in self.blocks:
            block.attention.zero_all_memory()
    
def init_weights(module):
    if isinstance(module, Linear):
        module.weight.data.normal_(mean=0., std=0.02)
        #init.kaiming_uniform_(module.weight.data, nonlinearity='relu')
        #if isinstance(module, Linear) and module.bias is not None:
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, Embedding):
        module.weight.data.normal_(mean=0., std=1.)
    #    module.weight.data.zero_()
    elif isinstance(module, LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.)
