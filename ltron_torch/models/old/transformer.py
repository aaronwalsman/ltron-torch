from functools import reduce

import torch
from torch.nn import (
    Module, Sequential, Linear, Dropout, ReLU, GELU, MultiheadAttention,
    Embedding, LayerNorm, TransformerEncoderLayer, Parameter, Identity,
    Conv3d
)
from torch.optim import AdamW

from ltron_torch.config import Config
from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.multihead_attention import (
    SlotAttention,
    SparseMultiheadAttention,
    FixedMaskMultiheadAttention,
    SpatialMultiheadAttention,
)
from ltron_torch.models.positional_encoding import sinusoid_positional_encoding
from ltron_torch.models.multihead_decoder import MultiheadDecoder
import ltron_torch.models.transformer_masks as transformer_masks


# Config =======================================================================

class TransformerConfig(Config):
    v = 4096
    t = 1
    h = None
    w = None
    grid_t = None
    grid_h = None
    grid_w = None
    decoder_tokens = 0
    decode_input = True
    
    mask = 'full'
    
    input_type = 'tokens' # images
    
    attention_module = 'torch'
    nonlinearity = 'gelu'
    
    block_type = 'gpt'
    num_blocks = 12
    channels = 768
    residual_channels = None
    num_heads = 12
    decoder_channels = 1
    
    randomize_decoder_embeddings = False
    
    learned_positional_encoding = True
    
    embedding_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1
    
    init_weights = True
    
    verbose = False
    
    def set_dependents(self):
        if self.residual_channels is None:
            self.residual_channels = 4 * self.channels
        
        if self.grid_t is None:
            self.grid_t = 1
        if self.h is None:
            if self.input_type == 'tokens':
                self.h = 32
            elif self.input_type == 'images':
                self.h = 256
        if self.grid_h is None:
            if self.input_type == 'tokens':
                self.grid_h = 1
            elif self.input_type == 'images':
                self.grid_h = 16
        if self.w is None:
            if self.input_type == 'tokens':
                self.w = 32
            elif self.input_type == 'images':
                self.w = 256
        if self.grid_w is None:
            if self.input_type == 'tokens':
                self.grid_w = 1
            elif self.input_type == 'images':
                self.grid_w = 16
        
        self.tt = self.t // self.grid_t
        self.hh = self.h // self.grid_h
        self.ww = self.w // self.grid_w


# Components ===================================================================

def print_verbose(verbose, statement):
    if verbose:
        print(statement)

def make_attention_mask(config):
    if config.mask == 'full':
        return None
    elif config.mask == 'blowfish':
        return transformer_masks.blowfish(
            config.decoder_tokens,
            config.h * config.w,
            config.t,
        )

def make_attention_module(config):
    mask = make_attention_mask(config)
    if config.attention_module == 'sparse':
        sparsity_pattern = torch.nonzero(mask, as_tuple=True)
        return SparseMultiheadAttention(
            sparsity_pattern,
            config.channels,
            config.num_heads,
            config.attention_dropout,
            config.residual_dropout,
        )
    
    elif config.attention_module == 'mingpt':
        return MinGPTMultiheadAttention(something)
    
    elif config.attention_module == 'torch':
        return FixedMaskMultiheadAttention(
            mask, config.channels, config.num_heads, config.attention_dropout)
    
    elif config.attention_module == 'spatial':
        return SpatialMultiheadAttention(
            config.h*config.w,
            config.channels,
            config.num_heads,
            config.attention_dropout,
        )
    
    elif config.attention_module == 'none':
        return Identity()

def make_nonlinearity(config):
    if config.nonlinearity == 'relu':
        return ReLU()
    elif config.nonlinearity == 'gelu':
        return GELU()


# Embedding ====================================================================

def make_embedding_block(config):
    return EmbeddingBlock(config)

class GridEmbedding(Module):
    def __init__(self, config):
        super(GridEmbedding, self).__init__()
        self.t = config.t
        self.h = config.h
        self.w = config.w
        self.grid_t = config.grid_t
        self.grid_h = config.grid_h
        self.grid_w = config.grid_w
        self.tt = config.tt
        self.hh = config.hh
        self.ww = config.ww
        self.channels = config.channels
        
        grid_cells = self.grid_t * self.grid_h * self.grid_w
        assert (config.channels % grid_cells) == 0
        
        self.embedding = Embedding(
            config.v, config.channels//grid_cells)
        
    def forward(self, x):
        x = self.embedding(x)
        *_, b, c = x.shape
        x = x.view(self.t, self.h, self.w, b, c)
        if self.grid_t > 1:
            t_parts = [x[i::self.grid_t] for i in range(self.grid_t)]
            x = torch.cat(t_parts, dim=-1)
        if self.grid_h > 1:
            h_parts = [x[:,i::self.grid_h] for i in range(self.grid_h)]
            x = torch.cat(h_parts, dim=-1)
        if self.grid_w > 1:
            w_parts = [x[:,:,i::self.grid_w] for i in range(self.grid_w)]
            x = torch.cat(w_parts, dim=-1)
        
        x = x.view(self.tt * self.hh * self.ww, b, self.channels)
        
        return x

class ImageBlockEmbedding(Module):
    def __init__(self, config):
        super(ImageBlockEmbedding, self).__init__()
        k = (config.grid_t, config.grid_h, config.grid_w)
        self.conv = Conv3d(3, config.channels, kernel_size=k, stride=k)
    
    def forward(self, x):
        x = self.conv(x)
        b, c, t, h, w = x.shape
        x = x.view(b, c, t, h*w)
        x = x.permute(2, 3, 0, 1).contiguous()
        return x

class EmbeddingBlock(Module):
    def __init__(self, config):
        super(EmbeddingBlock, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        self.randomize_decoder_embeddings = config.randomize_decoder_embeddings
        
        if config.input_type == 'tokens':
            self.embedding = GridEmbedding(config)
        elif config.input_type == 'images':
            self.embedding = ImageBlockEmbedding(config)
        else:
            raise NotImplementedError
        
        if self.randomize_decoder_embeddings:
            num_decoder_embeddings = 1
        else:
            num_decoder_embeddings = self.decoder_tokens
        self.decoder_embedding = Embedding(
            num_decoder_embeddings, config.channels)
        
        print_verbose(config.verbose, 'making positional encoding')
        
        num_data_tokens = config.tt * config.hh * config.ww
        if config.learned_positional_encoding:
            p = torch.zeros(num_data_tokens, 1, config.channels)
            self.positional_encoding = NoWeightDecayParameter(p)
        else:
            p = positional_encoding(config.channels, num_data_tokens, 1)
            self.register_buffer('positional_encoding', p)
        
        print_verbose(config.verbose, 'finished making positional encoding')
        
        self.dropout = Dropout(config.embedding_dropout)
    
    def forward(self, d):
        d = self.embedding(d)
        *thw, b, c = d.shape
        thw = reduce(lambda a,b:a*b, thw, 1)
        d = d.view(thw, b, c)
        d = d + self.positional_encoding
        
        if self.randomize_decoder_embeddings:
            raise NotImplementedError
        else:
            s = self.decoder_embedding.weight.view(self.decoder_tokens, 1, c)
            s = s.expand(self.decoder_tokens, b, c)
        
        x = torch.cat((s, d), dim=0)
        
        '''
        if self.randomize_decoder_embeddings:
            #s = self.decoder_embedding.weight.view(1, 1, c)
            #s = s.expand(self.decoder_tokens, b, c)
            #s = s + torch.randn(self.decoder_tokens, b, c, device=x.device)
            s = torch.randn(self.decoder_tokens, b, c, device=x.device)
        else:
            #s = self.decoder_embedding.weight.view(self.decoder_tokens, 1, c)
            s = self.decoder_embedding.weight.view(1,1,c)
            s = s.expand(self.decoder_tokens, b, c)
        x = torch.cat((s, d), dim=0)
        '''
        
        x = self.dropout(x)
        
        return x


# Blocks =======================================================================

def make_block(config):
    if config.block_type == 'gpt':
        return Block(config)
    elif config.block_type == 'time_only':
        return TimeOnlyBlock(config)
    elif config.block_type == 'time_then_space':
        return TimeThenSpaceBlock(config)
    elif config.block_type == 'slot':
        return SlotBlock(config)
    elif config.block_type == 'pblock':
        return PBlock(config)
    elif config.block_type == 'multi_pblock':
        return MultiPBlock(config)
    else:
        raise NotImplementedError

class Block(Module):
    def __init__(self, config):
        super(Block, self).__init__()
        
        print_verbose(config.verbose, 'building block with:')
        print_verbose(
            config.verbose, '  channels: %i'%config.channels)
        print_verbose(
            config.verbose, '  residual_channels: %i'%config.residual_channels)
        
        self.attention_residual = Sequential(
            LayerNorm(config.channels),
            make_attention_module(config),
        )
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            make_nonlinearity(config),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x):
        x = x + self.attention_residual(x)
        x = x + self.projection_residual(x)
        
        return x

class TimeOnlyBlock(Block):
    def __init__(self, config):
        super(TimeOnlyBlock, self).__init__(config)
        self.spatial_tokens = config.hh * config.ww
    
    def forward(self, x):
        thw, b, c = x.shape
        x = x.view(-1, self.spatial_tokens*b, c)
        x = super(TimeOnlyBlock, self).forward(x)
        x = x.view(thw, b, c)
        
        return x

class TimeThenSpaceBlock(Block):
    def __init__(self, config):
        super(TimeThenSpaceBlock, self).__init__(config)
        self.spatial_tokens = config.hh * config.ww
        
        self.spatial_attention_residual = Sequential(
            LayerNorm(config.channels),
            make_attention_module(config),
        )
    
    def forward(self, x):
        thw, b, c = x.shape
        hw = self.spatial_tokens
        t = thw//hw
        x = x.view(t, hw*b, c)
        x = x + self.attention_residual(x)
        
        x = x.view(t, hw, b, c)
        x = x.permute(1, 0, 2, 3).contiguous().view(hw, t*b, c)
        x = x + self.spatial_attention_residual(x)
        
        x = x.view(hw, t, b, c).permute(1, 0, 2, 3).contiguous().view(thw, b, c)
        x = x + self.projection_residual(x)
        
        return x

class SlotBlock(Module):
    def __init__(self, config):
        super(SlotBlock, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        
        self.layernorm = LayerNorm(config.channels)
        self.attention = SlotAttention(
            config.channels,
            config.num_heads,
            config.attention_dropout,
            config.residual_dropout,
        )
        
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            make_nonlinearity(config),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x):
        s = x[:self.decoder_tokens]
        d = x[self.decoder_tokens:]
        
        rs = self.layernorm(s)
        rd = self.layernorm(d)
        r = self.attention(rs, rd)
        s = s + r
        
        s = s + self.projection_residual(s)
        
        x = torch.cat((s,d), dim=0)
        
        return x

class PBlock(Module):
    def __init__(self, config):
        super(PBlock, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        
        self.cross_layernorm = LayerNorm(config.channels)
        self.cross_attention = MultiheadAttention(
            config.channels, config.num_heads, config.attention_dropout)
        
        self.self_attention_residual = Sequential(
            LayerNorm(config.channels),
            FixedMaskMultiheadAttention(
                None,
                config.channels,
                config.num_heads,
                config.attention_dropout
            ),
        )
        
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            make_nonlinearity(config),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x):
        s = x[:self.decoder_tokens]
        d = x[self.decoder_tokens:]
        
        rs = self.cross_layernorm(s)
        #rd = self.cross_layernorm(d)
        rd = d
        r = self.cross_attention(rs, rd, rd)[0]
        s = s + r
        
        s = s + self.self_attention_residual(s)
        
        s = s + self.projection_residual(s)
        
        x = torch.cat((s,d), dim=0)
        return x

class MultiPBlock(Module):
    def __init__(self, config):
        super(MultiPBlock, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        
        self.pblock = PBlock(config)
        block_config = TransformerConfig(
            t = self.decoder_tokens,
            h = 1,
            w = 1,
            channels = config.channels,
            residual_channels = config.residual_channels,
            num_heads = config.num_heads,
        )
        self.blocks = Sequential(*[Block(block_config) for _ in range(3)])
    
    def forward(self, x):
        x = self.pblock(x)
        s = x[:self.decoder_tokens]
        d = x[self.decoder_tokens:]
        s = self.blocks(s)
        x = torch.cat((s,d), dim=0)
        
        return x

# TODO: Gated block from Stabilizing Transformers For Reinforcement Learning


# Read head ====================================================================

class ReadHead(Module):
    def __init__(self, config):
        super(ReadHead, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        self.decode_input = config.decode_input
        
        if isinstance(config.decoder_channels, int):
            decoder = Linear(config.channels, config.decoder_channels)
        else:
            decoder = MultiheadDecoder(config.channels, config.decoder_channels)
        
        self.head = torch.nn.Sequential(
            #LayerNorm(config.channels),
            decoder,
        )
    
    def forward(self, x):
        if not self.decode_input:
            x = x[:self.decoder_tokens]
        
        x = self.head(x)
        
        return x


# Transformer ==================================================================

class Transformer(Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        
        #self.embedding = EmbeddingBlock(config)
        self.embedding = make_embedding_block(config)
        self.blocks = Sequential(
            *[make_block(config) for _ in range(config.num_blocks)])
        self.read_head = ReadHead(config)
        
        if config.init_weights:
            self.apply(self._init_weights)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.read_head(x)
        
        return x
    
    def _init_weights(self, module):
        # from minGPT
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0., std=0.02)
            if isinstance(module, Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.)
