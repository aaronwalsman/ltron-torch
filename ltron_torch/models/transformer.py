import torch
from torch.nn import (
    Module, Sequential, Linear, Dropout, ReLU, GELU, MultiheadAttention,
    Embedding, LayerNorm, TransformerEncoderLayer, Parameter, Identity
)
from torch.optim import AdamW

from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.multihead_attention import (
    SlotAttention,
    SparseMultiheadAttention,
    FixedMaskMultiheadAttention,
    SpatialMultiheadAttention,
)
from ltron_torch.models.positional_encoding import positional_encoding
import ltron_torch.models.transformer_masks as transformer_masks

class TrainConfig:
    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

class TransformerConfig:
    vocabulary = 4096
    sequence_length = 1
    map_height = 32
    map_width = 32
    decoder_tokens = 0
    decode_input = True
    
    mask = 'full'
    
    attention_module = 'torch'
    nonlinearity = 'gelu'
    
    block_type = 'gpt'
    num_blocks = 12
    channels = 768
    residual_channels = None
    num_heads = 12
    decoder_channels = 1
    
    learned_positional_encoding = False
    # so far, for most purposes, multi is worse than single
    positional_encoding_dimensions = 'single' # single/multi
    randomize_decoder_embeddings = False
    
    embedding_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1
    
    init_weights = True
    
    verbose = False
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key), 'Invalid Key: %s'%key
            setattr(self, key, value)
        
        if self.residual_channels is None:
            self.residual_channels = 4 * self.channels

def print_verbose(verbose, statement):
    if verbose:
        print(statement)

def make_block(config):
    if config.block_type == 'gpt':
        return Block(config)
    elif config.block_type == 'slot':
        return SlotBlock(config)
    elif config.block_type == 'time_only':
        return TimeOnlyBlock(config)
    elif config.block_type == 'time_then_space':
        return TimeThenSpaceBlock(config)
    else:
        raise NotImplementedError

def make_attention_mask(config):
    if config.mask == 'full':
        '''
        tokens = (
            config.map_height *
            config.map_width *
            config.sequence_length +
            config.decoder_tokens
        )
        return transformer_masks.full(tokens)
        '''
        return None
    elif config.mask == 'blowfish':
        return transformer_masks.blowfish(
            config.decoder_tokens,
            config.map_height * config.map_width,
            config.sequence_length,
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
            config.map_height*config.map_width,
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

class EmbeddingBlock(Module):
    def __init__(self, config):
        super(EmbeddingBlock, self).__init__()
        
        self.embedding = Embedding(config.vocabulary, config.channels)
        
        self.decoder_tokens = config.decoder_tokens
        self.randomize_decoder_embeddings = config.randomize_decoder_embeddings
        if self.randomize_decoder_embeddings:
            num_decoder_embeddings = 1
        else:
            num_decoder_embeddings = self.decoder_tokens
        
        self.decoder_embedding = Embedding(
            num_decoder_embeddings, config.channels)
        
        # TODO: make the positional encoding happen conditionally when/if we
        # try the Transformer-XL thing.
        
        self.learned_positional_encoding = config.learned_positional_encoding
        print_verbose(config.verbose, 'making positional encoding')
        
        if self.learned_positional_encoding:
            if config.positional_encoding_dimensions == 'multi':
                p = torch.zeros(
                    config.sequence_length,
                    config.map_height * config.map_width,
                    1,
                    config.channels)
            elif config.positional_encoding_dimensions == 'single':
                num_tokens = (
                    config.sequence_length *
                    config.map_height *
                    config.map_width
                )
                p = torch.zeros(num_tokens, 1, config.channels)
                    
            else:
                raise NotImplementedError
            
            self.positional_encoding = NoWeightDecayParameter(p)
        else:
            if config.positional_encoding_dimensions == 'multi':
                p = positional_encoding(
                    config.channels,
                    config.sequence_length,
                    config.map_height * config.map_width,
                ).unsqueeze(2)
            elif config.positional_encoding_dimensions == 'single':
                num_tokens = (
                    config.sequence_length *
                    config.map_height *
                    config.map_width
                )
                p = positional_encoding(
                    config.channels, num_tokens).unsqueeze(1)
            else:
                raise NotImplementedError
            
            self.register_buffer('positional_encoding', p)
        
        print_verbose(config.verbose, 'finished making positional encoding')
        
        self.dropout = Dropout(config.embedding_dropout)
    
    def forward(self, d):
        d = self.embedding(d)
        t, hw, b, c = d.shape
        d = d + self.positional_encoding.view(t, hw, 1, c)
        d = d.view(t*hw, b, c)
        
        if self.randomize_decoder_embeddings:
            #s = self.decoder_embedding.weight.view(1, 1, c)
            #s = s.expand(self.decoder_tokens, b, c)
            #s = s + torch.randn(self.decoder_tokens, b, c, device=x.device)
            s = torch.randn(self.decoder_tokens, b, c, device=x.device)
        else:
            s = self.decoder_embedding.weight.view(self.decoder_tokens, 1, c)
            s = s.expand(self.decoder_tokens, b, c)
        
        x = torch.cat((s, d), dim=0)
        #t, b, c = x.shape
        #x = x + self.p2[:t]
        
        x = self.dropout(x)
        
        return x

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
        self.sequence_length = config.sequence_length
    
    def forward(self, x):
        thw, b, c = x.shape
        x = x.view(self.sequence_length, -1, c)
        x = super(TimeOnlyBlock, self).forward(x)
        x = x.view(thw, b, c)
        
        return x

class TimeThenSpaceBlock(Block):
    def __init__(self, config):
        super(TimeThenSpaceBlock, self).__init__(config)
        self.sequence_length = config.sequence_length
        
        self.spatial_attention_residual = Sequential(
            LayerNorm(config.channels),
            make_attention_module(config),
        )
    
    def forward(self, x):
        thw, b, c = x.shape
        t = self.sequence_length
        hw = thw//self.sequence_length
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


# TODO: Gated block from Stabilizing Transformers For Reinforcement Learning

class ReadHead(Module):
    def __init__(self, config):
        super(ReadHead, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        self.decode_input = config.decode_input
        
        self.head = torch.nn.Sequential(
            #LayerNorm(config.channels),
            Linear(config.channels, config.decoder_channels),
        )
    
    def forward(self, x):
        if not self.decode_input:
            x = x[:self.decoder_tokens]
        
        x = self.head(x)
        
        return x

class TokenMapSequenceEncoder(Module):
    def __init__(self, config):
        super(TokenMapSequenceEncoder, self).__init__()
        
        print_verbose(config.verbose, 'building transformer embedding block')
        self.embedding = EmbeddingBlock(config)
        
        print_verbose(config.verbose, 'building transformer main blocks')
        self.blocks = Sequential(
            *[make_block(config) for _ in range(config.num_blocks)])
        
        print_verbose(config.verbose, 'building transformer read head')
        self.read_head = ReadHead(config)
        
        print_verbose(config.verbose, 'initializing weights')
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
