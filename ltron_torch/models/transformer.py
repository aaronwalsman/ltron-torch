import torch
from torch.nn import (
    Module, Sequential, Linear, Dropout, ReLU, GELU, MultiheadAttention,
    Embedding, LayerNorm, TransformerEncoderLayer, Parameter
)
import torch.nn.functional as F

from ltron_torch.models.multihead_attention import (
    SparseMultiheadAttention, FixedMaskMultiheadAttention,
)
from ltron_torch.models.positional_encoding import positional_encoding
import ltron_torch.models.transformer_masks as transformer_masks

class TransformerConfig:
    vocabulary = 4096
    sequence_length = 1
    map_height = 32
    map_width = 32
    decoder_tokens = 0
    decode_input = True
    
    mask = 'full'
    
    block = 'gpt'
    attention_module = 'torch'
    nonlinearity = 'gelu'
    
    num_layers = 12
    channels = 768
    residual_channels = None
    num_heads = 12
    decoder_channels = 1
    
    learned_positional_encoding = False
    # so far, for most purposes, multi is worse than single
    positional_encoding_dimensions = 'single' # single/multi
    
    embedding_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1
    
    init_weights = False
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            assert hasattr(self, key)
            setattr(self, key, value)

def make_attention_mask(config):
    if config.mask == 'full':
        tokens = (
            config.map_height *
            config.map_width *
            config.sequence_length +
            config.decoder_tokens
        )
        return transformer_masks.full(tokens)
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

def make_nonlinearity(config):
    if config.nonlinearity == 'relu':
        return ReLU()
    elif config.nonlinearity == 'gelu':
        return GELU()

def make_block(config):
    if config.block == 'transformer':
        return TransformerBlock(config)
    elif config.block == 'gpt':
        return GPTBlock(config)
    elif config.block == 'torch':
        residual_channels = config.residual_channels
        if residual_channels is None:
            residual_channels = 4 * config.channels
        return TransformerEncoderLayer(
            config.channels,
            config.num_heads,
            residual_channels,
            config.residual_dropout,
        )

class EmbeddingBlock(Module):
    def __init__(self, config):
        super(EmbeddingBlock, self).__init__()
        
        self.embedding = Embedding(config.vocabulary, config.channels)
        
        self.decoder_tokens = config.decoder_tokens
        self.decoder_embedding = Embedding(self.decoder_tokens, config.channels)
        
        # TODO: make the positional encoding happen conditionally when/if we
        # try the Transformer-XL thing.
        
        if config.learned_positional_encoding:
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
            
            self.positional_encoding = Parameter(p)
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
        
        self.dropout = Dropout(config.embedding_dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        s, hw, b, c = x.shape
        x = x + self.positional_encoding.view(s, hw, 1, c)
        x = x.view(s*hw, b, c)
        
        d = self.decoder_embedding.weight.view(self.decoder_tokens, 1, c)
        #d = d + self.p2[:self.decoder_tokens]
        #d = d.view(self.decoder_tokens, 1, c)
        d = d.expand(self.decoder_tokens, b, c)
        x = torch.cat((d, x), dim=0)
        #s, b, c = x.shape
        #x = x + self.p2[:s]
        
        x = self.dropout(x)
        
        return x

class TransformerBlock(Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        residual_channels = config.residual_channels
        if residual_channels is None:
            residual_channels = config.channels*4
        
        self.attention_residual = Sequential(
            make_attention_module(config),
            Dropout(config.residual_dropout),
        )
        self.attention_norm = LayerNorm(config.channels)
        
        self.projection_residual = Sequential(
            Linear(config.channels, residual_channels),
            make_nonlinearity(config),
            Dropout(config.residual_dropout),
            Linear(residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
        self.residual_norm = LayerNorm(config.channels)
    
    def forward(self, x):
        x = x + self.attention_residual(x)
        x = self.attention_norm(x)
        x = x + self.projection_residual(x)
        x = self.residual_norm(x)
        
        return x

class GPTBlock(Module):
    def __init__(self, config):
        super(GPTBlock, self).__init__()
        residual_channels = config.residual_channels
        if residual_channels is None:
            residual_channels = config.channels*4
        
        self.attention_residual = Sequential(
            LayerNorm(config.channels),
            make_attention_module(config),
        )
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, residual_channels),
            make_nonlinearity(config),
            Linear(residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x):
        x = x + self.attention_residual(x)
        x = x + self.projection_residual(x)
        
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
        
        self.embedding = EmbeddingBlock(config)
        self.blocks = Sequential(
            *[make_block(config) for _ in range(config.num_layers)])
        
        self.read_head = ReadHead(config)
        
        # this is so far bad for me, but all my other parameters are not dialed
        # in yet either, so...
        if config.init_weights:
            self.apply(self._init_weights)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.read_head(x)
        
        return x
    
    def _init_weights(self, module):
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0., std=0.02)
            if isinstance(module, Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.)
