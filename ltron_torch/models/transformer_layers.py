import torch
from torch.nn import (
    Module, Sequential, Linear, Dropout, ReLU, GELU, MultiheadAttention)
import torch.nn.functional as F

from ltron_torch.models.multihead_attention import SparseMultiheadAttention
import ltron_torch.models.transformer_masks as transformer_masks
from ltron_torch.models.positional_encoding import positional_encoding

class TransformerConfig:
    vocabulary = 4096
    sequence_length = 1
    map_height = 32
    map_width = 32
    decoder_tokens = 0
    decode_input = True
    
    mask = 'full'
    
    block = 'gpt'
    attention_module = 'sparse'
    nonlinearity = 'gelu'
    
    channels = 768
    num_layers = 12
    num_heads = 12
    output_channels = 1
    
    embedding_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1
    
    def __init__(self, **kwargs):
        for key, value in kwargs:
            assert hasattr(self, key)
            setattr(self, key, value)

def make_attention_mask(config):
    if config.mask == 'full':
        tokens = (
            config.map_height *
            config.map_width *
            sequence_length +
            config.decoder_tokens
        )
        return transformer_masks.full(tokens)

def make_attention_module(config):
    if config.attention_module == 'sparse':
        return SparseMultiheadAttention(something)
    
    elif config.attention_module == 'mingpt':
        return MinGPTMultiheadAttention(something)
    
    elif config.attention_module == 'torch':
        mask = make_attention_mask(config)
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

class EmbeddingBlock(Module):
    def __init__(self, config):
        super(GPTEmbeddingBlock, self).__init__()
        
        self.embedding = Embedding(config.vocabulary, config.channels)
        # TODO: make the positional encoding happen conditionally when/if we
        # try the Transformer-XL thing.
        self.positional_encoding = positional_encoding(
            config.channels,
            (config.sequence_length, config.map_height * config.map_width),
        )
        self.dropout = Dropout(config.embedding_dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.positional_encoding
        x = self.dropout(x)
        
        return x

class TransformerBlock(Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.attention_residual = Sequential(
            make_attention_module(config),
            Dropout(config),
        )
        self.attention_norm = LayerNorm(config.channels)
        
        self.projection_residual = Sequential(
            Linear(config.channels, 4*config.channels),
            make_nonlinearity(config),
            Dropout(config.residual_dropout),
            Linear(4*config.channels, config.channels),
            Dropout(config.residual_dropout),
        )
        self.residual_norm = LayerNorm(config.channels)
    
    def forward(self, x):
        x = x + self.attention_residual(x)
        x = self.attention_norm(x)
        x = x + self.projection_residual(x)
        x = self.residual_norm(x)

class GPTBlock(Module):
    def __init__(self, config):
        self.attention_residual = Sequential(
            LayerNorm(config.channels),
            make_attention_module(config),
        )
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, 4*config.channels),
            make_nonlinearity(config),
            Linear(4*config.channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x):
        x = x + attention_residual(x)
        x = x + projection_residual(x)

# TODO: Gated block from Stabilizing Transformers For Reinforcement Learning

class ReadHead(Module):
    def __init__(self, config):
        super(ReadHead, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        self.decode_input = config.decode_input
        
        self.head = torch.nn.Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.output_channels),
        )
    
    def forward(self, x):
        if not self.decode_input:
            x = x[:self.decoder_tokens]
        
        x = self.head(x)
        
        return x

class TokenMapSequenceEncoder(Module):
    def __init__(self, config):
        
        self.embedding = EmbeddingBlock(config)
        self.blocks = Sequential(
            *[make_block(config) for _ in range(config.num_layers)])
        
        self.read_head = ReadHand(config.decoder_tokens, config.read_from)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.layernorm(x)
