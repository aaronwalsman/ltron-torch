from functools import reduce

import torch
from torch.nn import (
    Module, ModuleList, Sequential, Linear, Dropout, ReLU, GELU,
    MultiheadAttention, Embedding, LayerNorm, TransformerEncoderLayer,
    Parameter, Identity, Conv3d
)
from torch.optim import AdamW

from ltron_torch.config import Config
from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.compressed_causal_attention import (
    CompressedCausalAttention,
    #make_compressed_causal_mask,
)
from ltron_torch.models.positional_encoding import (
    FactoredLearnedRelativePositionalEncoding,
    FactoredPositionalEncoding,
    PositionalEncoding,
)
import ltron_torch.models.transformer_masks as transformer_masks


# Config =======================================================================

class CompressedTransformerConfig(Config):
    #t = 128
    #h = 16
    #w = 16
    data_shape = (128, 16, 16)
    tile_h = 16
    tile_w = 16
    decoder_tokens = 0
    decode_input = False
    causal_dim = 0
    
    input_mode='tile'
    input_tile_channels = 3
    input_token_vocab = 4096
    
    nonlinearity = 'gelu'
    
    num_blocks = 12
    channels = 768
    residual_channels = None
    num_heads = 12
    decoder_channels = 1
    
    factored_positional_encoding = True
    relative_positional_encoding = False
    learned_positional_encoding = True
    
    embedding_dropout = 0.1
    residual_dropout = 0.1
    attention_dropout = 0.1
    content_dropout = 0.1
    
    init_weights = True
    
    def set_dependent_variables(self):
        if self.residual_channels is None:
            self.residual_channels = 4 * self.channels


# Components ===================================================================

def make_nonlinearity(config):
    if config.nonlinearity == 'relu':
        return ReLU()
    elif config.nonlinearity == 'gelu':
        return GELU()


# Embedding ====================================================================

class TokenEmbeddingBlock(Module):
    def __init__(self, config):
        super(TokenEmbeddingBlock, self).__init__()
        self.embedding = Embedding(config.input_token_vocab, config.channels)
        self.dropout = Dropout(config.embedding_dropout)
    
    def forward(self, x, pe=0):
        x = self.embedding(x)
        x = self.dropout(x+pe)
        
        return x


class TileEmbeddingBlock(Module):
    def __init__(self, config):
        super(TileEmbeddingBlock, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        
        self.tile_linear = Linear(
            config.tile_h*config.tile_w*config.input_tile_channels,
            config.channels,
        )
        
        self.dropout = Dropout(config.embedding_dropout)
    
    def forward(self, x, pe=0):
        n, b, h, w, c = x.shape
        x = x.view(n, b, h*w*c)
        x = self.tile_linear(x)
        x = self.dropout(x+pe)
        
        return x


class DecoderEmbeddingBlock(Module):
    def __init__(self, config):
        super(DecoderEmbeddingBlock, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        self.channels = config.channels
        self.decoder_embedding = Embedding(self.decoder_tokens, self.channels)
    
    def forward(self, x, i, t, pad_lengths):
        s, b, dims = i.shape
        
        dx = self.decoder_embedding.weight
        dx = dx.unsqueeze(1).expand(self.decoder_tokens, b, self.channels)
        di = torch.ones(
            (self.decoder_tokens, b, dims),
            dtype=torch.long,
            device=i.device,
        ) * -1
        di[:,:,0] = 1
        di[:,range(b),1] = t
        
        x = torch.cat((dx, x), dim=0)
        i = torch.cat((di, i), dim=0)
        
        return x, i, pad_lengths + self.decoder_tokens


# Blocks =======================================================================

class Block(Module):
    def __init__(self, config):
        super(Block, self).__init__()
        
        self.attention_norm = LayerNorm(config.channels)
        if config.relative_positional_encoding:
            self.attention = CompressedRelativeCausalAttention(
                config.channels,
                config.num_heads,
                attention_dropout = config.attention_dropout,
                content_dropout = config.content_dropout,
            )
        else:
            self.attention = CompressedCausalAttention(
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
    
    def forward(self, x, pe, causal_mask, pad_lengths, terminal=None):
        if isinstance(pe, torch.Tensor):
            pe = self.attention_norm(pe)
        x = x + self.attention(
            self.attention_norm(x), pe, causal_mask, pad_lengths, terminal)
        x = x + self.projection_residual(x)
        
        return x


# Read head ====================================================================

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


# Transformer ==================================================================

class CompressedTransformer(Module):
    def __init__(self, config):
        super(CompressedTransformer, self).__init__()
        
        if config.relative_positional_encoding:
            raise Exception('No longer supported')
            self.positional_encoding =  (
                FactoredLearnedRelativePositionalEncoding(
                    config.channels,
                    config.data_shape,
                    (0,),
                )
            )
        else:
            if config.factored_positional_encoding:
                self.positional_encoding = FactoredPositionalEncoding(
                    channels=config.channels,
                    data_shape=config.data_shape,
                    causal_dim=config.causal_dim,
                    learned=config.learned_positional_encoding,
                )
            else:
                raise Exception('No longer supported (need to implement -1)')
                self.positional_encoding = PositionalEncoding(
                    channels=config.channels,
                    data_shape=config.data_shape,
                    causal_dim=0,
                    learned=config.learned_positional_encoding,
                )
        
        if config.input_mode == 'tile':
            self.embedding_block = TileEmbeddingBlock(config)
        elif config.input_mode == 'token':
            self.embedding_block = TokenEmbeddingBlock(config)
        
        self.decoder_tokens = config.decoder_tokens
        if self.decoder_tokens:
            self.decoder_block = DecoderEmbeddingBlock(config)
        
        self.blocks = ModuleList(
            [Block(config) for _ in range(config.num_blocks)])
        self.read_head = ReadHead(config)
        
        if config.init_weights:
            self.apply(self._init_weights)
    
    def forward(self, x, i, pad_lengths, t=None, terminal=None):
        #causal_mask = make_compressed_causal_mask(i)
        x = self.embedding_block(x, 0)
        if self.decoder_tokens:
            x, i, pad_lengths = self.decoder_block(x, i, t, pad_lengths)
        
        pe, causal_mask = self.positional_encoding(i, pad_lengths)
        
        for block in self.blocks:
            x = block(x, pe, causal_mask, pad_lengths, terminal)
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
