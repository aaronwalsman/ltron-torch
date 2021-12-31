from functools import reduce

import torch
from torch.nn import (
    Module, ModuleList, Sequential, Linear, Dropout, ReLU, GELU,
    MultiheadAttention, Embedding, LayerNorm, TransformerEncoderLayer,
    Parameter, Identity
)
from torch.optim import AdamW

from ltron.config import Config

from ltron_torch.models.padding import cat_padded_seqs
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
from ltron_torch.models.heads import LinearMultiheadDecoder


# Config =======================================================================

class CompressedTransformerConfig(Config):
    data_shape = (128, 16, 16)
    causal_dim = 0
    include_tile_embedding=True,
    include_token_embedding=True,
    tile_h = 16
    tile_w = 16
    tile_c = 3
    token_vocab = 4096
    decoder_tokens = 0
    decode_input = False
    
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
    
    def set_dependents(self):
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
        self.embedding = Embedding(config.token_vocab, config.channels)
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
            config.tile_h*config.tile_w*config.tile_c,
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
    
    def forward(self, x, xi, x_pad, ti, t_pad):
        s, b, dims = xi.shape
        ts = ti.shape[0]
        
        d = self.decoder_embedding.weight
        d = d.view(1, self.decoder_tokens, 1, self.channels)
        d = d.expand(ts, self.decoder_tokens, b, self.channels)
        d = d.reshape(ts*self.decoder_tokens, b, self.channels)
        #d = d.unsqueeze(1).expand(self.decoder_tokens, b, self.channels)
        '''
        di = torch.ones(
            (self.decoder_tokens, b, dims),
            dtype=torch.long,
            device=xi.device,
        ) * -1
        di[:,:,0] = 1
        di[:,range(b),1] = ti
        '''
        # ti (ts x b)
        di = torch.ones(
            (self.decoder_tokens * ts, b, dims),
            dtype=torch.long,
            device=xi.device,
        ) * -1
        di[:,:,0] = 1
        di[:,:,1] = ti.view(
            ts, 1, b).expand(
            ts, self.decoder_tokens, b).reshape(
            ts * self.decoder_tokens, b)
        
        #x = torch.cat((d, x), dim=0)
        #xi = torch.cat((di, xi), dim=0)
        
        d_pad = t_pad*self.decoder_tokens
        dx, dx_pad = cat_padded_seqs(d, x, d_pad, x_pad)
        dxi, _ = cat_padded_seqs(di, xi, d_pad, x_pad)
        
        #return x, xi, pad + self.decoder_tokens
        return dx, dxi, dx_pad, d_pad


# Transformer Blocks ===========================================================

class Block(Module):
    def __init__(self, config):
        super(Block, self).__init__()
        
        self.attention_norm = LayerNorm(config.channels)
        
        self.attention = CompressedCausalAttention(
            config.channels,
            config.num_heads,
            attention_dropout = config.attention_dropout,
            content_dropout = config.content_dropout,
        )
        '''
        self.attention = MultiheadAttention(
            config.channels,
            config.num_heads,
            dropout=config.attention_dropout,
        )
        '''
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            make_nonlinearity(config),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x, pe, causal_mask, pad, terminal=None):
        if isinstance(pe, torch.Tensor):
            pe = self.attention_norm(pe)
        ####
        x = x + self.attention(
            self.attention_norm(x), pe, causal_mask, pad, terminal)
        ####
        #xn = self.attention_norm(x)
        #a, _ = self.attention(xn, xn, xn)
        #x = x + a
        ####
        x = x + self.projection_residual(x)
        
        return x


# Read head ====================================================================

class ReadHead(Module):
    def __init__(self, config):
        super(ReadHead, self).__init__()
        self.decoder_tokens = config.decoder_tokens
        self.decode_input = config.decode_input
        
        #if isinstance(config.decoder_channels, int):
        #    decoder = Linear(config.channels, config.decoder_channels)
        #else:
        decoder = LinearMultiheadDecoder(
            config.channels, config.decoder_channels)
        
        self.head = torch.nn.Sequential(
            #LayerNorm(config.channels),
            decoder
        )
    
    def forward(self, x, pad, d_pad):
        if not self.decode_input:
            #x = x[:self.decoder_tokens]
            max_pad = torch.max(d_pad)
            x = x[:max_pad]
            pad = d_pad
        
        x = self.head(x)
        
        return x, pad


# Transformer ==================================================================

class CompressedTransformer(Module):
    def __init__(self, config):
        super(CompressedTransformer, self).__init__()
        
        self.include_tile_embedding = config.include_tile_embedding
        self.include_token_embedding = config.include_token_embedding
        
        # positional encoding
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
        
        # embedding
        #if config.input_mode == 'tile':
        #    self.embedding_block = TileEmbeddingBlock(config)
        #elif config.input_mode == 'token':
        #    self.embedding_block = TokenEmbeddingBlock(config)
        if config.include_tile_embedding:
            self.tile_embedding_block = TileEmbeddingBlock(config)
        if config.include_token_embedding:
            self.token_embedding_block = TokenEmbeddingBlock(config)
        
        # decoder tokens
        self.decoder_tokens = config.decoder_tokens
        if self.decoder_tokens:
            self.decoder_block = DecoderEmbeddingBlock(config)
        
        # blocks
        self.blocks = ModuleList(
            [Block(config) for _ in range(config.num_blocks)])
        self.read_head = ReadHead(config)
        
        # initialize weights
        if config.init_weights:
            self.apply(self._init_weights)
    
    def forward(self,
        tile_x=None, tile_i=None, tile_pad=None,
        token_x=None, token_i=None, token_pad=None,
        decoder_i=None, decoder_pad=None,
        terminal=None,
    ):
        
        if not self.include_tile_embedding:
            assert tile_x is None
            assert tile_i is None
            assert tile_pad is None
        
        if not self.include_token_embedding:
            assert tile_x is None
            assert tile_i is None
            assert tile_pad is None
        
        # embed tiles
        if tile_x is not None:
            tile_x = self.tile_embedding_block(tile_x, 0)
        
        # embed tokens
        if token_x is not None:
            token_x = self.token_embedding_block(token_x, 0)
        
        # combine tokens
        if tile_x is not None and token_x is not None:
            x, pad = cat_padded_seqs(tile_x, token_x, tile_pad, token_pad)
            i, _ = cat_padded_seqs(tile_i, token_i, tile_pad, token_pad)
        elif tile_x is not None:
            x = tile_x
            i = tile_i
            pad = tile_pad
        elif token_x is not None:
            x = token_x
            i = token_i
            pad = poken_pad
        else:
            assert False, 'No input'
        
        # make decoder tokens
        if self.decoder_tokens:
            x, i, pad, decoder_pad = self.decoder_block(
                x, i, pad, decoder_i, decoder_pad)
        else:
            decoder_pad = None
        
        # make positional encoding
        pe, causal_mask = self.positional_encoding(i, pad)
        
        # run through each transformer block
        for block in self.blocks:
            x = block(x, pe, causal_mask, pad, terminal)
        
        # read off the values
        x, pad = self.read_head(x, pad, decoder_pad)
        
        return x, pad
    
    def _init_weights(self, module):
        # from minGPT
        if isinstance(module, (Linear, Embedding)):
            module.weight.data.normal_(mean=0., std=0.02)
            if isinstance(module, Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.)
