import numpy

import torch
from torch.nn import Module, Embedding, Linear, LayerNorm, Sequential

from ltron.compression import batch_deduplicate_tiled_seqs

from ltron_torch.config import Config
from ltron_torch.gym_tensor import default_tile_transform
from ltron_torch.models.embedding import (
    TileEmbedding,
    TokenEmbedding,
)
from ltron_torch.models.padding import cat_padded_seqs
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import TileEmbedding, TokenEmbedding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
)

class BreakAndMakeTransformerConfig:
    tile_h = 16
    tile_w = 16
    tile_c = 3
    table_h = 256
    table_w = 256
    hand_h = 96
    hand_w = 96
    
    table_decode_h = 64
    table_decode_w = 64
    hand_decode_h = 24
    hand_decode_w = 24
    
    max_sequence_length = 1024
    
    embedding_dropout = 0.1
    
    nonlinearity = 'gelu'
    
    encoder_blocks = 12
    encoder_channels = 768
    encoder_residual_channels = None
    encoder_heads = 12
    
    decoder_blocks = 4
    decoder_channels = 768
    decoder_residual_channels = None
    decoder_heads = 12
    
    residual_dropout = 0.1
    attention_dropout = 0.1
    content_dropout = 0.1
    
    def set_dependents(self):
        self.encoder_config = TransformerConfig.translate(
            self,
            blocks='encoder_blocks',
            channels='encoder_channels',
            residual_channels='encoder_residual_channels',
            heads='encoder_heads',
        )
        
        self.decoder_config = TransformerConfig.translate(
            self,
            blocks='decoder_blocks',
            channels='decoder_channels',
            residual_channels='decoder_residual_channels',
            heads='decoder_heads',
        )
        
        assert self.table_h % self.tile_h == 0
        assert self.table_w % self.table_w == 0
        self.table_tiles_h = self.table_h // self.tile_h
        self.table_tiles_w = self.table_w // self.tile_w
        self.table_tiles = self.table_tiles_h * self.table_tiles_w
        
        assert self.hand_h % self.tile_h == 0
        assert self.hand_w % self.tile_w == 0
        self.hand_tiles_h = self.hand_h // self.tile_h
        self.hand_tiles_w = self.hand_w // self.tile_w
        self.hand_tiles = self.hand_tiles_h * self.hand_tiles_w
        
        self.spatial_tiles = self.table_tiles + self.hand_tiles
        
        self.table_decoder_pixels = self.table_decode_h * self.table_decode_w
        self.hand_decoder_pixels = self.hand_decode_h * self.hand_decode_w
        self.decoder_pixels = (
            self.table_decode_pixels * self.hand_decoder_pixels)

class BreakAndMakeTransformer(Module):
    def __init__(self, config):
        super(BreakAndMakeTransformer, self).__init__()
        self.config = config
        
        # build the tokenizers
        self.tile_embedding = TileEmbedding(
            config.tile_h,
            config.tile_w,
            config.tile_c,
            config.encoder_channels,
            config.embedding_dropout,
        )
        self.token_embedding = TokenEmbedding(
            2, config.encoder_channels, config.embedding_dropout)
        
        self.mask_embedding = Embedding(1, config.encoder_channels)
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.encoder_channels, config.spatial_tiles)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.encoder_channels, config.max_sequence_length)
        
        # build the transformer
        self.encoder = Transformer(config.encoder_config)
        
        # build the linear layer that converts from encoder to decoder channels
        self.encode_to_decode = Sequential(
            Linear(config.encoder_channels, config.decoder_channels),
            LayerNorm(config.decoder_channels)
        )
        
        # build the decoder
        self.decoder = Transformer(config.decoder_config)
    
    def forward(self,
        tile_x, tile_t, tile_yx, tile_pad,
        token_x, token_t, token_pad,
        decode_t, decode_pad,
        use_memory=None,
    ):
        
        # make the tile embeddings
        tile_x = self.tile_embedding(tile_x)
        tile_ps = self.temporal_position_encoding(tile_t)
        tile_pyx = self.spatial_position_encoding(tile_yx)
        tile_x = tile_x + tile_ps + tile_pyx
        
        # make the tokens
        token_x = self.token_embedding(token_x)
        token_ps = self.temporal_position_encoding(token_t)
        token_x = token_x + token_ps
        
        # concatenate the tile and discrete tokens
        x, pad = cat_padded_seqs(tile_x, token_x, tile_pad, token_pad)
        t, _ = cat_padded_seqs(tile_t, token_t, tile_pad, token_pad)
        
        # use the encoder to encode
        x = self.encoder(x, t, pad, use_memory=use_memory)
        
        # convert encoder channels to decoder channels
        x = self.encode_to_decode(x)
        
        # use the decoder to decode
        x = self.decoder(decode_t, decode_pad, x, t, pad, use_memory=use_memory)
        
        return x

class CrossAttentionDecoder(Module):
    def __init__(self, config):
        super(CrossAttentionDecoder, self).__init__()
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.spatial_tiles)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.max_sequence_length)
        
        # build the attention layer
        self.block = TransformerBlock(config.decoder_config)
    
    def forward(self, tq, pad_q, xk, tk, pad_k):
        
        # use the positional encoding to generate the query tokens
        x_temporal = self.spatial_positional_encoding(tq)
        x_spatial = self.spatial_positional_encoding.encoding
        s, b, c = x_temporal.shape
        hw, c = x_spatial.shape
        xq = x_temporal.view(s, 1, b, c) + x_spatial.view(1, hw, 1, c)
        xq = xq.view(s*hw, b, c)

class BreakAndMakeTransformerInterface:
    def __init__(self, model, config):
        srelf.config = config
        self.model = model
    
    def observation_to_tensors(self, observation, pad):
        device = next(self.model.parameters()).device
        
        # process table tiles
        table_tiles, table_tyx, table_pad = batch_deduplicate_tiled_seqs(
            observation['workspace_color_render'],
            pad,
            self.config.tile_h,
            self.config.tile_w,
            background=102,
        )
        table_pad = torch.LongTensor(table_pad).to(device)
        table_tiles = default_tile_transform(table_tiles).to(device)
        table_t = torch.LongTensor(table_tyx[...,0]).to(device)
        table_yx = torch.LongTensor(
            table_tyx[...,1] *
            self.config.table_tiles_w +
            table_tyx[...,2],
        ).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_tiled_seqs(
            observation['handspace_color_render'],
            pad,
            self.config.tile_h,
            self.config.tile_w,
            background=102,
        )
        hand_pad = torch.LongTensor(hand_pad).to(device)
        hand_tiles = default_tile_transform(hand_tiles).to(device)
        hand_t = torch.LongTensor(hand_tyx[...,0]).to(device)
        hand_yx = torch.LongTensor(
            hand_tyx[...,1] *
            self.config.hand_tiles_w +
            hand_tyx[...,2] +
            self.config.table_tiles,
        ).to(device)
        
        # cat table and hand ties
        tile_x, tile_pad = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        tile_t, _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
        tile_yx, _ = cat_padded_seqs(table_yx, hand_yx, table_pad, hand_pad)
        
        # process tokens
        token_x = torch.LongTensor(observation['phase_switch']).to(device)
        s = numpy.max(pad)
        b, = pad.shape
        token_t = torch.arange(s).view(s, 1).expand(s, b).contiguous().to(
            device)
        token_pad = torch.LongTensor(pad).to(device)
        
        # step
        decode_t = torch.LongTensor(observation['step']).to(device)
        
        return (
            tile_x, tile_t, tile_yx, tile_pad,
            token_x, token_t, token_pad,
            decode_t,
        )
    
    def loss(self, x, pad, y, log=None, clock=None):
        pass
    
    def tensor_to_actions(self, x, mode='sample'):
        pass
    
    def visualize_episodes(self, epoch, episodes, visualization_directory):
        pass
    
    def eval_episodes(self, episodes, log, clock):
        pass
