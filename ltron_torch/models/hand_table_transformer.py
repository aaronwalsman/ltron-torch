import os

import numpy

import torch
from torch.nn import Module, Embedding, Linear, LayerNorm, Sequential

import tqdm

from splendor.image import save_image

from ltron.config import Config
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.gym.envs.blocks_env import BlocksEnv
from ltron.compression import (
    batch_deduplicate_tiled_seqs, batch_deduplicate_from_masks)
from ltron.visualization.drawing import write_text

from ltron_torch.gym_tensor import default_tile_transform
from ltron_torch.models.embedding import (
    TileEmbedding,
    TokenEmbedding,
)
from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.models.mask import padded_causal_mask
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import TileEmbedding, TokenEmbedding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
    init_weights,
)
from ltron_torch.models.heads import LinearMultiheadDecoder
#from ltron_torch.interface.utils import categorical_or_max, categorical_or_max_2d

class HandTableTransformerConfig(Config):
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
    
    global_tokens = 1
    
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
    
    table_channels = 2
    hand_channels = 2
    num_modes = 20
    num_shapes = 6
    num_colors = 6
    
    model_checkpoint = False
    init_weights = True
    
    def set_dependents(self):
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
            self.table_decoder_pixels + self.hand_decoder_pixels)
        
        assert self.table_decode_h % self.table_tiles_h == 0
        assert self.table_decode_w % self.table_tiles_w == 0
        self.upsample_h = self.table_decode_h // self.table_tiles_h
        self.upsample_w = self.table_decode_w // self.table_tiles_w
        assert self.hand_decode_h // self.hand_tiles_h == self.upsample_h
        assert self.hand_decode_w // self.hand_tiles_w == self.upsample_w
        
        #self.global_tokens = self.num_modes + self.num_shapes + self.num_colors

class HandTableTransformer(Module):
    def __init__(self, config):
        super().__init__()
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
        encoder_config = TransformerConfig.translate(
            config,
            blocks='encoder_blocks',
            channels='encoder_channels',
            residual_channels='encoder_residual_channels',
            num_heads='encoder_heads',
        )
        self.encoder = Transformer(encoder_config)
        
        # build the linear layer that converts from encoder to decoder channels
        self.encode_to_decode = Sequential(
            Linear(config.encoder_channels, config.decoder_channels),
            LayerNorm(config.decoder_channels)
        )
        
        # build the decoder
        self.decoder = CrossAttentionDecoder(config)
        
        # initialize weights
        if config.model_checkpoint:
            self.load_state_dict(torch.load(config.model_checkpoint))
        elif config.init_weights:
            self.apply(init_weights)
    
    def forward(self,
        tile_x, tile_t, tile_yx, tile_pad,
        token_x, token_t, token_pad,
        decode_t, decode_pad,
        use_memory=None,
    ):
        
        # make the tile embeddings
        tile_x = self.tile_embedding(tile_x)
        tile_pt = self.temporal_position_encoding(tile_t)
        tile_pyx = self.spatial_position_encoding(tile_yx)
        tile_x = tile_x + tile_pt + tile_pyx
        
        # make the tokens
        token_x = self.token_embedding(token_x)
        token_pt = self.temporal_position_encoding(token_t)
        token_x = token_x + token_pt
        
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
        super().__init__()
        
        # store config
        self.config = config
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.spatial_tiles+config.global_tokens)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.max_sequence_length)
        
        # build the cross-attention block
        decoder_config = TransformerConfig.translate(
            config,
            blocks='decoder_blocks',
            channels='decoder_channels',
            residual_channels='decoder_residual_channels',
            num_heads='decoder_heads',
        )
        self.block = TransformerBlock(decoder_config)
        
        # output
        self.norm = LayerNorm(config.decoder_channels)
        
        table_upsample_channels = (
            config.table_channels * config.upsample_h * config.upsample_w)
        self.table_decoder = Linear(
            config.decoder_channels, table_upsample_channels)
        
        hand_upsample_channels = (
            config.hand_channels * config.upsample_h * config.upsample_w)
        self.hand_decoder = Linear(
            config.decoder_channels, hand_upsample_channels)
        
        #self.mode_decoder = Linear(
        #    config.decoder_channels, config.global_channels)
        global_head_spec = {
            'mode' : config.num_modes,
            'shape' : config.num_shapes,
            'color' : config.num_colors,
        }
        self.global_decoder = LinearMultiheadDecoder(
            config.decoder_channels,
            global_head_spec,
        )
    
    def forward(self, tq, pad_q, xk, tk, pad_k, use_memory=None):
        
        # use the positional encoding to generate the query tokens
        x_spatial = self.spatial_position_encoding.encoding
        x_temporal = self.temporal_position_encoding(tq)
        s, b, c = x_temporal.shape
        hw, c = x_spatial.shape
        xq = x_temporal.view(s, 1, b, c) + x_spatial.view(1, hw, 1, c)
        xq = xq.view(s*hw, b, c)
        tq = tq.view(s, 1, b).expand(s, hw, b).reshape(s*hw, b)
        pad_q = pad_q * hw
        
        # compute the mask
        mask = padded_causal_mask(tq, pad_q, tk, pad_k)
        
        # use the transformer block to compute the output
        x = self.block(xq, pad_k, xk=xk, mask=mask, use_memory=use_memory)
        
        # reshape the output
        uh = self.config.upsample_h
        uw = self.config.upsample_w
        assert self.config.table_channels == self.config.hand_channels
        uc = self.config.table_channels
        uhwc = uh*uw*uc
        x = x.view(s, hw, b, c)
        
        # split off the table tokens, upsample and reshape into a rectangle
        table_start = 0
        table_end = table_start + self.config.table_tiles
        table_x = self.table_decoder(x[:,table_start:table_end])
        th = self.config.table_tiles_h
        tw = self.config.table_tiles_w
        table_x = table_x.view(s, th, tw, b, uh, uw, uc)
        table_x = table_x.permute(0, 3, 6, 1, 4, 2, 5)
        table_x = table_x.reshape(s, b, uc, th*uh, tw*uw)
        
        # split off the hand tokens, upsample and reshape into a rectangle
        hand_start = table_end
        hand_end = hand_start + self.config.hand_tiles
        hand_x = self.hand_decoder(x[:,hand_start:hand_end])
        hh = self.config.hand_tiles_h
        hw = self.config.hand_tiles_w
        hand_x = hand_x.view(s, hh, hw, b, uh, uw, uc)
        hand_x = hand_x.permute(0, 3, 6, 1, 4, 2, 5)
        hand_x = hand_x.reshape(s, b, uc, hh*uh, hw*uw)
        
        # split off the global tokens
        global_start = hand_end
        global_end = global_start + self.config.global_tokens
        
        assert self.config.global_tokens == 1
        #global_x = x[:,global_start:global_end].permute(0, 2, 1, 3)
        #s, b, _, c = global_x.shape
        #global_x = global_x.view(s,b,c)
        global_x = x[:,global_start:global_end]
        s, _, b, c = global_x.shape
        global_x = global_x.view(s,b,c)
        
        x = self.global_decoder(global_x)
        x['table'] = table_x
        x['hand'] = hand_x
        #mode_x = global_x['mode']
        #shape_x = global_x['shape']
        #color_x = global_x['color']
        
        return x
