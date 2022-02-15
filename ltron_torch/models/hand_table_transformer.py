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
from ltron_torch.models.hand_table_embedding import (
    HandTableEmbeddingConfig, HandTableEmbedding)
#from ltron_torch.interface.utils import categorical_or_max, categorical_or_max_2d

class HandTableTransformerConfig(HandTableEmbeddingConfig):
    table_decode_tokens_h = 16
    table_decode_tokens_w = 16
    table_decode_h = 64
    table_decode_w = 64
    hand_decode_tokens_h = 6
    hand_decode_tokens_w = 6
    hand_decode_h = 24
    hand_decode_w = 24
    
    global_tokens = 1
    
    max_sequence_length = 1024
    
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
    
    cursor_channels = 2
    num_modes = 20
    num_shapes = 6
    num_colors = 6
    
    init_weights = True
    
    def set_dependents(self):
        assert self.table_decode_h % self.table_decode_tokens_h == 0, (
            '%i, %i'%(self.table_decode_tokens_h, self.table_decode_h))
        assert self.table_decode_w % self.table_decode_tokens_w == 0
        self.upsample_h = self.table_decode_h // self.table_decode_tokens_h
        self.upsample_w = self.table_decode_w // self.table_decode_tokens_w
        self.table_decode_tokens = (
            self.table_decode_tokens_h * self.table_decode_tokens_w)
        
        assert self.hand_decode_h % self.hand_decode_tokens_h == 0
        assert self.hand_decode_w % self.hand_decode_tokens_w == 0
        assert self.hand_decode_h//self.hand_decode_tokens_h == self.upsample_h
        assert self.hand_decode_w//self.hand_decode_tokens_w == self.upsample_w
        self.hand_decode_tokens = (
            self.hand_decode_tokens_h * self.hand_decode_tokens_w)
        
        self.decode_tokens = self.table_decode_tokens + self.hand_decode_tokens
        
        self.table_decoder_pixels = self.table_decode_h * self.table_decode_w
        self.hand_decoder_pixels = self.hand_decode_h * self.hand_decode_w
        self.decoder_pixels = (
            self.table_decoder_pixels + self.hand_decoder_pixels)

class HandTableTransformer(Module):
    def __init__(self, config, checkpoint=None):
        super().__init__()
        self.config = config
        
        # build the token embedding
        self.embedding = HandTableEmbedding(config)
        
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
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            if 'token_embedding.embedding.weight' in checkpoint:
                checkpoint['phase_embedding.embedding.weight'] = checkpoint[
                    'token_embedding.embedding.weight']
                del(checkpoint['token_embedding.embedding.weight'])
            if 'mask_embedding.weight' in checkpoint:
                del(checkpoint['mask_embedding.weight'])
            self.load_state_dict(checkpoint)
        elif config.init_weights:
            self.apply(init_weights)
    
    def forward(self,
        #tile_x, tile_t, tile_yx, tile_pad,
        table_tiles, table_t, table_yx, table_pad,
        hand_tiles, hand_t, hand_yx, hand_pad,
        phase_x,
        table_cursor_yx,
        table_cursor_p,
        hand_cursor_yx,
        hand_cursor_p,
        token_t, token_pad,
        decode_t, decode_pad,
        use_memory=None,
    ):
        x, t, pad = self.embedding(
            table_tiles, table_t, table_yx, table_pad,
            hand_tiles, hand_t, hand_yx, hand_pad,
            phase_x,
            table_cursor_yx,
            table_cursor_p,
            hand_cursor_yx,
            hand_cursor_p,
            token_t, token_pad,
        )
        
        # use the encoder to encode
        x = self.encoder(x, t, pad, use_memory=use_memory)
        
        # DIFFERENT
        #print('new x')
        #print(torch.sum(x).cpu())
        
        # convert encoder channels to decoder channels
        x = self.encode_to_decode(x)
        
        #print('new enc to dec x')
        #print(torch.sum(x).cpu())
        
        # use the decoder to decode
        x = self.decoder(decode_t, decode_pad, x, t, pad, use_memory=use_memory)
        
        #print('new mode/shape/color/table/hand x')
        #print('   ', torch.sum(x['mode']).cpu())
        #print('   ', torch.sum(x['shape']).cpu())
        #print('   ', torch.sum(x['color']).cpu())
        #print('   ', torch.sum(x['table']).cpu())
        #print('   ', torch.sum(x['hand']).cpu())
        
        return x

class CrossAttentionDecoder(Module):
    def __init__(self, config):
        super().__init__()
        
        # store config
        self.config = config
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels,
            config.decode_tokens + config.global_tokens,
        )
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
        
        upsample_channels = (
            config.cursor_channels * config.upsample_h * config.upsample_w)
        self.table_decoder = Linear(
            config.decoder_channels, upsample_channels)
        self.hand_decoder = Linear(
            config.decoder_channels, upsample_channels)
        
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
        uc = self.config.cursor_channels
        uhwc = uh*uw*uc
        x = x.view(s, hw, b, c)
        
        # split off the table tokens, upsample and reshape into a rectangle
        table_start = 0
        table_end = table_start + self.config.table_decode_tokens
        table_x = self.table_decoder(x[:,table_start:table_end])
        th = self.config.table_decode_tokens_h
        tw = self.config.table_decode_tokens_w
        table_x = table_x.view(s, th, tw, b, uh, uw, uc)
        table_x = table_x.permute(0, 3, 6, 1, 4, 2, 5)
        table_x = table_x.reshape(s, b, uc, th*uh, tw*uw)
        
        # split off the hand tokens, upsample and reshape into a rectangle
        hand_start = table_end
        hand_end = hand_start + self.config.hand_decode_tokens
        hand_x = self.hand_decoder(x[:,hand_start:hand_end])
        hh = self.config.hand_decode_tokens_h
        hw = self.config.hand_decode_tokens_w
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
