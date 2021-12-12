from torch.nn import Module

from ltron.compression import batch_deduplicate_tiled_seq

from ltron_torch.config import Config, CompositeConfig
from ltron_torch.model.tokenizer import (
    TileTokenizerConfig,
    TileTokenizer,
    DiscreteTokenizerConfig,
    DiscreteTokenizer,
)

from ltron_torch.padding import cat_padded_seqs
from ltron_torch.positional_encoding import LearnedPositionalEncoding
from ltron_torch.transformer import TransformerConfig, Transformer

class BreakAndMakeTransformerConfig(TransformerConfig):
    tile_h = 16
    tile_w = 16
    table_h = 256
    table_w = 256
    hand_h = 96
    hand_w = 96
    
    max_sequence_length = 1024
    
    def set_dependents(self):
        assert self.table_h % self.tile_h == 0
        assert self.table_w % self.table_w == 0
        self.table_tiles_h = self.table_h // self.tile_h
        self.table_tiles_w = self.table_w // self.tile_w
        
        assert self.hand_h % self.tile_h == 0
        assert self.hand_w % self.tile_w == 0
        self.hand_tiles_h = self.hand_h // self.tile_h
        self.hand_tiles_w = self.hand_w // self.tile_w
        
        self.spatial_tiles = (
            self.table_tiles_h * self.table_tiles_w + 
            self.hand_tiles_h * self.hand_tiles_w
        )

class BreakAndMakeTransformer(Module):
    def __init__(self, config):
        super(BreakAndMakeTokenizer, self).__init__()
        
        # build the tokenizers
        self.tile_tokenizer = TileTokenizer(
            config.tile_h,
            config.tile_w,
            config.tile_c,
            config.channels,
            config.embedding_dropout,
        )
        self.discrete_tokenizer = DiscreteTokenizer(
            config.vocabulary,
            config.channels,
            config.embedding_dropout,
        )
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.channels, config.spatial_tiles)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.channels, config.max_sequence_length)
        
        # build the transformer
        self.transformer = Transformer(config)
        
        # build the decoder
        # who knows?
    
    def forward(self,
        tile_x, tile_i, tile_hw, tile_pad,
        discrete_x, discrete_i, discrete_pad,
    ):
        # make the tile tokens
        tile_x = self.tile_tokenizer(tile_x)
        tile_pi = self.temporal_position_encoding(tile_i)
        tile_phw = self.spatial_position_encoding(tile_hw)
        tile_x = tile_x + tile_pi + tile_phw
        
        # make the discrete tokens
        discrete_x = self.discrete_tokenizer(discrete_x)
        discrete_pi = self.temporal_position_encoding(discrete_i)
        discrete_x = discrete_x + discrete_pi
        
        # concatenate the tile and discrete tokens
        x, pad = cat_padded_seqs(tile_x, discrete_x)
        
        # use the transformer to encode
        x = self.transformer(x)
        
        # use the decoder to decode
        # who knows?
        
        return x

class BreakAndMakeTransformerInterface:
    def __init__(self, config):
        self.config = config
    
    def observations_to_tensors(self, observation, pad, device):
        table_tiles, table_shw, table_pad = batch_deduplicate_tiled_seqs(
            observation['workspace_color_render'],
            pad,
            self.config.tile_h,
            self.config.tile_w,
            background=102,
        )
        table_i = table_shw[...,0]
        table_hw = table_shw[...,1] * config.table_tiles_w + table_shw[...,2]
        
        hand_tiles, hand_shw, hand_pad = batch_deduplicate_tiled_seqs(
            observation['handspace_color_render'],
            pad,
            self.config.tile_h,
            self.config.tile_w,
            background=102,
        )
        hand_i = hand_shw[...,0]
        hand_hw = hand_shw[...,1] * config.hand_tiles_w + hand_shw[...,2]
        hand_hw += config.table_tiles_h * config.table_tiles_w
        
        tile_x, tile_pad = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        tile_s, _ = cat_padded_seqs(table_s, hand_s, table_pad, hand_pad)
        tile_hw, _ = cat_padded_seqs(table_hw, hand_hw, table_pad, hand_pad)
        
        return (
            tile_x, tile_s, tile_hw, tile_pad,
            discrete_x, discrete_i, discrete_pad,
        )
    
    def loss(self, x, pad, y, log=None, clock=None):
        pass
    
    def tensor_to_actions(self, x, mode='sample'):
        pass
    
    def visualize_episodes(self, epoch, episodes, visualization_directory):
        pass
    
    def eval_episodes(self, episodes, log, clock):
        pass
