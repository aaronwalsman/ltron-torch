import torch
from torch.nn import Module, Linear, LayerNorm, Sequential

from ltron.config import Config

from ltron_torch.models.mask import padded_causal_mask
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.hand_table_embedding import (
    HandTableEmbeddingConfig, HandTableEmbedding)
from ltron_torch.models.resnet import named_backbone

class StubnetDecoderConfig(Config):
    max_sequence_length = 1024
    
    encoder_channels = 768
    stub_channels = 64
    decode_spatial_locations = 64**2 + 24**2
    
    cursor_channels = 2
    num_modes = 20
    num_shapes = 6
    num_colors = 6
    
    pretrained = True

class StubnetDecoder(Module):
    def __init__(self, config):
        super().__init__()
        
        # store config
        self.config = config
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels,
            config.decode_spatial_locations,
        )
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.max_sequence_length)
        
        # build the resnet stub
        self.stubnet = named_backbone(
            'resnet18', 'layer1', pretrained=config.pretrained)
        
        # build the linear layer that converts from encoder to decoder channels
        self.cursor_linear = Linear(
            config.encoder_channels, config.stub_channels*2)
        self.cursor_layer_norm = LayerNorm(config.stub_channels)
        
        global_head_spec = {
            'mode' : config.num_modes,
            'shape' : config.num_shapes,
            'color' : config.num_colors,
        }
        self.global_decoder = LinearMultiheadDecoder(
            config.encoder_channels,
            global_head_spec,
        )
    
    def forward(
        self,
        decode_x,
        decode_pad,
        table_x,
        hand_x,
    ):
        
        # run the stubnet on the input images
        s, b, c, h, w = table_x.shape
        table_x, = self.stubnet(table_x.view(s*b,c,h,w))
        _, c, h, w = table_x.shape
        table_x = table_x.view(s, b, c, h, w)
        
        s, b, c, h, w = hand_x.shape
        hand_x, = self.stubnet(hand_x.view(s*b,c,h,w))
        _, c, h, w = hand_x.shape
        hand_x = hand_x.view(s, b, c, h, w)

        # add the positional encoding
        # ...
        
        cursor_x = self.cursor_linear(decode_x)
        s, b, cc = cursor_x.shape
        c = cc//2
        cursor_x = cursor_x.view(s, b, 2, c)
        cursor_x = self.cursor_layer_norm(cursor_x)
        
        table_x = torch.einsum('sbchw,sbnc->sbnhw', table_x, cursor_x)
        table_x = table_x / (c**0.5)
        hand_x = torch.einsum('sbchw,sbnc->sbnhw', hand_x, cursor_x)
        hand_x = hand_x / (c**0.5)
        
        x = self.global_decoder(decode_x)
        x['table'] = table_x
        x['hand'] = hand_x
        
        return x
