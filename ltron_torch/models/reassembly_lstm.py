import random

import numpy

import torch
from torch.nn import Module, Linear, LSTM
from torch.distributions import Categorical

from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.dataset.paths import get_dataset_info
from ltron.hierarchy import index_hierarchy
from ltron.bricks.brick_type import BrickType
from ltron.gym.envs.reassembly_env import reassembly_template_action

from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, default_tile_transform, default_image_transform)
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)
from ltron_torch.models.sequence_fcn import (
    named_resnet_independent_sequence_fcn)
from ltron_torch.models.heads import (
    LinearMultiheadDecoder, Conv2dMultiheadDecoder)

from ltron_torch.models.resnet import named_backbone, named_encoder_channels
from ltron_torch.models.mlp import Conv2dStack
from ltron_torch.models.simple_fcn import SimpleDecoder

# build functions ==============================================================

def build_model(config):
    print('-'*80)
    print('Building resnet disassembly model')
    global_heads = 9
    dense_heads = 3
    model = LSTMModel()
    
    return model.cuda()


# input and output utilities ===================================================

def observations_to_tensors(train_config, observation, pad):
    
    frames = observation['workspace_color_render']
    s, b, h, w, c = frames.shape
    frames = frames.reshape(s*b, h, w, c)
    frames = [default_image_transform(frame) for frame in frames]
    frames = torch.stack(frames)
    frames = frames.view(s, b, c, h, w)
    
    return frames

class LSTMModel(Module):
    def __init__(self,
        resnet_name='resnet18',
        freeze_resnet=True,
        v_channels=64,
        vwf_channels=1500,
        vhf_channels=1500,
        hidden_channels=512,
        lstm_channels=512,
        decoder_channels=256,
        global_actions=8,
        workspace_shape=(8,8),
        handspace_shape=(3,3),
    ):
        super(LSTMModel, self).__init__()
        fcn_layers = ('layer4', 'layer3', 'layer2', 'layer1')
        self.v_backbone = named_backbone(
            resnet_name, *fcn_layers, pretrained=True)
        resnet_channels = named_encoder_channels(resnet_name)
        self.v_conv = Conv2dStack(
            2, resnet_channels[0], resnet_channels[0], v_channels)
        vw_channels = v_channels * workspace_shape[0] * workspace_shape[1]
        vh_channels = v_channels * handspace_shape[0] * handspace_shape[1]
        self.vw_linear = Linear(vw_channels, vwf_channels)
        self.vh_linear = Linear(vh_channels, vhf_channels)
        
        lstm_in_channels = vwf_channels + vhf_channels
        
        self.lstm = LSTM(
            input_size=lstm_in_channels,
            hidden_size=lstm_channels,
            num_layers=1,
        )
        
        self.dense_linear = Linear(lstm_channels, resnet_channels[0])
        
        self.workspace_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=decoder_channels,
        )
        
        self.handspace_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=decoder_channels,
        )
        
        self.global_decoder = Linear(lstm_channels, global_actions)
    
    def forward(self, xw, xh, hc0=None):
        x_layers = []
        xs = []
        for x in (xw, xh):
            s, b, c, h, w = x.shape
            x_layer = self.v_backbone(x.view(s*b, c, h, w))
            x_layers.append(x_layer)
            x = self.v_conv(x_layer[0])
            sb, c, h, w = x.shape
            xs.append(x.view(s, b, c, h, w))
        
        xw, xh = xs
        xw_layers, xh_layers = x_layers
        
        xw = xw.view(s, b, -1)
        xw = self.vw_linear(xw)
        
        xh = xh.view(s, b, -1)
        xh = self.vh_linear(xh)
        
        xlstm = torch.cat((xw, xh), dim=-1)
        
        x, hcn = self.lstm(xlstm, hc0)
        s, b, c = x.shape
        
        dense_x = self.dense_linear(x).view(s*b, -1, 1, 1)
        
        layer_4w = xw_layers[0] + dense_x
        xw_dense = self.workspace_decoder(layer_4w, *xw_layers[1:])
        
        layer_4h = xh_layers[0] + dense_x
        xh_dense = self.workspace_decoder(layer_4, *xh_layers[1:])
        
        import pdb
        pdb.set_trace()
