import random

import numpy

import torch
from torch.nn import Module, Linear, Embedding, LSTM
from torch.distributions import Categorical

from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.dataset.paths import get_dataset_info
from ltron.hierarchy import index_hierarchy
from ltron.bricks.brick_type import BrickType
from ltron.gym.envs.reassembly_env import reassembly_template_action
from ltron.dataset.paths import get_dataset_info

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
    # (14 camera + 2 reset + 1 dis + 1 pnp + 3 rotate + 1 insert)
    global_actions = 23
    dataset_info = get_dataset_info(config.dataset)
    num_classes = max(dataset_info['class_ids'].values())+1
    num_colors = max(dataset_info['color_ids'].values())+1
    global_heads = LinearMultiheadDecoder(
        512, (global_actions, num_classes, num_colors))
    dense_workspace_heads = Conv2dMultiheadDecoder(256, 3, kernel_size=1)
    dense_handspace_heads = Conv2dMultiheadDecoder(256, 3, kernel_size=1)
    model = LSTMModel(
        global_heads=global_heads,
        dense_workspace_heads=dense_workspace_heads,
        dense_handspace_heads=dense_handspace_heads,
    )
    
    return model.cuda()


# input and output utilities ===================================================

def observations_to_tensors(train_config, observation, pad):

    output = []
    for component in 'workspace_color_render', 'handspace_color_render':
        frames = observation[component]
        s, b, h, w, c = frames.shape
        frames = frames.reshape(s*b, h, w, c)
        frames = [default_image_transform(frame) for frame in frames]
        frames = torch.stack(frames)
        frames = frames.view(s, b, c, h, w)
        output.append(frames)
    
    reassembling = torch.LongTensor(observation['reassembly']['reassembling'])
    output.append(reassembling)
    
    return tuple(output)

class LSTMModel(Module):
    def __init__(self,
        global_heads,
        dense_workspace_heads,
        dense_handspace_heads,
        resnet_name='resnet18',
        freeze_resnet=False,
        v_channels=64,
        vwf_channels=768,
        vhf_channels=768,
        reassembly_channels=512,
        hidden_channels=512,
        lstm_channels=512,
        decoder_channels=256,
        workspace_shape=(8,8),
        handspace_shape=(3,3),
    ):
        super(LSTMModel, self).__init__()
        self.global_heads = global_heads
        self.dense_workspace_heads = dense_workspace_heads
        self.dense_handspace_heads = dense_handspace_heads
        
        fcn_layers = ('layer4', 'layer3', 'layer2', 'layer1')
        self.v_backbone = named_backbone(
            resnet_name, *fcn_layers, frozen=freeze_resnet, pretrained=True)
        resnet_channels = named_encoder_channels(resnet_name)
        self.v_conv = Conv2dStack(
            2, resnet_channels[0], resnet_channels[0], v_channels)
        vw_channels = v_channels * workspace_shape[0] * workspace_shape[1]
        vh_channels = v_channels * handspace_shape[0] * handspace_shape[1]
        self.vw_linear = Linear(vw_channels, vwf_channels)
        self.vh_linear = Linear(vh_channels, vhf_channels)
        
        lstm_in_channels = vwf_channels + vhf_channels + reassembly_channels
        
        self.lstm = LSTM(
            input_size=lstm_in_channels,
            hidden_size=lstm_channels,
            num_layers=1,
        )
        
        self.reassembly_linear = Embedding(2, reassembly_channels)
        
        self.dense_linear = Linear(lstm_channels, resnet_channels[0])
        
        self.workspace_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=decoder_channels,
        )
        
        self.handspace_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=decoder_channels,
        )
        
    def forward(self, xw, xh, r, hc0=None):
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
        
        xr = self.reassembly_linear(r)
        
        xlstm = torch.cat((xw, xh, xr), dim=-1)
        
        # compute dense features
        x, hcn = self.lstm(xlstm, hc0)
        s, b, c = x.shape
        
        # compute global action features
        x_global = self.global_heads(x)
        
        dense_x = self.dense_linear(x).view(s*b, -1, 1, 1)
        
        layer_4w = xw_layers[0] + dense_x
        xw_dense = self.workspace_decoder(layer_4w, *xw_layers[1:])
        xw_dense = self.dense_workspace_heads(xw_dense)
        sh, c, h, w = xw_dense.shape
        xw_dense = xw_dense.view(s, b, c, h, w)
        
        layer_4h = xh_layers[0] + dense_x
        xh_dense = self.handspace_decoder(layer_4h, *xh_layers[1:])
        xh_dense = self.dense_handspace_heads(xh_dense)
        sh, c, h, w = xh_dense.shape
        xh_dense = xh_dense.view(s, b, c, h, w)
        
        return x_global, xw_dense, xh_dense, hcn
