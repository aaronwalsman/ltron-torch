import random

import numpy

import torch
from torch.nn import Module, Linear, Embedding, LSTM, Dropout
from torch.distributions import Categorical

from ltron.dataset.paths import get_dataset_info

from ltron_torch.gym_tensor import default_image_transform
from ltron_torch.models.heads import (
    LinearMultiheadDecoder, Conv2dMultiheadDecoder)

from ltron_torch.models.resnet import named_backbone, named_encoder_channels
from ltron_torch.models.mlp import conv2d_stack
from ltron_torch.models.simple_fcn import SimpleDecoder

# build functions ==============================================================

def build_model(config, device):
    print('-'*80)
    print('Building lstm reassembly model')
    # THIS SHOULD ALL BE WRAPPED INTO THE CLASS BELOW
    # (14 camera + 2 reset + 1 dis + 2 pnp + 3 rotate + 1 insert)
    mode_actions = 23
    dataset_info = get_dataset_info(config.dataset)
    num_classes = max(dataset_info['class_ids'].values())+1
    num_colors = max(dataset_info['color_ids'].values())+1
    global_heads = LinearMultiheadDecoder(
        512, {'mode':mode_actions, 'class':num_classes, 'color':num_colors})
    dense_workspace_heads = Conv2dMultiheadDecoder(256, 2, kernel_size=1)
    dense_handspace_heads = Conv2dMultiheadDecoder(256, 2, kernel_size=1)
    model = BreakAndMakeLSTM(
        global_heads=global_heads,
        dense_workspace_heads=dense_workspace_heads,
        dense_handspace_heads=dense_handspace_heads,
        resnet_name=config.resnet_name,
    )
    
    return model.to(device)


class BreakAndMakeLSTM(Module):
    def __init__(self,
        global_heads,
        dense_workspace_heads,
        dense_handspace_heads,
        resnet_name='resnet50',
        freeze_resnet=False,
        compact_visual_channels=64,
        global_workspace_channels=768,
        global_handspace_channels=768,
        reassembly_channels=512,
        lstm_hidden_channels=512,
        decoder_channels=256,
        visual_dropout=0.,
        global_dropout=0.,
        workspace_shape=(8,8),
        handspace_shape=(3,3),
    ):
        
        # intiialization and storage
        super(BreakAndMakeLSTM, self).__init__()
        self.lstm_hidden_channels = lstm_hidden_channels
        self.global_heads = global_heads
        self.dense_workspace_heads = dense_workspace_heads
        self.dense_handspace_heads = dense_handspace_heads
        
        # resnet backbone
        fcn_layers = ('layer4', 'layer3', 'layer2', 'layer1')
        self.visual_backbone = named_backbone(
            resnet_name, *fcn_layers, frozen_batchnorm=True, pretrained=True)
        
        # visual feature extractor
        resnet_channels = named_encoder_channels(resnet_name)
        self.visual_stack = conv2d_stack(
            num_layers=2,
            in_channels=resnet_channels[0],
            hidden_channels=resnet_channels[0]//2,
            out_channels=compact_visual_channels,
        )
        
        # visual dropout
        self.visual_dropout = Dropout(visual_dropout)
        
        # global feature layers
        work_pixels = workspace_shape[0] * workspace_shape[1]
        work_channels = compact_visual_channels * work_pixels
        self.workspace_feature = Linear(
            work_channels, global_workspace_channels)
        hand_pixels = handspace_shape[0] * handspace_shape[1]
        hand_channels = compact_visual_channels * hand_pixels
        self.handspace_feature = Linear(
            hand_channels, global_handspace_channels)
        
        # reassembly embedding
        self.reassembly_embedding = Embedding(2, reassembly_channels)
        
        # global dropout
        self.global_dropout = Dropout(global_dropout)
        
        # build the lstm
        lstm_in_channels = (
            global_workspace_channels +
            global_handspace_channels +
            reassembly_channels
        )
        self.lstm = LSTM(
            input_size=lstm_in_channels,
            hidden_size=lstm_hidden_channels,
            num_layers=1,
        )
        
        # map decoders
        self.dense_linear = Linear(lstm_hidden_channels, resnet_channels[0])
        
        self.workspace_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=decoder_channels,
        )
        
        self.handspace_decoder = SimpleDecoder(
            encoder_channels=resnet_channels,
            decoder_channels=decoder_channels,
        )
    
    def initialize_memory(self, batch_size):
        device = next(self.parameters()).device
        hidden_state = torch.zeros(
            1, batch_size, self.lstm_hidden_channels, device=device)
        cell_state = torch.zeros(
            1, batch_size, self.lstm_hidden_channels, device=device)
        return hidden_state, cell_state
    
    def reset_memory(self, memory, terminal):
        hidden, cell = memory
        for i, t in enumerate(terminal):
            if t:
                hidden[:,i] = 0
                cell[:,i] = 0
    
    def forward(self, x_work, x_hand, r, memory=None):
        
        # compute visual features
        x_layers = []
        xs = []
        for x in (x_work, x_hand):
            s, b, c, h, w = x.shape
            x_layer = self.visual_backbone(x.view(s*b, c, h, w))
            x_layers.append(x_layer)
            x = self.visual_stack(x_layer[0])
            sb, c, h, w = x.shape
            xs.append(x.view(s, b, c, h, w))
        
        x_work, x_hand = xs
        x_work_layers, x_hand_layers = x_layers
        
        # compute global feature
        x_work = x_work.view(s, b, -1)
        x_work = self.workspace_feature(x_work)
        
        x_hand = x_hand.view(s, b, -1)
        x_hand = self.handspace_feature(x_hand)
        
        xr = self.reassembly_embedding(r)
        
        x_global = torch.cat((x_work, x_hand, xr), dim=-1)
        
        # compute sequence features
        x, hcn = self.lstm(x_global, memory)
        s, b, c = x.shape
        
        # compute global action features
        x_out = self.global_heads(x)
        x_out['memory'] = hcn
        
        # compute maps
        dense_x = self.dense_linear(x).view(s*b, -1, 1, 1)
        
        layer_4w = x_work_layers[0] + dense_x
        xw_dense = self.workspace_decoder(layer_4w, *x_work_layers[1:])
        xw_dense = self.dense_workspace_heads(xw_dense)
        sh, c, h, w = xw_dense.shape
        xw_dense = xw_dense.view(s, b, c, h, w)
        x_out['workspace'] = xw_dense
        
        layer_4h = x_hand_layers[0] + dense_x
        xh_dense = self.handspace_decoder(layer_4h, *x_hand_layers[1:])
        xh_dense = self.dense_handspace_heads(xh_dense)
        sh, c, h, w = xh_dense.shape
        xh_dense = xh_dense.view(s, b, c, h, w)
        x_out['handspace'] = xh_dense
        
        return x_out
