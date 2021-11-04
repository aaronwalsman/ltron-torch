import random

import numpy

import torch
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


# build functions ==============================================================

def build_model(config):
    print('-'*80)
    print('Building resnet disassembly model')
    global_heads = 9
    dense_heads = 3
    model = named_resnet_independent_sequence_fcn(
        'resnet50',
        256,
        global_heads = LinearMultiheadDecoder(2048, global_heads),
        dense_heads = Conv2dMultiheadDecoder(256, dense_heads, kernel_size=1)
    )
    
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
