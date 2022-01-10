import random

import numpy

import torch
from torch.distributions import Categorical

from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.dataset.paths import get_dataset_info
from ltron.hierarchy import index_hierarchy
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

# ppo model ====================================================================

class ReassemblyPPOModel: #(ActorCriticModel[CategoricalDistr]):
    def __init__(self, config):
        self.model = build_reassembly_model(config)
    
    def forward(self, observations, memory, prev_actions, masks):
        # TODO: is_blind
        
        # chop frames into tiles
        if single_frame:
            # generate workspace tiles
            pad = numpy.ones(b, dtype=numpy.long)
            wx, wi, w_pad = batch_deduplicate_tiled_seqs(
                workspace_frame, pad, tw, th,
                background=prev_workspace_frame,
                s_start=seq_lengths,
            )
            wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
            wi[:,:,0] = 0
            
            # generate handspace tiles
            hx, hi, h_pad = batch_deduplicate_tiled_seqs(
                handspace_frame, pad, tw, th,
                background=prev_handspace_frame,
                s_start=seq_lengths,
            )
            hi = numpy.insert(hi, (0,1,1), -1, axis=-1)
            hi[:,:,0] = 0
            
        elif multi_frame:
            wx, wi, w_pad = batch_deduplicate_tiled_seqs(
            
            )
        
        # TODO[Klemen] check input dimensions of observation and trim to size
        # 5 vs. 4 dimensional multi-agent thing
        
        x_out, x_pad = self.model(
            tile_x, tile_i, tile_pad,
            token_x, token_i, token_pad,
            decoder_i, decoder_pad,
            terminal=terminal,
        )
        
        # separate x_out in to actions and values
        
        ac_out = ActorCriticOutput(
            distrbutions = x_act, values=x_val, extras={},
        )
        
        return ac_out, memory

