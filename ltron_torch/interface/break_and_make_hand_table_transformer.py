import random
import os

import numpy

import torch
from torch.nn.functional import cross_entropy

import tqdm

from ltron.compression import batch_deduplicate_from_masks

from ltron_torch.gym_tensor import default_tile_transform
from ltron_torch.models.padding import cat_padded_seqs
from ltron_torch.interface.break_and_make import (
    BreakAndMakeInterfaceConfig,
    BreakAndMakeInterface,
)

class BreakAndMakeHandTableTransformerInterfaceConfig(
    BreakAndMakeInterfaceConfig
):
    simulate_misclick = 0.15
    shift_images = True

class BreakAndMakeHandTableTransformerInterface(BreakAndMakeInterface):
    def observation_to_tensors(self, observation, pad):
        # get the device
        device = next(self.model.parameters()).device
        
        # process table tiles
        table_tiles, table_tyx, table_pad = batch_deduplicate_from_masks(
            observation['table_color_render'],
            observation['table_tile_mask'],
            observation['step'],
            pad,
        )
        
        table_pad = torch.LongTensor(table_pad).to(device)
        table_tiles = default_tile_transform(table_tiles).to(device)
        table_t = torch.LongTensor(table_tyx[...,0]).to(device)
        table_yx = torch.LongTensor(
            table_tyx[...,1] *
            self.model.config.table_tiles_w +
            table_tyx[...,2],
        ).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_from_masks(
            observation['hand_color_render'],
            observation['hand_tile_mask'],
            observation['step'],
            pad,
        )
        
        hand_pad = torch.LongTensor(hand_pad).to(device)
        hand_tiles = default_tile_transform(hand_tiles).to(device)
        hand_t = torch.LongTensor(hand_tyx[...,0]).to(device)
        hand_yx = torch.LongTensor(
            hand_tyx[...,1] *
            self.model.config.hand_tiles_w +
            hand_tyx[...,2] +
            self.model.config.table_tiles,
        ).to(device)
        
        # cat table and hand ties
        tile_x, tile_pad = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        tile_t, _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
        tile_yx, _ = cat_padded_seqs(table_yx, hand_yx, table_pad, hand_pad)
        
        # process token x/t/pad
        token_x = torch.LongTensor(observation['phase']).to(device)
        token_t = torch.LongTensor(observation['step']).to(device)
        token_pad = torch.LongTensor(pad).to(device)
        
        # augmentations
        if self.config.simulate_misclick:
            b = tile_t.shape[1]
            max_t = max(
                torch.max(tile_t),
                torch.max(token_t),
            )
            for i in range(b):
                shift_map = torch.arange(max_t+1)
                for j in range(max_t+1):
                    if random.random() < self.config.simulate_misclick:
                        shift_map[j:] += 1
                tile_t[:,i] = shift_map[tile_t[:,i]]
                token_t[:,i] = shift_map[token_t[:,i]]
        
        if self.config.shift_images:
            pass
        
        # process decode t/pad
        decode_t = token_t
        decode_pad = token_pad
        
        return (
            tile_x, tile_t, tile_yx, tile_pad,
            token_x, token_t, token_pad,
            decode_t, decode_pad,
        )
    
    def forward_rollout(self, terminal, *x):
        use_memory = torch.BoolTensor(~terminal).to(x[0].device)
        return self.model(*x, use_memory=use_memory)
