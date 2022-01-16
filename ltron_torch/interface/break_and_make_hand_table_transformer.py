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
    misclick_augmentation = 0.15
    tile_shift_augmentation = True
    tile_h = 16
    tile_w = 16
    table_h = 256
    table_w = 256
    hand_h = 96
    hand_w = 96
    
    def set_dependents(self):
        assert self.table_h % self.tile_h == 0
        assert self.table_w % self.table_w == 0
        self.table_tiles_h = self.table_h // self.tile_h
        self.table_tiles_w = self.table_w // self.tile_w
        
        assert self.hand_h % self.tile_h == 0
        assert self.hand_w % self.tile_w == 0
        self.hand_tiles_h = self.hand_h // self.tile_h
        self.hand_tiles_w = self.hand_w // self.tile_w

class BreakAndMakeHandTableTransformerInterface(BreakAndMakeInterface):
    
    def observation_to_tensors(self, observation, pad):
        
        # initialize x
        x = {}
        
        # get the device
        device = next(self.model.parameters()).device
        
        # process table tiles
        table_tiles, table_tyx, table_pad = batch_deduplicate_from_masks(
            observation['table_color_render'],
            observation['table_tile_mask'],
            observation['step'],
            pad,
        )
        x['table_tiles'] = default_tile_transform(table_tiles).to(device)
        x['table_t'] = torch.LongTensor(table_tyx[...,0]).to(device)
        #table_y = table_tyx[...,1]
        #table_x = table_tyx[...,2]
        x['table_yx'] = torch.LongTensor(table_tyx[...,1:]).to(device)
        x['table_pad'] = torch.LongTensor(table_pad).to(device)
        #w = self.config.table_tiles_w
        #table_yx = torch.LongTensor(table_y * w + table_x).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_from_masks(
            observation['hand_color_render'],
            observation['hand_tile_mask'],
            observation['step'],
            pad,
        )
        x['hand_tiles'] = default_tile_transform(hand_tiles).to(device)
        x['hand_t'] = torch.LongTensor(hand_tyx[...,0]).to(device)
        #hand_y = hand_tyx[...,1]
        #hand_x = hand_tyx[...,2]
        x['hand_yx'] = torch.LongTensor(hand_tyx[...,1:]).to(device)
        x['hand_pad'] = torch.LongTensor(hand_pad).to(device)
        #w = self.config.hand_tiles_w
        #offset = self.model.config.table_tiles
        #hand_yx = torch.LongTensor(hand_y * w + hand_x + offset).to(device)
        
        # cat table and hand ties
        '''
        x['tile_x'], x['tile_pad'] = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        x['tile_t'], _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
        x['tile_yx'], _ = cat_padded_seqs(
            table_yx, hand_yx, table_pad, hand_pad)
        '''
        
        # process token x/t/pad
        x['token_x'] = torch.LongTensor(observation['phase']).to(device)
        x['token_t'] = torch.LongTensor(observation['step']).to(device)
        x['token_pad'] = torch.LongTensor(pad).to(device)
        
        # process decode t/pad
        x['decode_t'] = x['token_t']
        x['decode_pad'] = x['token_pad']
        
        return x
    
    def augment(self, x, y):
        
        return x, y
        
        '''
        # apply yx shift augmentation
        #def tile_shift_augmentation(y, x, h, w, pad):
        b = x['token_pad'].shape[0]
        for region in 'table', 'hand':
            w = getattr(self.config, '%s_tiles_w'%region)
            if region == 'hand':
                offset = self.config.table_tiles
            else:
                offset = 0
            
            yx = x['tile_yx']
            import pdb
            pdb.set_trace()
            
            for i in range(b):
                # shift tiles
                p = x['token_pad'][i]
                min_shift_y = -numpy.min(y[:p,i])
                max_shift_y = (h-1) - numpy.max(y[:p,i])
                min_shift_x = -numpy.min(x[:p,i])
                max_shift_x = (w-1) - numpy.max(x[:p,i])
                y_shift = random.randint(min_shift_y, max_shift_y)
                x_shift = random.randint(min_shift_x, max_shift_y)
                
                y[:p,i] += y_shift
                x[:p,i] += x_shift
                
                # shift click positions
                
                import pdb
                pdb.set_trace()
            
        
        if train_mode == 'train':
            if self.config.tile_shift_augmentation:
                tile_shift_augmentation(
                    table_y,
                    table_x,
                    self.config.table_tiles_h,
                    self.config.table_tiles_w,
                    table_pad,
                )
                tile_shift_augmentation(
                    hand_y,
                    hand_x,
                    self.config.hand_tiles_h,
                    self.config.hand_tiles_w,
                    hand_pad,
                )
        
        # apply simulated misclicks
        if self.config.misclick_augmentation:
            b = tile_t.shape[1]
            max_t = max(
                torch.max(tile_t),
                torch.max(token_t),
            )
            for i in range(b):
                shift_map = torch.arange(max_t+1)
                for j in range(max_t+1):
                    if random.random() < self.config.misclick_augmentation:
                        shift_map[j:] += 1
                p = tile_pad[i]
                tile_t[:p,i] = shift_map[tile_t[:p,i]]
                p = token_pad[i]
                token_t[:p,i] = shift_map[token_t[:p,i]]
        '''
    
    def forward_rollout(self, terminal, *x):
        use_memory = torch.BoolTensor(~terminal).to(x[0].device)
        return self.model(*x, use_memory=use_memory)
