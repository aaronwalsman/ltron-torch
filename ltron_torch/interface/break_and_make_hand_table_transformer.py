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
        x['table_tiles'] = default_tile_transform(
            table_tiles).to(device).contiguous()
        x['table_t'] = torch.LongTensor(table_tyx[...,0]).to(device)
        x['table_yx'] = torch.LongTensor(table_tyx[...,1:]).to(device)
        x['table_pad'] = torch.LongTensor(table_pad).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_from_masks(
            observation['hand_color_render'],
            observation['hand_tile_mask'],
            observation['step'],
            pad,
        )
        x['hand_tiles'] = default_tile_transform(
            hand_tiles).to(device).contiguous()
        x['hand_t'] = torch.LongTensor(hand_tyx[...,0]).to(device)
        x['hand_yx'] = torch.LongTensor(hand_tyx[...,1:]).to(device)
        x['hand_pad'] = torch.LongTensor(hand_pad).to(device)
        
        # process token x/t/pad
        x['token_x'] = torch.LongTensor(observation['phase']).to(device)
        x['token_t'] = torch.LongTensor(observation['step']).to(device)
        x['token_pad'] = torch.LongTensor(pad).to(device)
        
        # process decode t/pad
        x['decode_t'] = x['token_t'].clone()
        x['decode_pad'] = x['token_pad'].clone()
        
        return x
    
    def augment(self, x, y):
        
        # get the batch size and device
        b = x['token_pad'].shape[0]
        device = x['token_pad'].device
        
        '''
        def dump(name, region, tiles, t, tile_yx, cursor_yx):
            s = tiles.shape[0]
            max_t = torch.max(t)
            frames = numpy.ones(
                (max_t+1, b, 16, 16, 16, 16, 3), dtype=numpy.uint8) * 16
            tiles = tiles * torch.FloatTensor(
                [0.229, 0.224, 0.225]).to(device)
            tiles = tiles + torch.FloatTensor(
                [0.485, 0.456, 0.406]).to(device)
            tiles = (tiles * 255.).cpu().view(s*b,16,16,3)
            tiles = tiles.numpy().astype(numpy.uint8)
            
            bb = numpy.zeros(t.shape, dtype=numpy.long)
            bb[:] = numpy.arange(b).reshape(1,b)
            
            tt = t.view(-1).cpu().numpy()
            
            yy = tile_yx[...,0].view(-1).cpu().numpy()
            xx = tile_yx[...,1].view(-1).cpu().numpy()
            
            frames[tt, bb.reshape(-1), yy, xx] = tiles
            frames = numpy.transpose(
                frames, axes=(0, 1, 2, 4, 3, 5, 6)).reshape(
                max_t+1,b,16*16,16*16,3)
            
            from splendor.image import save_image
            for i in range(max_t):
                for j in range(b):
                    frame = frames[i,j]
                    yy = cursor_yx[i,j,0]
                    xx = cursor_yx[i,j,1]
                    frame[yy*4:(yy+1)*4, xx*4:(xx+1)*4] = [255,255,255]
                    save_image(frame, '%s_%s_%i_%i.png'%(name, region, j, i))
        '''
        
        # apply tile shift augmentation
        if self.config.tile_shift_augmentation:
            '''
            for region in 'table', 'hand':
                dump(
                    'orig',
                    region,
                    x['%s_tiles'%region],
                    x['%s_t'%region],
                    x['%s_yx'%region],
                    y['%s_yx'%region],
                )
            '''
            for region in 'table', 'hand':
                region_yx = x['%s_yx'%region]
                h = getattr(self.config, '%s_tiles_h'%region)
                w = getattr(self.config, '%s_tiles_w'%region)
                up_h = getattr(self.config, '%s_decode_h'%region) // h
                up_w = getattr(self.config, '%s_decode_w'%region) // w
                tile_shifts = []
                cursor_shifts = []
                for i in range(b):
                    
                    # compute shift min/max
                    p = x['%s_pad'%region][i]
                    min_shift_y = -torch.min(region_yx[:p,i,0])
                    max_shift_y = (h-1) - torch.max(region_yx[:p,i,0])
                    min_shift_x = -torch.min(region_yx[:p,i,1])
                    max_shift_x = (w-1) - torch.max(region_yx[:p,i,1])
                    
                    # pick shift randomly within min/max
                    shift_y = random.randint(min_shift_y, max_shift_y)
                    shift_x = random.randint(min_shift_x, max_shift_x)
                    #shift_x = random.choice([-1,0,1])
                    #shift_y = random.choice([-1,0,1])
                    tile_shifts.append([shift_y, shift_x])
                    cursor_shifts.append([shift_y * up_h, shift_x * up_w])
                
                # move shifts to torch/cuda
                tile_shifts = torch.LongTensor(tile_shifts).to(device)
                cursor_shifts = torch.LongTensor(cursor_shifts).to(device)
                
                # shift tile and cursor positions
                x['%s_yx'%region] = x['%s_yx'%region] + tile_shifts
                y['%s_yx'%region] = y['%s_yx'%region] + cursor_shifts
                
                '''
                dump(
                    'offset',
                    region,
                    x['%s_tiles'%region],
                    x['%s_t'%region],
                    x['%s_yx'%region],
                    y['%s_yx'%region],
                )
                '''
        
        # apply simulated misclicks
        if self.config.misclick_augmentation:
            max_t = max(
                torch.max(x['table_t']),
                torch.max(x['hand_t']),
                torch.max(x['token_t']),
            )
            for i in range(b):
                shift_map = torch.arange(max_t+1)
                for j in range(max_t+1):
                    if random.random() < self.config.misclick_augmentation:
                        shift_map[j:] += 1
                
                # shift the table, hand, token and decode time steps
                x['table_t'][:,i] = shift_map[x['table_t'][:,i]]
                x['hand_t'][:,i] = shift_map[x['hand_t'][:,i]]
                x['token_t'][:,i] = shift_map[x['token_t'][:,i]]
                x['decode_t'][:,i] = shift_map[x['decode_t'][:,i]]
        
        return x, y
    
    def forward_rollout(self, terminal, **x):
        device = x['table_tiles'].device
        use_memory = torch.BoolTensor(~terminal).to(device)
        return self.model(**x, use_memory=use_memory)
