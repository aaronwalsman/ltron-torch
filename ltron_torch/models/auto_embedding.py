import numpy

import torch
from torch.nn import Module, Embedding, ModuleDict, LayerNorm

from ltron.config import Config
from ltron.name_span import NameSpan
from ltron.gym.spaces import (
    MaskedTiledImageSpace, PhaseSpace, TimeStepSpace, MultiScreenPixelSpace,
)
from ltron.compression import batch_deduplicate_from_masks
from ltron.visualization.drawing import stack_images_horizontal, draw_crosshairs

from ltron_torch.models.padding import cat_padded_seqs, cat_multi_padded_seqs
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import TileEmbedding, TokenEmbedding
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.gym_tensor import default_tile_transform

visualization_colors = [
    [  0,   0, 255],
    [255, 255,   0],
    [  0, 255,   0],
    [255,   0, 255],
    [255,   0,   0],
    [  0, 255, 255],
]

class AutoEmbeddingConfig(Config):
    channels = 768
    embedding_dropout = 0.1

class AutoEmbedding(Module):
    def __init__(self,
        config,
        observation_space,
        action_space,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.cursor_colors = {}
        self.cursor_color_n = 0
        
        # find tile and token generating elements of the observation space
        tile_shapes = set()
        self.tile_position_layout = NameSpan()
        self.token_vocabulary_layout = NameSpan()
        self.cursor_fine_layout = NameSpan()
        self.observation_token_names = []
        self.time_step_space = {}
        for name, space in observation_space.items():
            if isinstance(space, MaskedTiledImageSpace):
                self.tile_shape = (
                    space.tile_height, space.tile_width, space.channels)
                tile_shapes.add(self.tile_shape)
                
                mask_shape = (space.mask_height, space.mask_width)
                self.tile_position_layout.add_names(**{name:mask_shape})
            
            if name == 'action':
                for key, subspace in space.items():
                    if isinstance(subspace, PhaseSpace):
                        name_key = '%s,%s'%(name,key)
                        self.observation_token_names.append(name_key)
                        self.token_vocabulary_layout.add_names(
                            **{name_key:subspace.n})
            
            elif isinstance(space, TimeStepSpace):
                self.time_step_space[name] = space
        
        assert len(tile_shapes) == 1
        assert len(self.time_step_space) == 1
        
        # find token generating elements of the action space
        self.readout_layout = NameSpan(PAD=1)
        self.cursor_names = []
        self.noncursor_names = []
        for name, space in action_space.subspaces.items():
            if isinstance(space, MultiScreenPixelSpace):
                self.cursor_names.append(name)
                self.readout_layout.add_names(**{name:1})
                
                # build the cursor fine-grained embedding
                first_key = next(
                    k for k in space.screen_span.keys() if k != 'NO_OP')
                mh = observation_space[first_key].mask_height
                mw = observation_space[first_key].mask_width
                h,w,c = space.screen_span.get_shape(first_key)
                assert h % mh == 0
                assert w % mw == 0
                fh = h//mh
                fw = w//mw
                self.cursor_fine_layout.add_names(**{name:(fh,fw,2)})
                
                self.cursor_colors[name] = []
                for i in range(c):
                    self.cursor_colors[name].append(visualization_colors[
                            self.cursor_color_n % len(visualization_colors)])
                    self.cursor_color_n += 1
            else:
                self.noncursor_names.append(name)
        
        self.cursor_fine_embedding = Embedding(
            self.cursor_fine_layout.total, config.channels)
        self.cursor_fine_norm = LayerNorm(config.channels)
        
        self.readout_layout.add_names(noncursor=1)
        
        # build the tokenizers
        self.tile_embedding = TileEmbedding(
            *self.tile_shape,
            config.channels,
            config.embedding_dropout,
        )
        self.tile_norm = LayerNorm(config.channels)
        
        self.token_embedding = TokenEmbedding(
            self.token_vocabulary_layout.total,
            config.channels,
            config.embedding_dropout,
        )
        self.token_norm = LayerNorm(config.channels)
        
        self.readout_embedding = TokenEmbedding(
            self.readout_layout.total,
            config.channels,
            config.embedding_dropout,
        )
        self.readout_norm = LayerNorm(config.channels)
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.channels, self.tile_position_layout.total, init='normal')
        self.spatial_norm = LayerNorm(config.channels)
        
        (time_step_name, time_step_space), = self.time_step_name.items()
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.channels, time_step_space.max_steps, init='normal')
        self.temporal_norm = LayerNorm(config.channels)
    
    def observation_to_tensors(self, batch, seq_pad, device):
        observation = batch['observation']
        x = {}
        
        # move the time_step to torch/device
        (time_step_name, time_step_space), = self.time_step_name.items()
        time_step = torch.LongTensor(observation[time_step_name]).to(device)
        s, b = time_step.shape[:2]
        
        # move seq_pad to torch/device
        seq_pad = torch.LongTensor(seq_pad).to(device)
        
        # make the tiles, positions and padding
        tile_x = []
        tile_tyx = []
        tile_pad = []
        for name in self.tile_position_layout.keys():
            # make the tiles
            name_tiles, name_tyx, name_pad = batch_deduplicate_from_masks(
                observation[name]['image'],
                observation[name]['tile_mask'],
                observation[time_step_name],
                seq_pad,
                flatten_hw=True,
            )
            name_tiles = default_tile_transform(name_tiles)
            name_tiles = name_tiles.to(device).contiguous()
            
            # make the tile spatial and temporal positions
            tyx_offset = self.tile_position_layout.ravel(name, 0, 0)
            name_tyx[:,:,1] += tyx_offset
            name_tyx = torch.LongTensor(name_tyx).to(device)
            
            # make the pad
            name_pad = torch.LongTensor(name_pad).to(device)
            
            tile_x.append(name_tiles)
            tile_tyx.append(name_tyx)
            tile_pad.append(name_pad)
        
        # concatenate the tile data
        tile_x, cat_tile_pad = cat_multi_padded_seqs(tile_x, tile_pad)
        tile_tyx, _ = cat_multi_padded_seqs(tile_tyx, tile_pad)
        tile_t = tile_tyx[...,0]
        tile_yx = tile_tyx[...,1]
        tile_pad = cat_tile_pad
        
        # make the discrete observation tokens, positions and padding
        token_x = []
        token_t = []
        token_pad = []
        for name in self.observation_token_names:
            keys = name.split(',')
            offset = self.token_vocabulary_layout.ravel(name, 0)
            obs = observation
            for key in keys:
                obs = obs[key]
            token_x.append(torch.LongTensor(obs + offset).to(device))
            token_t.append(time_step)
            token_pad.append(seq_pad)
        
        # concatenate the token data
        token_x, cat_token_pad = cat_multi_padded_seqs(token_x, token_pad)
        token_t, _ = cat_multi_padded_seqs(token_t, token_pad)
        token_pad = cat_token_pad
        
        # make the cursor tokens, positions and padding
        cursor_t = []
        cursor_fine_yxp = []
        cursor_coarse_yx = []
        cursor_pad = []
        for name in self.cursor_names:
            
            # make the observation token for the cursor
            cursor_t.append(time_step)
            name_fine_yxp = []
            name_coarse_yx = []
            for observation_row in observation['action'][name]:
                fine_yxp_row = []
                coarse_yx_row = []
                for o in observation_row:
                    if o == 0:
                        # this case doesn't matter, it only comes up in padding
                        # all cases with real data are >= 1
                        fine_yxp_row.append(0)
                        coarse_yx_row.append(0)
                        continue
                    
                    n, y, x, p = self.action_space.subspaces[name].unravel(o)
                    
                    th, tw, _ = self.cursor_fine_layout.get_shape(name)
                    
                    # compute fine position
                    fine_y = y % th
                    fine_x = x % tw
                    fine_yxp = self.cursor_fine_layout.ravel(
                        name, fine_y, fine_x, p)
                    fine_yxp_row.append(fine_yxp)
                    
                    # compute coarse_position
                    coarse_y = y // th
                    coarse_x = x // tw
                    coarse_yx = self.tile_position_layout.ravel(
                        n, coarse_y, coarse_x)
                    coarse_yx_row.append(coarse_yx)
                
                name_fine_yxp.append(fine_yxp_row)
                name_coarse_yx.append(coarse_yx_row)
            
            cursor_fine_yxp.append(torch.LongTensor(name_fine_yxp).to(device))
            cursor_coarse_yx.append(torch.LongTensor(name_coarse_yx).to(device))
            cursor_pad.append(seq_pad)
        
        # concatenate the cursor data
        cursor_t, cat_cursor_pad = cat_multi_padded_seqs(cursor_t, cursor_pad)
        cursor_fine_yxp, _ = cat_multi_padded_seqs(cursor_fine_yxp, cursor_pad)
        cursor_coarse_yx,_ = cat_multi_padded_seqs(cursor_coarse_yx, cursor_pad)
        cursor_pad = cat_cursor_pad
        
        # make the readout tokens
        readout_x = []
        readout_t = []
        readout_pad = []
        for name in self.readout_layout.keys():
            index = self.readout_layout.ravel(name, 0)
            readout_x.append(torch.full_like(time_step, index))
            readout_t.append(time_step)
            readout_pad.append(seq_pad)
        
        # concatenate the readout data
        readout_x, cat_readout_pad = cat_multi_padded_seqs(
            readout_x, readout_pad)
        readout_t, _ = cat_multi_padded_seqs(readout_t, readout_pad)
        readout_pad = cat_readout_pad
        
        return {
            'tile_x':tile_x,
            'tile_t':tile_t,
            'tile_yx':tile_yx,
            'tile_pad':tile_pad,
            'token_x':token_x,
            'token_t':token_t,
            'token_pad':token_pad,
            'cursor_t':cursor_t,
            'cursor_fine_yxp':cursor_fine_yxp,
            'cursor_coarse_yx':cursor_coarse_yx,
            'cursor_pad':cursor_pad,
            'readout_x':readout_x,
            'readout_t':readout_t,
            'readout_pad':readout_pad,
            'seq_pad':seq_pad,
        }

    
    def forward(self,
        tile_x,
        tile_t,
        tile_yx,
        tile_pad,
        token_x,
        token_t,
        token_pad,
        cursor_t,
        cursor_fine_yxp,
        cursor_coarse_yx,
        cursor_pad,
        readout_x,
        readout_t,
        readout_pad,
        seq_pad,
    ):
        xs = []
        ts = []
        pads = []
        
        # compute token features for the tiles
        # (sometimes there are no tiles)
        if tile_x.shape[0]:
            tile_e = self.tile_embedding(tile_x)
            tile_tpe = self.temporal_position_encoding(tile_t)
            tile_spe = self.spatial_position_encoding(tile_yx)
            tile_x = (
                self.tile_norm(tile_e) +
                self.temporal_norm(tile_tpe) +
                self.spatial_norm(tile_spe)
            )
            xs.append(tile_x)
            ts.append(tile_t)
            pads.append(tile_pad)
        
        # compute token features for the discrete observations
        token_e = self.token_embedding(token_x)
        token_tpe = self.temporal_position_encoding(token_t)
        token_x = (
            self.token_norm(token_e) +
            self.temporal_norm(token_tpe)
        )
        xs.append(token_x)
        ts.append(token_t)
        pads.append(token_pad)
        
        # compute token features for the cursors
        cursor_f = self.cursor_fine_embedding(cursor_fine_yxp)
        cursor_tpe = self.temporal_position_encoding(cursor_t)
        cursor_spe = self.spatial_position_encoding(cursor_coarse_yx)
        cursor_x = (
            self.cursor_fine_norm(cursor_f) +
            self.temporal_norm(cursor_tpe) +
            self.spatial_norm(cursor_spe)
        )
        xs.append(cursor_x)
        ts.append(cursor_t)
        pads.append(cursor_pad)
        
        # compute token features for the readout tokens
        readout_e = self.readout_embedding(readout_x)
        readout_tpe = self.temporal_position_encoding(readout_t)
        readout_x = (
            self.readout_norm(readout_e) +
            self.temporal_norm(readout_tpe)
        )
        xs.append(readout_x)
        ts.append(readout_t)
        pads.append(readout_pad)
        
        # concatenate everything together
        cat_x, cat_pad = cat_multi_padded_seqs(xs, pads)
        cat_t, _ = cat_multi_padded_seqs(ts, pads)
        
        # return
        return cat_x, cat_t, cat_pad
