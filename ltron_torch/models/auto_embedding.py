import numpy

import torch
from torch.nn import Module, Embedding, ModuleDict, LayerNorm

from gym.spaces import Discrete

from ltron.constants import MAX_SNAPS_PER_BRICK
from ltron.config import Config
from ltron.name_span import NameSpan
from ltron.gym.spaces import (
    MaskedTiledImageSpace,
    AssemblySpace,
    PhaseSpace,
    TimeStepSpace,
    MultiScreenPixelSpace,
    MultiScreenInstanceSnapSpace,
    SymbolicSnapSpace,
)
from ltron.compression import batch_deduplicate_from_masks
from ltron.visualization.drawing import stack_images_horizontal, draw_crosshairs

from ltron_torch.models.padding import cat_padded_seqs, cat_multi_padded_seqs
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import (
    TemporalEmbedding,
    #TileEmbedding,
    #OldTokenEmbedding,
    #TokenEmbedding,
    build_shared_assembly_embeddings,
    #AssemblyEmbedding,
    build_shared_masked_tiled_image_embeddings,
    DiscreteEmbedding,
    MultiDiscreteEmbedding,
)
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
        readout_layout,
        #action_space,
    ):
        super().__init__()
        
        self.observation_space = observation_space
        self.readout_layout = readout_layout
        #self.action_space = action_space
        
        #self.cursor_colors = {}
        #self.cursor_color_n = 0
        
        # find tile and token generating elements of the observation space
        tile_shapes = set()
        #self.tile_position_layout = NameSpan()
        #self.token_vocabulary_layout = NameSpan()
        #self.cursor_fine_layout = NameSpan()
        embeddings = {}
        #self.assembly_layout = NameSpan()
        #self.observation_token_names = []
        
        self.time_step_name = None
        self.time_step_space = None
        
        '''
        assembly_shape_embedding = None
        assembly_shape_norm = None
        assembly_color_embedding = None
        assembly_color_norm = None
        assembly_pose_embedding = None
        assembly_pose_norm = None
        '''
        
        # build the temporal embedding
        for name, space in observation_space.items():
            if isinstance(space, TimeStepSpace):
                assert self.time_step_space is None
                self.time_step_name = name
                self.time_step_space = space
        
        assert self.time_step_space is not None
        self.temporal_embedding = TemporalEmbedding(
            self.time_step_space, config.channels)
        
        # build the readout embedding
        self.readout_embedding = DiscreteEmbedding(
            readout_layout.total,
            config.channels,
            config.embedding_dropout,
            self.temporal_embedding,
        )
        
        # build the tile embeddings
        tile_subspaces = {
            name:subspace for name, subspace in observation_space.items()
            if isinstance(subspace, MaskedTiledImageSpace)
        }
        if len(tile_subspaces):
            tile_embeddings = build_shared_masked_tiled_image_embeddings(
                tile_subspaces,
                config.channels,
                config.embedding_dropout,
                self.temporal_embedding,
            )
            embeddings.update(tile_embeddings)
        
        # build assembly embeddings
        assembly_subspaces = {
            name:subspace for name, subspace in observation_space.items()
            if isinstance(subspace, AssemblySpace)
        }
        if len(assembly_subspaces):
            assembly_embeddings = build_shared_assembly_embeddings(
                assembly_subspaces,
                config.channels,
                config.embedding_dropout,
                self.temporal_embedding,
            )
            embeddings.update(assembly_embeddings)
        
        # build other embeddings
        for name, space in observation_space.items():
            #if isinstance(space, MaskedTiledImageSpace):
            #    self.tile_shape = (
            #        space.tile_height, space.tile_width, space.channels)
            #    tile_shapes.add(self.tile_shape)
            #    
            #    mask_shape = (space.mask_height, space.mask_width)
            #    self.tile_position_layout.add_names(**{name:mask_shape})
            
            '''
            elif isinstance(space, AssemblySpace):
                # add to assembly span
                embedding = AssemblyEmbedding(
                    space,
                    config.channels,
                    config.embedding_dropout,
                    self.temporal_embedding,
                    shape_embedding = assembly_shape_embedding,
                    shape_norm = assembly_shape_norm,
                    color_embedding = assembly_color_embedding,
                    color_norm = assembly_color_norm,
                    pose_embedding = assembly_pose_embedding,
                    pose_norm = assembly_pose_norm,
                    #temporal_embedding = self.temporal_embedding,
                    #temporal_norm = self.temporal_norm,
                )
                embeddings[name] = embedding
                
                assembly_shape_embedding = embedding.shape_embedding
                assembly_shape_norm = embedding.shape_norm
                assembly_color_embedding = embedding.color_embedding
                assembly_color_norm = embedding.color_norm
                assembly_pose_embedding = embedding.pose_embedding
                assembly_pose_norm = embedding.pose_norm
            '''
            #if name == 'action':
            #    for key, subspace in space.items():
            if isinstance(space, MultiScreenInstanceSnapSpace):
                #embeddings[key] = TokenEmbedding(
                #embeddings[name] = TokenEmbedding(
                #    subspace,
                #    config.channels,
                #    config.embedding_dropout,
                #    temporal_embedding,
                #    #temporal_embedding=self.temporal_embedding,
                #    #temporal_norm=self.temporal_norm,
                #)
                embeddings[name] = MultiDiscreteEmbedding(
                    space,
                    config.channels,
                    config.embedding_dropout,
                    self.temporal_embedding,
                )
            
            elif isinstance(space, TimeStepSpace):
                # this case avoids generating an embedding for the time step
                pass
            
            elif isinstance(space, Discrete):
                #name_key = '%s,%s'%(name,key)
                #self.observation_token_names.append(name_key)
                #self.observation_token_names.append(name)
                #self.token_vocabulary_layout.add_names(
                #    **{name_key:subspace.n})
                embeddings[name] = DiscreteEmbedding(
                    space.n,
                    config.channels,
                    config.embedding_dropout,
                    self.temporal_embedding,
                )
        
        #assert len(tile_shapes) <= 1
        
        # find token generating elements of the action space
        #self.readout_layout = NameSpan(PAD=1)
        '''
        self.visual_cursor_names = []
        self.symbolic_cursor_names = []
        self.noncursor_names = []
        for name, space in action_space.subspaces.items():
            if isinstance(space, MultiScreenPixelSpace):
                self.visual_cursor_names.append(name)
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
            elif isinstance(space, SymbolicSnapSpace):
                self.readout_layout.add_names(**{name:1})
                self.symbolic_cursor_names.append(name)
            elif isinstance(space, BrickShapeColorSpace):
                self.readout_layout.add_names(**{name:1})
                self.brick_inserters[name] = (
                    space.num_shapes, space.num_colors)
            else:
                self.noncursor_names.append(name)
        
        self.readout_layout.add_names(noncursor=1)
        '''
        
        # build the tokenizers
        
        # auto
        self.embeddings = ModuleDict(embeddings)
        
        # image tiles
        #if self.tile_position_layout.total:
        #    self.tile_embedding = TileEmbedding(
        #        *self.tile_shape,
        #        config.channels,
        #        config.embedding_dropout,
        #    )
        #    self.tile_norm = LayerNorm(config.channels)
        
        # discrete input tokens
        #self.token_embedding = OldTokenEmbedding(
        #    self.token_vocabulary_layout.total,
        #    config.channels,
        #    config.embedding_dropout,
        #)
        #self.token_norm = LayerNorm(config.channels)
        
        # visual cursor fine embedding
        #if len(self.visual_cursor_names):
        #    self.visual_cursor_fine_embedding = Embedding(
        #        self.cursor_fine_layout.total, config.channels)
        #    self.visual_cursor_fine_norm = LayerNorm(config.channels)
        
        # readout tokens
        #self.readout_embedding = OldTokenEmbedding(
        #    self.readout_layout.total,
        #    config.channels,
        #    config.embedding_dropout,
        #)
        #self.readout_norm = LayerNorm(config.channels)
        
        # build the positional encodings ---------------------------------------
        
        # spatial
        #self.spatial_position_encoding = LearnedPositionalEncoding(
        #    config.channels, self.tile_position_layout.total, init='normal')
        #self.spatial_norm = LayerNorm(config.channels)
    
    def observation_to_tensors(self, batch, seq_pad):
        device = next(iter(self.parameters())).device
        observation = batch['observation']
        
        # move the time_step to torch/device
        #(time_step_name, time_step_space), = self.time_step_space.items()
        time_step = torch.LongTensor(
            observation[self.time_step_name]).to(device)
        s, b = time_step.shape[:2]
        
        '''
        # make the tiles, positions and padding
        if self.tile_position_layout.total:
            tile_x = []
            tile_tyx = []
            tile_pad = []
            for name in self.tile_position_layout.keys():
                
                # make the tiles
                name_tiles, name_tyx, name_pad = batch_deduplicate_from_masks(
                    observation[name]['image'],
                    observation[name]['tile_mask'],
                    observation[self.time_step_name],
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
        else:
            tile_x = None
            tile_t = None
            tile_yx = None
            tile_pad = torch.zeros(
                seq_pad.shape, dtype=torch.long, device=device)
        '''
        
        # move seq_pad to torch/device
        seq_pad_t = torch.LongTensor(seq_pad).to(device)
        
        '''
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
            token_pad.append(seq_pad_t)
        
        # concatenate the token data
        token_x, cat_token_pad = cat_multi_padded_seqs(token_x, token_pad)
        token_t, _ = cat_multi_padded_seqs(token_t, token_pad)
        token_pad = cat_token_pad
        
        # make the cursor tokens, positions and padding
        if len(self.visual_cursor_names):
            visual_cursor_t = []
            visual_cursor_fine_yxp = []
            visual_cursor_coarse_yx = []
            visual_cursor_pad = []
            for name in self.visual_cursor_names:
                
                # make the observation token for the cursor
                visual_cursor_t.append(time_step)
                name_fine_yxp = []
                name_coarse_yx = []
                for observation_row in observation['action'][name]:
                    fine_yxp_row = []
                    coarse_yx_row = []
                    for o in observation_row:
                        if o == 0:
                            # this case doesn't matter, it only comes up in
                            # padding all cases with real data are >= 1
                            fine_yxp_row.append(0)
                            coarse_yx_row.append(0)
                            continue
                        
                        n,y,x,p = self.action_space.subspaces[name].unravel(o)
                        
                        th,tw,_ = self.cursor_fine_layout.get_shape(name)
                        
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
                
                visual_cursor_fine_yxp.append(
                    torch.LongTensor(name_fine_yxp).to(device))
                visual_cursor_coarse_yx.append(
                    torch.LongTensor(name_coarse_yx).to(device))
                visual_cursor_pad.append(seq_pad_t)
            
            # concatenate the cursor data
            visual_cursor_t, cat_visual_cursor_pad = cat_multi_padded_seqs(
                visual_cursor_t, visual_cursor_pad)
            visual_cursor_fine_yxp, _ = cat_multi_padded_seqs(
                visual_cursor_fine_yxp, visual_cursor_pad)
            visual_cursor_coarse_yx,_ = cat_multi_padded_seqs(
                visual_cursor_coarse_yx, visual_cursor_pad)
            visual_cursor_pad = cat_visual_cursor_pad
        
        else:
            visual_cursor_t = None
            visual_cursor_fine_yxp = None
            visual_cursor_coarse_yx = None
            visual_cursor_pad = torch.zeros_like(seq_pad_t)
        '''
        
        auto_x = {}
        auto_t = {}
        auto_pad = {}
        for name, embedding in self.embeddings.items():
            #embedding = self.embeddings[name]
            name_obs = observation[name]
            # ugh
            #if name in observation:
            #    name_obs = observation[name]
            #else:
            #    name_obs = observation['action'][name]
            a_x, a_t, a_pad = embedding.observation_to_tensors(
                name_obs,
                observation[self.time_step_name],
                seq_pad,
                device,
            )
            auto_x[name] = a_x
            auto_t[name] = a_t
            auto_pad[name] = a_pad
        
        # make the readout tokens
        readout_x = []
        readout_t = []
        readout_pad = []
        for name in self.readout_layout.keys():
            if name == 'PAD':
                continue
            index = self.readout_layout.ravel(name, 0)
            readout_x.append(torch.full_like(time_step, index))
            readout_t.append(time_step)
            readout_pad.append(seq_pad_t)
        
        # concatenate the readout data
        readout_x, cat_readout_pad = cat_multi_padded_seqs(
            readout_x, readout_pad)
        readout_t, _ = cat_multi_padded_seqs(readout_t, readout_pad)
        readout_pad = cat_readout_pad
        
        assert 'readout' not in auto_x
        auto_x['readout'] = readout_x
        auto_t['readout'] = readout_t
        auto_pad['readout'] = readout_pad
        
        return {'x':auto_x, 't':auto_t, 'pad':auto_pad}
        '''
        return {
            'tile_x':tile_x,
            'tile_t':tile_t,
            'tile_yx':tile_yx,
            'tile_pad':tile_pad,
            'token_x':token_x,
            'token_t':token_t,
            'token_pad':token_pad,
            'visual_cursor_t':visual_cursor_t,
            'visual_cursor_fine_yxp':visual_cursor_fine_yxp,
            'visual_cursor_coarse_yx':visual_cursor_coarse_yx,
            'visual_cursor_pad':visual_cursor_pad,
            'auto_x':auto_x,
            'auto_t':auto_t,
            'auto_pad':auto_pad,
            'readout_x':readout_x,
            'readout_t':readout_t,
            'readout_pad':readout_pad,
            'seq_pad':seq_pad_t,
        }
        '''
    
    def forward(self, x, t, pad):
    #    tile_x,
    #    tile_t,
    #    tile_yx,
    #    tile_pad,
    #    token_x,
    #    token_t,
    #    token_pad,
    #    visual_cursor_t,
    #    visual_cursor_fine_yxp,
    #    visual_cursor_coarse_yx,
    #    visual_cursor_pad,
    #    auto_x,
    #    auto_t,
    #    auto_pad,
    #    readout_x,
    #    readout_t,
    #    readout_pad,
    #    seq_pad,
    #):
        out_x = {}
        out_t = {}
        out_pad = {}
        #out_t = {}
        #out_pad = {}
        #name_order = []
        
        # compute token features for the tiles
        # (sometimes there are no tiles)
        #if tile_x is not None and tile_x.shape[0]:
        #    tile_e = self.tile_embedding(tile_x)
        #    tile_tpe = self.temporal_embedding(tile_t)
        #    tile_spe = self.spatial_position_encoding(tile_yx)
        #    tile_x = (
        #        self.tile_norm(tile_e) +
        #        #self.temporal_norm(tile_tpe) +
        #        tile_tpe +
        #        self.spatial_norm(tile_spe)
        #    )
        #    xs.append(tile_x)
        #    ts.append(tile_t)
        #    pads.append(tile_pad)
        
        # compute token features for the discrete observations
        #token_e = self.token_embedding(token_x)
        #token_tpe = self.temporal_embedding(token_t)
        #token_x = (
        #    self.token_norm(token_e) +
        #    #self.temporal_norm(token_tpe)
        #    token_tpe
        #)
        #xs.append(token_x)
        #ts.append(token_t)
        #pads.append(token_pad)
        #
        ## compute token features for the visual cursors
        #if visual_cursor_t is not None:
        #    visual_cursor_f = self.visual_cursor_fine_embedding(
        #        visual_cursor_fine_yxp)
        #    visual_cursor_tpe = self.temporal_embedding(visual_cursor_t)
        #    visual_cursor_spe = self.spatial_position_encoding(
        #        visual_cursor_coarse_yx)
        #    visual_cursor_x = (
        #        self.visual_cursor_fine_norm(visual_cursor_f) +
        #        #self.temporal_norm(visual_cursor_tpe) +
        #        visual_cursor_tpe +
        #        self.spatial_norm(visual_cursor_spe)
        #    )
        #    xs.append(visual_cursor_x)
        #    ts.append(visual_cursor_t)
        #    pads.append(visual_cursor_pad)
        
        # embeddings
        for name, embedding in self.embeddings.items():
            out_x[name], out_t[name], out_pad[name] = embedding(
                **x[name], t=t[name], pad=pad[name]
            )
            #ts.append(auto_t[name])
            #pads.append(auto_pad[name])
            #name_order.append(name)
        
        name = 'readout'
        out_x[name], out_t[name], out_pad[name] = self.readout_embedding(
            x[name], t[name], pad[name])
        
        ## compute token features for the readout tokens
        #readout_e = self.readout_embedding(readout_x)
        #readout_tpe = self.temporal_embedding(readout_t)
        #readout_x = (
        #    self.readout_norm(readout_e) +
        #    #self.temporal_norm(readout_tpe)
        #    readout_tpe
        #)
        #xs.append(readout_x)
        #ts.append(readout_t)
        #pads.append(readout_pad)
        
        # concatenate everything together
        #cat_x, cat_pad = cat_multi_padded_seqs(xs, pads)
        #cat_t, _ = cat_multi_padded_seqs(ts, pads)
        
        # return
        return out_x, t, pad #cat_x, cat_t, cat_pad
