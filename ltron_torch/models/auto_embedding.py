import numpy

import torch
from torch.nn import Module, Embedding, ModuleDict, LayerNorm

from ltron.constants import MAX_SNAPS_PER_BRICK
from ltron.config import Config
from ltron.name_span import NameSpan
from ltron.gym.spaces import (
    MaskedTiledImageSpace,
    AssemblySpace,
    PhaseSpace,
    TimeStepSpace,
    MultiScreenPixelSpace,
    SymbolicSnapSpace,
)
from ltron.compression import batch_deduplicate_from_masks
from ltron.visualization.drawing import stack_images_horizontal, draw_crosshairs

from ltron_torch.models.padding import cat_padded_seqs, cat_multi_padded_seqs
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import (
    TileEmbedding,
    OldTokenEmbedding,
    TokenEmbedding,
    AssemblyEmbedding,
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
        #self.symbolic_cursor_layout = NameSpan()
        auto_embeddings = {}
        self.assembly_layout = NameSpan()
        self.observation_token_names = []
        
        self.time_step_name = None
        self.time_step_space = None
        
        assembly_shape_embedding = None
        assembly_shape_norm = None
        assembly_color_embedding = None
        assembly_color_norm = None
        assembly_pose_embedding = None
        assembly_pose_norm = None
        
        # find the time step space
        for name, space in observation_space.items():
            if isinstance(space, TimeStepSpace):
                assert self.time_step_space is None
                self.time_step_name = name
                self.time_step_space = space
        
        self.temporal_embedding = Embedding(
            self.time_step_space.max_steps, config.channels)
        self.temporal_norm = LayerNorm(config.channels)
        
        for name, space in observation_space.items():
            if isinstance(space, MaskedTiledImageSpace):
                self.tile_shape = (
                    space.tile_height, space.tile_width, space.channels)
                tile_shapes.add(self.tile_shape)
                
                mask_shape = (space.mask_height, space.mask_width)
                self.tile_position_layout.add_names(**{name:mask_shape})
            
            elif isinstance(space, AssemblySpace):
                # add to assembly span
                #self.assembly_layout.add_names(**{name:space.max_instances})
                
                #self.num_shapes = numpy.max(space['shape'].high + 1)
                #self.num_colors = numpy.max(space['color'].high + 1)
                
                embedding = AssemblyEmbedding(
                    space,
                    config.channels,
                    config.embedding_dropout,
                    shape_embedding = assembly_shape_embedding,
                    shape_norm = assembly_shape_norm,
                    color_embedding = assembly_color_embedding,
                    color_norm = assembly_color_norm,
                    pose_embedding = assembly_pose_embedding,
                    pose_norm = assembly_pose_norm,
                    temporal_embedding = self.temporal_embedding,
                    temporal_norm = self.temporal_norm,
                )
                auto_embeddings[name] = embedding
                    
                assembly_shape_embedding = embedding.shape_embedding
                assembly_shape_norm = embedding.shape_norm
                assembly_color_embedding = embedding.color_embedding
                assembly_color_norm = embedding.color_norm
                assembly_pose_embedding = embedding.pose_embedding
                assembly_pose_norm = embedding.pose_norm
            
            if name == 'action':
                for key, subspace in space.items():
                    if isinstance(subspace, PhaseSpace):
                        name_key = '%s,%s'%(name,key)
                        self.observation_token_names.append(name_key)
                        self.token_vocabulary_layout.add_names(
                            **{name_key:subspace.n})
            
                    elif isinstance(subspace, SymbolicSnapSpace):
                        #self.symbolic_cursor_layout.add_names(
                        #    **{key:subspace.span})
                        auto_embeddings[key] = TokenEmbedding(
                            subspace,
                            config.channels,
                            config.embedding_dropout,
                            temporal_embedding=self.temporal_embedding,
                            temporal_norm=self.temporal_norm,
                        )
        
        assert len(tile_shapes) <= 1
        
        # find token generating elements of the action space
        self.readout_layout = NameSpan(PAD=1)
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
            else:
                self.noncursor_names.append(name)
        
        self.readout_layout.add_names(noncursor=1)
        
        # build the tokenizers -------------------------------------------------
        
        # auto
        self.auto_embeddings = ModuleDict(auto_embeddings)
        
        # image tiles
        if self.tile_position_layout.total:
            self.tile_embedding = TileEmbedding(
                *self.tile_shape,
                config.channels,
                config.embedding_dropout,
            )
            self.tile_norm = LayerNorm(config.channels)
        
        ## symbolic assemblies
        #if self.assembly_layout.total:
        #    
        #    # shape
        #    self.shape_embedding = Embedding(self.num_shapes, config.channels)
        #    self.shape_norm = LayerNorm(config.channels)
        #    
        #    # color
        #    self.color_embedding = Embedding(self.num_colors, config.channels)
        #    self.color_norm = LayerNorm(config.channels)
        #    
        #    # se3
        #    self.pose_embedding = SE3Embedding(
        #        config.channels, config.embedding_dropout)
        #    self.pose_norm = LayerNorm(config.channels)
        #    
        #    # position
        #    self.symbolic_instance_embedding = Embedding(
        #        self.assembly_layout.total, config.channels)
        #    self.symbolic_instance_norm = LayerNorm(config.channels)
        
        # discrete input tokens
        self.token_embedding = OldTokenEmbedding(
            self.token_vocabulary_layout.total,
            config.channels,
            config.embedding_dropout,
        )
        self.token_norm = LayerNorm(config.channels)
        
        # visual cursor fine embedding
        if len(self.visual_cursor_names):
            self.visual_cursor_fine_embedding = Embedding(
                self.cursor_fine_layout.total, config.channels)
            self.visual_cursor_fine_norm = LayerNorm(config.channels)
        
        '''
        if len(self.symbolic_cursor_names):
            self.symbolic_cursor_instance_embedding = Embedding(
                self.symbolic_cursor_layout.total, config.channels)
            #self.symbolic_cursor_snap_embedding = Embedding(
            #    MAX_SNAPS_PER_BRICK, config.channels)
        '''
        
        # readout tokens
        self.readout_embedding = OldTokenEmbedding(
            self.readout_layout.total,
            config.channels,
            config.embedding_dropout,
        )
        self.readout_norm = LayerNorm(config.channels)
        
        # build the positional encodings ---------------------------------------
        
        # spatial
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.channels, self.tile_position_layout.total, init='normal')
        self.spatial_norm = LayerNorm(config.channels)
        
        # temporal
        # TEMP
        #self.temporal_position_encoding = LearnedPositionalEncoding(
        #    config.channels, self.time_step_space.max_steps, init='normal')
        #self.temporal_norm = LayerNorm(config.channels)
    
    def observation_to_tensors(self, batch, seq_pad):
        device = next(iter(self.parameters())).device
        observation = batch['observation']
        
        # move the time_step to torch/device
        #(time_step_name, time_step_space), = self.time_step_space.items()
        time_step = torch.LongTensor(
            observation[self.time_step_name]).to(device)
        s, b = time_step.shape[:2]
        
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
        
        # make the symbolic assembly shape, color, pose and padding
        #assembly_shape = []
        #assembly_color = []
        #assembly_pad = []
        #assembly_instance = []
        #assembly_t = []
        
        '''
        assembly_scit = []
        assembly_pose = []
        assembly_pad = []
        for name in self.assembly_layout.keys():
            
            # make the name/shape/pose
            name_shape = observation[name]['shape']
            name_color = observation[name]['color']
            name_pose = observation[name]['pose']
            
            nonzero_instances = name_shape != 0
            num_instances = nonzero_instances.sum(axis=-1)
            max_instances = num_instances.max()
            name_pad = nonzero_instances.sum(axis=(0,-1))
            assembly_pad.append(torch.LongTensor(name_pad).to(device))
            
            s_coord, b_coord, i_coord = numpy.where(name_shape)
            j_coord = numpy.concatenate([numpy.arange(cc) for cc in name_pad])
            
            # make the padded shapes
            compressed_name_shape = name_shape[s_coord, b_coord, i_coord]
            padded_name_shape = numpy.zeros(
                (b, max(name_pad)), dtype=numpy.long)
            padded_name_shape[b_coord, j_coord] = compressed_name_shape
            #assembly_shape.append(
            #    torch.LongTensor(padded_name_shape).to(device))
            
            # make the padded colors
            compressed_name_color = name_color[s_coord, b_coord, i_coord]
            padded_name_color = numpy.zeros(
                (b, max(name_pad)), dtype=numpy.long)
            padded_name_color[b_coord, j_coord] = compressed_name_color
            #assembly_color.append(
            #    torch.LongTensor(padded_name_color).to(device))
            
            # make the padded instance indices
            padded_name_instance = numpy.zeros_like(padded_name_shape)
            padded_name_instance[b_coord, j_coord] = i_coord
            #assembly_instance.append(
            #    torch.LongTensor(padded_name_instance).to(device))
            
            k = name_shape.shape[-1]
            name_t = time_step.reshape(s, b, 1).expand(s, b, k)
            compressed_name_t = name_t[s_coord, b_coord, i_coord]
            padded_name_t = torch.zeros(
                padded_name_shape.shape, dtype=torch.long, device=device)
            padded_name_t[b_coord, j_coord] = compressed_name_t
            #assembly_t.append(padded_name_t)
            
            padded_name_scit = torch.stack(
                [
                 torch.LongTensor(padded_name_shape).to(device),
                 torch.LongTensor(padded_name_color).to(device),
                 torch.LongTensor(padded_name_instance).to(device),
                 padded_name_t,
                ],
                dim=-1,
            ).permute(1,0,2)
            assembly_scit.append(padded_name_scit)
            
            compressed_name_pose = name_pose[s_coord, b_coord, i_coord]
            padded_name_pose = numpy.zeros((b, max(name_pad), 4, 4))
            padded_name_pose[b_coord, j_coord] = compressed_name_pose
            padded_name_pose = torch.FloatTensor(
                padded_name_pose).to(device).permute(1,0,2,3)
            assembly_pose.append(padded_name_pose)
        
        assembly_scit, cat_assembly_pad = cat_multi_padded_seqs(
            assembly_scit, assembly_pad)
        assembly_shape = assembly_scit[...,0]
        assembly_color = assembly_scit[...,1]
        assembly_instance = assembly_scit[...,2]
        assembly_t = assembly_scit[...,3]
        assembly_pose, _ = cat_multi_padded_seqs(
            assembly_pose, assembly_pad)
        assembly_pad = cat_assembly_pad
        '''
        
        # move seq_pad to torch/device
        seq_pad_t = torch.LongTensor(seq_pad).to(device)
        
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
        if len(self.symbolic_cursor_names):
            symbolic_cursor_x = []
            symbolic_cursor_t = []
            symbolic_cursor_pad = []
            for name in self.symbolic_cursor_names:
                symbolic_cursor_x.append(
                    torch.LongTensor(observation['action'][name]).to(device))
                symbolic_cursor_t.append(time_step)
                symbolic_cursor_pad.append(seq_pad)
            
            symbolic_cursor_x, cat_symbolic_cursor_pad = cat_multi_padded_seqs(
                symbolic_cursor_x, symbolic_cursor_pad)
            symbolic_cursor_t, _ = cat_multi_padded_seqs(
                symbolic_cursor_t, symbolic_cursor_pad)
            symbolic_cursor_pad = cat_symbolic_cursor_pad
        
        else:
            symbolic_cursor_x = None
            symbolic_cursor_t = None
            symbolic_cursor_pad = None
        '''
        auto_x = {}
        auto_t = {}
        auto_pad = {}
        for name in self.auto_embeddings:
            auto_embedding = self.auto_embeddings[name]
            # ugh
            if name in observation:
                name_obs = observation[name]
            else:
                name_obs = observation['action'][name]
            a_x, a_t, a_pad = auto_embedding.observation_to_tensors(
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
            index = self.readout_layout.ravel(name, 0)
            readout_x.append(torch.full_like(time_step, index))
            readout_t.append(time_step)
            readout_pad.append(seq_pad_t)
        
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
            #'assembly_shape':assembly_shape,
            #'assembly_color':assembly_color,
            #'assembly_instance':assembly_instance,
            #'assembly_pose':assembly_pose,
            #'assembly_t':assembly_t,
            #'assembly_pad':assembly_pad,
            'visual_cursor_t':visual_cursor_t,
            'visual_cursor_fine_yxp':visual_cursor_fine_yxp,
            'visual_cursor_coarse_yx':visual_cursor_coarse_yx,
            'visual_cursor_pad':visual_cursor_pad,
            #'symbolic_cursor_x':symbolic_cursor_x,
            #'symbolic_cursor_t':symbolic_cursor_t,
            #'symbolic_cursor_pad':symbolic_cursor_pad,
            'auto_x':auto_x,
            'auto_t':auto_t,
            'auto_pad':auto_pad,
            'readout_x':readout_x,
            'readout_t':readout_t,
            'readout_pad':readout_pad,
            'seq_pad':seq_pad_t,
        }

    
    def forward(self,
        tile_x,
        tile_t,
        tile_yx,
        tile_pad,
        token_x,
        token_t,
        token_pad,
        #assembly_shape,
        #assembly_color,
        #assembly_instance,
        #assembly_pose,
        #assembly_t,
        #assembly_pad,
        visual_cursor_t,
        visual_cursor_fine_yxp,
        visual_cursor_coarse_yx,
        visual_cursor_pad,
        #symbolic_cursor_x,
        #symbolic_cursor_t,
        #symbolic_cursor_pad,
        auto_x,
        auto_t,
        auto_pad,
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
        if tile_x is not None and tile_x.shape[0]:
            tile_e = self.tile_embedding(tile_x)
            #tile_tpe = self.temporal_position_encoding(tile_t)
            tile_tpe = self.temporal_embedding(tile_t)
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
        #token_tpe = self.temporal_position_encoding(token_t)
        token_tpe = self.temporal_embedding(token_t)
        token_x = (
            self.token_norm(token_e) +
            self.temporal_norm(token_tpe)
        )
        xs.append(token_x)
        ts.append(token_t)
        pads.append(token_pad)
        
        ## compute the token features for the assemblies
        #if assembly_shape is not None:
        #    assembly_shape_x = self.shape_embedding(assembly_shape)
        #    assembly_color_x = self.color_embedding(assembly_color)
        #    assembly_instance_x = self.symbolic_instance_embedding(
        #        assembly_instance)
        #    assembly_pose_x = self.pose_embedding(assembly_pose)
        #    #assembly_tpe = self.temporal_position_encoding(assembly_t)
        #    assembly_tpe = self.temporal_embedding(assembly_t)
        #    
        #    assembly_x = (
        #        self.shape_norm(assembly_shape_x) +
        #        self.color_norm(assembly_color_x) +
        #        self.symbolic_instance_norm(assembly_instance_x) +
        #        self.pose_norm(assembly_pose_x) +
        #        self.temporal_norm(assembly_tpe)
        #    )
        #    xs.append(assembly_x)
        #    ts.append(assembly_t)
        #    pads.append(assembly_pad)
        
        # compute token features for the visual cursors
        if visual_cursor_t is not None:
            visual_cursor_f = self.visual_cursor_fine_embedding(
                visual_cursor_fine_yxp)
            #visual_cursor_tpe = self.temporal_position_encoding(
            #    visual_cursor_t)
            visual_cursor_tpe = self.temporal_embedding(visual_cursor_t)
            visual_cursor_spe = self.spatial_position_encoding(
                visual_cursor_coarse_yx)
            visual_cursor_x = (
                self.visual_cursor_fine_norm(visual_cursor_f) +
                self.temporal_norm(visual_cursor_tpe) +
                self.spatial_norm(visual_cursor_spe)
            )
            xs.append(visual_cursor_x)
            ts.append(visual_cursor_t)
            pads.append(visual_cursor_pad)
        
        # compute token features for the symbolic cursors
        #if symbolic_cursor_t is not None:
        #    import pdb
        #    pdb.set_trace()
        #    #symbolic_cursor_x = self.
        #    symbolic_cursor_t
        #    symbolic_cursor_pad
        
        # auto_embeddings
        for name, embedding in self.auto_embeddings.items():
            xs.append(embedding(
                **auto_x[name], t=auto_t[name], pad=auto_pad[name]
            ))
            ts.append(auto_t[name])
            pads.append(auto_pad[name])
        
        # compute token features for the readout tokens
        readout_e = self.readout_embedding(readout_x)
        #readout_tpe = self.temporal_position_encoding(readout_t)
        readout_tpe = self.temporal_embedding(readout_t)
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
