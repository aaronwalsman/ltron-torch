import random
import os
import json

import numpy

import torch
from torch.nn import Module, Linear, LayerNorm, Sequential, ModuleDict
from torch.distributions import Categorical

import tqdm

from splendor.image import save_image
from splendor.json_numpy import NumpyEncoder

from ltron.name_span import NameSpan
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.bricks.brick_scene import BrickScene
from ltron.gym.spaces import (
    MultiScreenPixelSpace, MaskedTiledImageSpace, AssemblySpace,
)
from ltron.visualization.drawing import (
    draw_crosshairs, draw_box, stack_images_horizontal,
)

from ltron_torch.models.mlp import linear_stack
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
    init_weights,
)
from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.cursor_decoder import (
    CoarseToFineVisualCursorDecoder,
    CoarseToFineSymbolicCursorDecoder,
)
from ltron_torch.models.padding import decat_padded_seq, get_seq_batch_indices

class AutoTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
):
    action_decoder_dropout = 0.1

class AutoTransformer(Module):
    def __init__(self,
        config,
        observation_space,
        action_space,
        checkpoint=None
    ):
        super().__init__()
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        
        # build the token embedding
        self.embedding = AutoEmbedding(config, observation_space, action_space)
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # build cursor decoders
        decoders = {}
        for name in self.embedding.visual_cursor_names:
            subspace = self.action_space.subspaces[name]
            coarse_cursor_keys = [
                k for k in subspace.screen_span.keys() if k != 'NO_OP']
            coarse_cursor_span = self.embedding.tile_position_layout.subspace(
                coarse_cursor_keys)
            fine_shape = self.embedding.cursor_fine_layout.get_shape(name)
            decoders[name] = CoarseToFineVisualCursorDecoder(
                coarse_cursor_span,
                fine_shape,
                channels=config.channels,
                nonlinearity=config.nonlinearity,
                default_k=4,
            )
        
        for name in self.embedding.symbolic_cursor_names:
            subspace = self.action_space.subspaces[name]
            coarse_cursor_span = NameSpan()
            for key in subspace.layout.keys():
                if key != 'NO_OP':
                    num_instances, num_snaps = subspace.layout.get_shape(key)
                    coarse_cursor_span.add_names(**{key:num_instances})
            fine_shape = (num_snaps,)
            decoders[name] = CoarseToFineSymbolicCursorDecoder(
                config, coarse_cursor_span, fine_shape)
        
        # build the noncursor decoder
        decoders['noncursor'] = Sequential(
            linear_stack(
                2,
                config.channels,
                nonlinearity=config.nonlinearity,
                final_nonlinearity=True,
                hidden_dropout=config.action_decoder_dropout,
                out_dropout=config.action_decoder_dropout,
            ),
            LinearMultiheadDecoder(
                config.channels,
                {name:self.action_space.subspaces[name].n
                 for name in self.embedding.noncursor_names
                }
            ),
        )
        
        self.decoders = ModuleDict(decoders)
        
        # initialize weights
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint)
        elif config.init_weights:
            self.apply(init_weights)
    
    def zero_all_memory(self):
        self.encoder.zero_all_memory()
    
    def observation_to_tensors(self, batch, sequence_pad):
        return self.embedding.observation_to_tensors(batch, sequence_pad)
    
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
        use_memory=None,
    ):
        x, t, pad = self.embedding(
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
        )
        
        # use the encoder to encode
        x = self.encoder(x, t, pad, use_memory=use_memory)[-1]
        device = x.device
        s, b, c = x.shape
        
        # extract decoder tokens
        _, x = decat_padded_seq(
            x,
            tile_pad+token_pad+visual_cursor_pad+sum(auto_pad.values()),
            readout_pad,
        )
        
        # use the decoders to decode
        max_seq = torch.max(seq_pad)
        min_t = torch.min(readout_t, dim=0).values
        
        head_xs = {}
        
        for name in self.embedding.readout_layout.keys():
            if name == 'PAD':
                continue
            readout_index = self.embedding.readout_layout.ravel(name, 0)
            readout_s, readout_b = torch.where(readout_x == readout_index)
            readout_i = (readout_t - min_t.view(1,-1))[readout_s, readout_b]
            name_x = x[readout_s, readout_b]
            sb = name_x.shape[0]
            name_x = self.decoders[name](name_x)
            if name in self.embedding.visual_cursor_names:
                head_xs[name] = name_x
            elif name in self.embedding.symbolic_cursor_names:
                head_xs[name] = name_x
            else:
                head_xs.update(name_x)
            
        flat_x = torch.zeros(sb, self.action_space.n, device=device)
        last_dim = len(flat_x.shape)-1
        self.action_space.ravel_vector(head_xs, out=flat_x, dim=last_dim)
        
        out_x = torch.zeros(max_seq, b, self.action_space.n, device=device)
        out_x[readout_i, readout_b] = flat_x
        
        return out_x
    
    def forward_rollout(self, terminal, *args, **kwargs):
        device = next(self.parameters()).device
        use_memory = torch.BoolTensor(~terminal).to(device)
        return self(*args, **kwargs, use_memory=use_memory)
    
    def tensor_to_actions(self, x, mode='sample'):
        s, b, a = x.shape
        assert s == 1
        if mode == 'sample':
            distribution = Categorical(logits=x)
            actions = distribution.sample().cpu().view(b).numpy()
        elif mode == 'max':
            actions = torch.argmax(x, dim=-1).cpu().view(b).numpy()

        return actions
    
    def tensor_to_distribution(self, x):
        s, b, a = x.shape
        assert s == 1
        distribution = Categorical(logits=x)
        
        return distribution
    
    #def observation_to_labels(self, batch, pad):
    #    device = next(self.parameters()).device
    #    return torch.LongTensor(batch['observation']['expert']).to(device)
    
    def observation_to_uniform_single_label(self, batch, pad):
        # get the labels
        labels = self.observation_to_labels(batch, pad)
        s, b = labels.shape[:2]
        
        # pull out the non-zero values
        ss, bb, nn = torch.where(labels)
        
    
    def observation_to_label(self, batch, pad, supervision_mode):
        device = next(self.parameters()).device
        
        # get the labels
        #labels = self.observation_to_labels(batch, pad)
        #s, b = labels.shape[:2]
        
        # build the label tensor
        #y = torch.zeros((s,b,self.action_space.n), device=device)
        
        if supervision_mode == 'action':
            # pull the actions
            actions = torch.LongTensor(batch['action']).to(device)
            s, b = actions.shape[:2]
            
            # build the y tensor
            y = torch.zeros((s, b, self.action_space.n), device=device)
            ss = torch.arange(s).view(s,1).expand(s,b).reshape(-1).cuda()
            bb = torch.arange(b).view(1,b).expand(s,b).reshape(-1).cuda()
            y[ss,bb,actions.view(-1)] = 1
        
        elif supervision_mode == 'expert_uniform_distribution':
            # pull the expert labels
            labels = torch.LongTensor(batch['observation']['expert']).to(device)
            s, b = labels.shape[:2]
            
            # build the y tensor
            y = torch.zeros((s,b,self.action_space.n), device=device)
            ss, bb, nn = torch.where(labels)
            ll = labels[ss,bb,nn]
            y[ss,bb,ll] = 1
            total = torch.sum(y, dim=-1, keepdim=True)
            total[total == 0] = 1 # (handle the padded values)
            y = y/total
        
        elif supervision_mode == 'expert_uniform_sample':
            # pull the expert labels
            labels = torch.LongTensor(batch['observation']['expert']).to(device)
            s, b = labels.shape[:2]
            
            # build the y tensor
            y = torch.zeros((s,b,self.action_space.n), device=device)
            ll = labels[:,:,0]
            ss = torch.arange(s).view(s,1).expand(s,b).reshape(-1).cuda()
            bb = torch.arange(b).view(1,b).expand(s,b).reshape(-1).cuda()
            y[ss,bb,ll.view(-1)] = 1
        
        return y
    
    def visualize_episodes(
        self,
        epoch,
        episodes,
        visualization_episodes_per_epoch,
        visualization_directory,
    ):
        # iterate through each episode
        #for seq_id, batch in enumerate(tqdm.tqdm(episodes)):
        #for seq_id in tqdm.tqdm(range(visualization_episodes_per_epoch)):
        for seq_id, batch in tqdm.tqdm(zip(
            range(visualization_episodes_per_epoch), episodes),
            total=visualization_episodes_per_epoch
        ):
            
            # get the sequence length
            seq, pad = batch
            seq_len = len_hierarchy(seq)
            
            # make the sequence directory
            seq_path = os.path.join(
                visualization_directory, 'seq_%06i'%seq_id)
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            
            # build the summary
            summary = {
                'action':[],
                'reward':seq['reward'],
            }
            
            # compute action probabilities
            seq_action_p = numpy.exp(numpy.clip(seq['activations'],-50,50))[:,0]
            seq_action_n = numpy.sum(seq_action_p, axis=-1)
            seq_action_p = seq_action_p / seq_action_n.reshape(-1,1)
            max_action_p = numpy.max(seq_action_p, axis=-1)
            
            # get the cursor prediction maps
            # TODO: Fix this using unravel_vector now
            #subspace_p = self.action_space.unravel_subspace(seq_action_p)
            #cursor_maps = {}
            #for name, space in self.action_space.subspaces.items():
            #    if isinstance(space, MultiScreenPixelSpace):
            #        cursor_maps[name] = space.unravel_maps(subspace_p[name])
            
            # build the frames
            for frame_id in range(seq_len):
                frame_observation = index_hierarchy(index_hierarchy(
                    seq['observation'], frame_id), 0)
                
                # add the action to the summary
                summary['action'].append(
                    self.action_space.unravel(seq['action'][frame_id,0]))
                
                images = {}
                for name, space in self.observation_space.items():
                    if isinstance(space, MaskedTiledImageSpace):
                        image = frame_observation[name]['image']
                        for map_name, maps in cursor_maps.items():
                            if name in maps:
                                colors = self.embedding.cursor_colors[map_name]
                                m = maps[name][frame_id]
                                for c in range(m.shape[-1]):
                                    mc = m[..., c] / max_action_p[frame_id]
                                    mc = mc.reshape(*mc.shape, 1)
                                    image = image * (1.-mc) + mc*colors[c]

                        images[name] = image.astype(numpy.uint8)
                    
                    elif isinstance(space, AssemblySpace):
                        assembly_scene = BrickScene()
                        assembly_scene.set_assembly(
                            frame_observation[name],
                            space.shape_ids,
                            space.color_ids,
                        )
                        assembly_path = os.path.join(
                            seq_path,
                            '%s_%04i_%04i.mpd'%(name, seq_id, frame_id)
                        )
                        assembly_scene.export_ldraw(assembly_path)
                
                action = index_hierarchy(seq['action'], frame_id)
                action_name, *nyxc = self.action_space.unravel(action)
                action_subspace = self.action_space.subspaces[action_name]
                if isinstance(action_subspace, MultiScreenPixelSpace):
                    #n, *yxc = action_subspace.unravel(action_index)
                    n, *yxc = nyxc
                    if n != 'NO_OP':
                        y, x, c = yxc
                        action_image = images[n]
                        color = self.embedding.cursor_colors[action_name][c]
                        draw_crosshairs(action_image, y, x, 5, color)
                
                for name in self.embedding.visual_cursor_names:
                    observation_space = self.observation_space['action'][name]
                    o = frame_observation['action'][name]
                    n, y, x, p = observation_space.unravel(o)
                    color = self.embedding.cursor_colors[name][p]
                    draw_box(images[n], x-3, y-3, x+3, y+3, color)
                
                expert = frame_observation['expert']
                for e in expert:
                    expert_name, *nyxc = self.action_space.unravel(e)
                    expert_subspace = self.action_space.subspaces[expert_name]
                    if isinstance(expert_subspace, MultiScreenPixelSpace):
                        #n, y, x, c = expert_subspace.unravel(expert_index)
                        n, y, x, c = nyxc
                        expert_image = images[n]
                        color = numpy.array(
                            self.embedding.cursor_colors[expert_name][c])
                        pixel_color = expert_image[y,x] * 0.5 + color * 0.5
                        expert_image[y,x] = pixel_color.astype(numpy.uint8)
                
                if images:
                    out_image = stack_images_horizontal(
                        images.values(), spacing=4)
                    frame_path = os.path.join(
                        seq_path,
                        'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                    )
                    save_image(out_image, frame_path)
            
            summary_path = os.path.join(seq_path, 'summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, cls=NumpyEncoder)
