import random
import copy
from collections import OrderedDict
import os

import numpy

import torch
from torch.distributions import Categorical

import tqdm

from gym.spaces import Discrete

from splendor.image import save_image

from ltron.config import Config
from ltron.compression import batch_deduplicate_from_masks
from ltron.visualization.drawing import stack_images_horizontal, draw_crosshairs
from ltron.gym.spaces import (
    MaskedTiledImageSpace,
    PhaseSpace,
    TimeStepSpace,
    MultiScreenPixelSpace,
    ActionModeSelectorSpace,
)
from ltron.hierarchy import len_hierarchy, index_hierarchy

from ltron_torch.gym_tensor import default_tile_transform
from ltron_torch.models.padding import cat_multi_padded_seqs

visualization_colors = [
    [  0,   0, 255],
    [255, 255,   0],
    [  0, 255,   0],
    [255,   0, 255],
    [255,   0,   0],
    [  0, 255, 255],
]

class AutoTransformerInterfaceConfig(Config):
    pass

class AutoTransformerInterface:
    def __init__(self, config, env):
        
        self.config = config
        self.env = env
        self.no_op_action = env.metadata['no_op_action']
        
        self.num_readout_tokens = 0
        # token_vocabulary starts at 1 so that 0 padded elements don't show up
        # as readout tokens
        self.token_vocabulary = 1
        self.tile_spatial_positions = 0
        self.tile_sources = {}
        self.observation_token_offsets = {}
        self.discrete_action_order = []
        self.readout_token_indices = {}
        self.time_step_space = None
        self.cursor_observation_offsets = {}
        
        # pull tile and token information from the observation space
        tile_shapes = set()
        for name, space in self.env.metadata['observation_space'].items():
            if isinstance(space, MaskedTiledImageSpace):
                self.tile_shape = (
                    space.tile_height, space.tile_width, space.channels)
                tile_shapes.add(self.tile_shape)
                assert len(tile_shapes) == 1
                
                self.tile_sources[name] = {
                    'mask_shape':(space.mask_height, space.mask_width),
                    'offset':self.tile_spatial_positions,
                }
                self.tile_spatial_positions += (
                    space.mask_height * space.mask_width)
            
            #if isinstance(space, PhaseSpace):
            #    num_tokens = space.n
            #    self.observation_token_offsets[name] = self.token_vocabulary
            #    self.token_vocabulary += num_tokens
            
            if isinstance(space, TimeStepSpace):
                assert self.time_step_space is None
                self.time_step_space = name
                self.max_time_steps = space.max_steps
            
            #if isinstance(space, MultiScreenPixelSpace):
            #    self.cursor_observation_offsets[name] = {
            #        'offset':self.token_vocabulary,
            #    }
            #    import pdb
            #    pdb.set_trace()
            #    #cursor_positions_per_tile = 
            #    self.token_vocabulary += cursor_positions_per_tile
        
        assert self.time_step_space is not None
        self.tile_spatial_positions += 1
        
        # pull cursor and discrete action information from the action space
        action_space = self.env.metadata['action_space']
        self.noncursor_actions = 0
        #self.noncursor_name_action_to_chain = {}
        self.noncursor_name_range = {}
        self.name_range = action_space.name_range
        self.cursor_screen_data = {}
        self.cursor_colors = {}
        self.cursor_color_n = 0
        self.num_actions = action_space.n
        for name, space in action_space.subspaces.items():
            if isinstance(space, MultiScreenPixelSpace):
                self.cursor_screen_data[name] = space.screen_data
                #for screen_name, screen_data in space.screen_data.items():
                #    name_screen = '%s,%s'%(name, screen_name)
                #    self.readout_token_indices[name_screen] = (
                #        self.token_vocabulary)
                #    self.token_vocabulary += 1
                self.readout_token_indices[name] = self.token_vocabulary
                self.token_vocabulary += 1
                
                self.cursor_colors[name] = []
                for c in range(space.channels):
                    self.cursor_colors[name].append(visualization_colors[
                            self.cursor_color_n % len(visualization_colors)])
                    self.cursor_color_n += 1
                
                ##################################
                first_key = next(iter(space.screen_data.keys()))
                first_source_key = first_key + '_tile_mask' # gross!
                mh, mw = self.tile_sources[first_source_key]['mask_shape']
                h,w,c = space.screen_data[first_key]['shape']
                assert h % mh == 0
                assert w % mw == 0
                fh = h//mh
                fw = w//mw
                cursor_positions_per_tile = fh*fw*c
                self.cursor_observation_offsets[name] = {
                    'offset':self.token_vocabulary,
                    'fine_shape':(fh,fw,c),
                    'fine_positions':cursor_positions_per_tile
                }
                self.token_vocabulary += cursor_positions_per_tile
            
            else:
                self.noncursor_name_range[name] = action_space.name_range[name]
                #self.noncursor_name_action_to_chain[name] = {}
                #for i, c in action_space.name_action_to_chain[name].items():
                #    self.noncursor_name_action_to_chain[name] = c
                #    self.noncursor_actions += 1
        self.readout_token_indices['noncursor'] = self.token_vocabulary
        self.token_vocabulary += 1
        '''
        self.total_actions = 1
        self.discrete_actions = 1
        self.action_lookup = {0:(None, None)}
        self.cursor_screen_data = {}
        for name, space in env.metadata['action_space'].items():
            if isinstance(space, Discrete):
                for i in range(space.n):
                    if i != self.no_op_action[name]:
                        self.action_lookup[self.total_actions] = name, i
                        self.total_actions += 1
            else:
                raise Exception('Unsupported action space for %s'%name)
            
            if isinstance(space, MultiScreenPixelSpace):
                self.cursor_screen_data[name] = space.screen_data
                for screen_name, screen_data in space.screen_data.items():
                    name_screen = '%s,%s'%(name, screen_name)
                    self.discrete_action_order.append(name_screen)
                    self.readout_token_indices[name_screen] = (
                        self.token_vocabulary)
                    self.token_vocabulary += 1
            else:
                self.discrete_actions += space.n
        
        self.discrete_action_order.append('discrete_actions')
        self.readout_token_indices['discrete_actions'] = self.token_vocabulary
        self.token_vocabulary += 1
        '''
        #self.token_sources['discrete_actions'] = {
        #    'vocabulary_offset':self.token_vocabulary,
        #    'type':'readout',
        #    'tokens_per_step':1,
        #}
        #self.token_vocabulary += 1
        
        # find the cursors
        #self.cursor_screen_data = {}
        #for name, space in env.metadata['action_space'].items():
            #if isinstance(space, MultiScreenPixelSpace):
                #for screen_name, screen_data in space.screen_data.items():
                #    self.token_sources[name, screen_name] = {
                #        'vocabulary_offset':self.token_vocabulary,
                #        'type':'readout',
                #        'tokens_per_step':1,
                #    }
                #    self.token_vocabulary += 1
                #self.cursor_screen_data[name] = space.screen_data
    
    def observation_to_tensors(self, batch, pad, device):
        
        observation = batch['observation']
        x = {}
        
        time_step = observation[self.time_step_space]
        s, b = time_step.shape[:2]
        
        # make the tiles
        tile_x = []
        tile_tyx = []
        tile_pad = []
        for name, source in self.tile_sources.items():
            source_tiles, source_tyx, source_pad = batch_deduplicate_from_masks(
                observation[name]['image'],
                observation[name]['tile_mask'],
                time_step,
                pad,
                flatten_hw=True,
            )
            
            source_tiles = default_tile_transform(source_tiles)
            source_tiles = source_tiles.to(device).contiguous()
            
            offset = source['offset']
            source_tyx[:,:,1] += offset
            source_tyx = torch.LongTensor(source_tyx).to(device)
            
            source_pad = torch.LongTensor(source_pad).to(device)
            
            tile_x.append(source_tiles)
            tile_tyx.append(source_tyx)
            tile_pad.append(source_pad)
        
        tile_x, cat_tile_pad = cat_multi_padded_seqs(tile_x, tile_pad)
        tile_tyx, _ = cat_multi_padded_seqs(tile_tyx, tile_pad)
        tile_t = tile_tyx[...,0]
        tile_yx = tile_tyx[...,1]
        tile_pad = cat_tile_pad
        
        # move pad to torch/device
        pad = torch.LongTensor(pad).to(device)
        
        # make the tokens
        token_x = []
        token_tyx = []
        token_pad = []
        for name, index in self.readout_token_indices.items():
            token_x.append(torch.full(
                time_step.shape, index, dtype=torch.long, device=device))
            
            #token_t.append(torch.LongTensor(time_step).to(device))
            source_tyx = torch.zeros((*time_step.shape, 2), dtype=torch.long)
            source_tyx[:,:,0] = torch.LongTensor(time_step)
            source_tyx[:,:,1] = self.tile_spatial_positions-1
            token_tyx.append(source_tyx.to(device))
            
            token_pad.append(pad)
        
        for name, offset in self.observation_token_offsets.items():
            token_x.append(
                torch.LongTensor(observation[name] + offset).to(device))
            
            #token_t.append(torch.LongTensor(time_step).to(device))
            source_tyx = torch.zeros((*time_step.shape, 2), dtype=torch.long)
            source_tyx[:,:,0] = torch.LongTensor(time_step)
            source_tyx[:,:,1] = self.tile_spatial_positions-1
            token_tyx.append(source_tyx.to(device))
            
            token_pad.append(pad)
        
        '''
        for name, cursor_offset in self.cursor_observation_offsets.items():
            cursor_observations = observation['action'][name]
            subspace = self.env.metadata['observation_space']['action'][name]
            source_x = torch.zeros_like(cursor_observations)
            source_tyx = 
            s, b = source_x.shape
            unraveled_cursor_observations = [
                subspace.unravel_index(a) for a in cursor_actions.reshape(-1)]
            for ss in range(s):
                for bb in range(b):
                    n, y, x, c = subspace.unravel_index(
                        cursor_observations[ss,bb])
                    fh, fy, fc = cursor_offset['fine_shape']
                    fy = y % fh
                    fx = x % fy
            import pdb
            pdb.set_trace()
        '''
        
        token_x, cat_token_pad = cat_multi_padded_seqs(token_x, token_pad)
        token_tyx, _ = cat_multi_padded_seqs(token_tyx, token_pad)
        token_t = token_tyx[...,0]
        token_yx = token_tyx[...,1]
        token_pad = cat_token_pad
        
        return {
            'tile_x':tile_x,
            'tile_t':tile_t,
            'tile_yx':tile_yx,
            'tile_pad':tile_pad,
            'token_x':token_x,
            'token_t':token_t,
            'token_yx':token_yx,
            'token_pad':token_pad,
            'seq_pad':pad
        }
    
    def tensor_to_actions(self, x, mode='sample'):
        s, b, a = x.shape
        assert s == 1
        if mode == 'sample':
            distribution = Categorical(logits=x)
            actions = distribution.sample().cpu().view(b).numpy()
        elif mode == 'max':
            actions = torch.argmax(x, dim=-1).cpu().view(b).numpy()
        
        return actions
    
    def observation_to_labels(self, batch, pad, device):
        return torch.LongTensor(batch['observation']['expert']).to(device)
    
    def observation_to_single_label(self, batch, pad, device):
        labels = batch['observation']['expert']
        s, b = labels.shape[:2]
        y = torch.zeros((s,b), dtype=torch.long)
        for ss in range(s):
            for bb in range(b):
                label = [l for l in labels[ss,bb] if l != 0]
                if len(label):
                    y[ss,bb] = random.choice(label)
        
        return y.to(device)
    
    def visualize_episodes(
        self,
        epoch,
        episodes,
        visualization_episodes_per_epoch,
        visualization_directory,
    ):
        num_seqs = min(visualization_episodes_per_epoch, episodes.num_seqs())
        action_space = self.env.metadata['action_space']
        observation_space = self.env.metadata['observation_space']
        for seq_id in tqdm.tqdm(range(num_seqs)):
            seq_path = os.path.join(
                visualization_directory, 'seq_%06i'%seq_id)
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            
            seq = episodes.get_seq(seq_id)
            seq_len = len_hierarchy(seq)
            
            seq_action_p = numpy.exp(numpy.clip(seq['activations'],-50,50))
            seq_action_p = (
                seq_action_p / numpy.sum(seq_action_p, axis=-1).reshape(-1,1))
            max_action_p = numpy.max(seq_action_p, axis=-1)
            
            subspace_p = action_space.unravel_subspace(seq_action_p)
            cursor_maps = {}
            for name, space in action_space.subspaces.items():
                if isinstance(space, MultiScreenPixelSpace):
                    #a = action_space.extract_subspace(seq_action_p, name)
                    cursor_space = action_space.subspaces[name]
                    cursor_maps[name] = cursor_space.unravel_maps(
                        subspace_p[name])
            
            for frame_id in range(seq_len):
                frame_observation = index_hierarchy(
                    seq['observation'], frame_id)
                
                images = {}
                for name, space in observation_space.items():
                    if isinstance(space, MaskedTiledImageSpace):
                        image = frame_observation[name]['image']
                        for map_name, maps in cursor_maps.items():
                            # gross noodles...
                            short_name = name.replace('_tile_mask', '')
                            if short_name in maps:
                                colors = self.cursor_colors[map_name]
                                m = maps[short_name][frame_id]
                                for c in range(m.shape[-1]):
                                    mc = m[..., c] / max_action_p[frame_id]
                                    mc = mc.reshape(*mc.shape, 1)
                                    image = image * (1.-mc) + mc*colors[c]
                        
                        images[name] = image.astype(numpy.uint8)
                
                action = index_hierarchy(seq['action'], frame_id)
                action_name, action_index = action_space.unravel_index(action)
                action_subspace = action_space.subspaces[action_name]
                if isinstance(action_subspace, MultiScreenPixelSpace):
                    n, y, x, c = action_subspace.unravel_index(action_index)
                    action_image = images['%s_tile_mask'%n] # gross!
                    color = self.cursor_colors[action_name][c]
                    #action_image[y,x] = color
                    draw_crosshairs(action_image, y, x, 5, color)
                
                expert = frame_observation['expert']
                for e in expert:
                    expert_name, expert_index = action_space.unravel_index(e)
                    expert_subspace = action_space.subspaces[expert_name]
                    if isinstance(expert_subspace, MultiScreenPixelSpace):
                        n, y, x, c = expert_subspace.unravel_index(expert_index)
                        expert_image = images['%s_tile_mask'%n] # gross!
                        color = numpy.array(self.cursor_colors[expert_name][c])
                        pixel_color = expert_image[y,x] * 0.5 + color * 0.5
                        expert_image[y,x] = pixel_color.astype(numpy.uint8)
                
                out_image = stack_images_horizontal(images.values(), spacing=4)
                frame_path = os.path.join(
                    seq_path,
                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                )
                save_image(out_image, frame_path)
