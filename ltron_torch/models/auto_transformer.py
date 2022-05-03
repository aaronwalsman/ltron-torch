import random
import os

import numpy

import torch
from torch.nn import Module, Linear, LayerNorm, Sequential, ModuleDict
from torch.distributions import Categorical

import tqdm

from splendor.image import save_image

from ltron.name_span import NameSpan
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.gym.spaces import MultiScreenPixelSpace, MaskedTiledImageSpace
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
from ltron_torch.models.coarse_to_fine_decoder import (
    CoarseToFineCursorDecoderConfig, CoarseToFineCursorDecoder)
from ltron_torch.models.padding import decat_padded_seq, get_seq_batch_indices

class AutoTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    CoarseToFineCursorDecoderConfig,
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
        for name in self.embedding.cursor_names:
            subspace = self.action_space.subspaces[name]
            coarse_cursor_keys = [
                k for k in subspace.screen_span.keys() if k != 'NO_OP']
            coarse_cursor_span = self.embedding.tile_position_layout.subspace(
                coarse_cursor_keys)
            fine_shape = self.embedding.cursor_fine_layout.get_shape(name)
            decoders[name] = CoarseToFineCursorDecoder(
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
    
    def forward_rollout(self, terminal, *args, **kwargs):
        device = next(self.parameters()).device
        use_memory = torch.BoolTensor(~terminal).to(device)
        return self(*args, **kwargs, use_memory=use_memory)
    
    def observation_to_tensors(self, batch, sequence_pad, device):
        return self.embedding.observation_to_tensors(
            batch, sequence_pad, device)
    
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
            cursor_t,
            cursor_fine_yxp,
            cursor_coarse_yx,
            cursor_pad,
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
        _, x = decat_padded_seq(x, tile_pad+token_pad+cursor_pad, readout_pad)
        
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
            if name in self.embedding.cursor_names:
                head_xs[name] = name_x
            else:
                head_xs.update(name_x)
            
        flat_x = torch.zeros(sb, self.action_space.n, device=device)
        self.action_space.ravel_subspace(head_xs, out=flat_x)
        
        out_x = torch.zeros(max_seq, b, self.action_space.n, device=device)
        out_x[readout_i, readout_b] = flat_x
        
        return out_x
    
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
        
        '''
        s,b = batch['observation']['expert'].shape[:2]
        color = batch['observation']['table']['image']
        white = (
            (color[:,:,:,:,0] == 255) &
            (color[:,:,:,:,1] == 255) &
            (color[:,:,:,:,2] == 255)
        )
        ss, bb, yy, xx = numpy.where(white)
        yyxx = yy * 256 * 2 + xx * 2 + 1
        labels = [[[] for _ in range(b)] for _ in range(s)]
        for sss, bbb, yx in zip(ss, bb, yyxx):
            a = self.action_space.ravel('pick_cursor', yx)
            labels[sss][bbb].append(a)
        
        for ss in range(s):
            for bb in range(b):
                labels[ss][bb].extend([0] * (256-len(labels[ss][bb])))
        
        #position = batch['observation']['table_color_render']
        #s, b = position.shape[:2]
        #labels = [[[] for _ in range(b)] for _ in range(s)]
        #for ss in range(s):
        #    for bb in range(b):
        #        p = position[ss,bb,0]
        #        a = self.action_space.ravel('table_viewpoint', p)
        #        labels[ss][bb].append(a)
        
        labels = torch.LongTensor(labels).to(device)
        return labels
        '''

    def observation_to_single_label(self, batch, pad, device):
        '''
        labels = batch['observation']['expert']
        s, b = labels.shape[:2]
        y = torch.zeros((s,b), dtype=torch.long)
        for ss in range(s):
            for bb in range(b):
                label = [l for l in labels[ss,bb] if l != 0]
                if len(label):
                    y[ss,bb] = random.choice(label)
        '''
        
        step = batch['observation']['step']
        s, b = step.shape
        y = torch.zeros((s,b), dtype=torch.long)
        for ss in range(s):
            for bb in range(b):
                '''
                screen = ['table','hand'][step[ss,bb] % 2]
                pick_space = self.action_space.subspaces['pick_cursor']
                yy = 48
                xx = 48
                p = 1
                pick_action = pick_space.ravel(screen, yy, xx, p)
                label = self.action_space.ravel('pick_cursor', pick_action)
                '''
                '''
                i = step[ss,bb] % 2 + 1
                label = self.action_space.ravel('table_viewpoint', i)
                '''
                
                screen = 'table'
                pick_space = self.action_space.subspaces['pick_cursor']
                yy, xx = batch['observation']['table_color_render'][ss,bb]
                p = 1
                pick_action = pick_space.ravel(screen, yy, xx, p)
                label = self.action_space.ravel('pick_cursor', pick_action)
                
                y[ss, bb] = label

        return y.to(device)
    
    def observation_to_label_distribution(self, batch, pad, device):
        # get the labels
        labels = self.observation_to_labels(batch, pad, device)
        s, b = labels.shape[:2]
        
        # pull out the non-zero values
        ss, bb, nn = torch.where(labels)
        ll = labels[ss,bb,nn]
        y = torch.zeros(
            (s,b,self.action_space.n), dtype=torch.long, device=device)
        y[ss,bb,ll] = 1
        total = torch.sum(y, dim=-1, keepdim=True)
        total[total == 0] = 1
        if torch.any(total == 0):
            print('total zero')
            import pdb
            pdb.set_trace()
        y = y/total
        
        return y
    
    def visualize_episodes(
        self,
        epoch,
        episodes,
        visualization_episodes_per_epoch,
        visualization_directory,
    ):
        num_seqs = min(visualization_episodes_per_epoch, episodes.num_seqs())
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
            
            subspace_p = self.action_space.unravel_subspace(seq_action_p)
            cursor_maps = {}
            for name, space in self.action_space.subspaces.items():
                if isinstance(space, MultiScreenPixelSpace):
                    cursor_maps[name] = space.unravel_maps(subspace_p[name])
            
            for frame_id in range(seq_len):
                frame_observation = index_hierarchy(
                    seq['observation'], frame_id)

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

                action = index_hierarchy(seq['action'], frame_id)
                action_name, action_index = self.action_space.unravel(action)
                action_subspace = self.action_space.subspaces[action_name]
                if isinstance(action_subspace, MultiScreenPixelSpace):
                    n, *yxc = action_subspace.unravel(action_index)
                    if n != 'NO_OP':
                        y, x, c = yxc
                        action_image = images[n]
                        color = self.embedding.cursor_colors[action_name][c]
                        draw_crosshairs(action_image, y, x, 5, color)
                
                for name in self.embedding.cursor_names:
                    observation_space = self.observation_space['action'][name]
                    o = frame_observation['action'][name]
                    n, y, x, p = observation_space.unravel(o)
                    color = self.embedding.cursor_colors[name][p]
                    draw_box(images[n], x-3, y-3, x+3, y+3, color)
                
                expert = frame_observation['expert']
                for e in expert:
                    expert_name, expert_index = self.action_space.unravel(e)
                    expert_subspace = self.action_space.subspaces[expert_name]
                    if isinstance(expert_subspace, MultiScreenPixelSpace):
                        n, y, x, c = expert_subspace.unravel(expert_index)
                        expert_image = images[n]
                        color = numpy.array(
                            self.embedding.cursor_colors[expert_name][c])
                        pixel_color = expert_image[y,x] * 0.5 + color * 0.5
                        expert_image[y,x] = pixel_color.astype(numpy.uint8)
                
                out_image = stack_images_horizontal(images.values(), spacing=4)
                frame_path = os.path.join(
                    seq_path,
                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                )
                save_image(out_image, frame_path)
