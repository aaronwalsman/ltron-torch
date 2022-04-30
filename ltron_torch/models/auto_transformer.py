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
from ltron.visualization.drawing import draw_crosshairs, stack_images_horizontal

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
        #interface,
        observation_space,
        action_space,
        checkpoint=None
    ):
        super().__init__()
        self.config = config
        #self.interface = interface
        self.observation_space = observation_space
        self.action_space = action_space
        
        # build the token embedding
        self.embedding = AutoEmbedding(config, observation_space, action_space)
        '''
        self.embedding = AutoEmbedding(
            config,
            interface.tile_shape,
            interface.token_vocabulary,
            interface.tile_spatial_positions,
            interface.max_time_steps,
        )
        '''
        
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
        
        '''
        # build the cursor decoders
        decoders = {}
        self.decoder_data = {}
        self.fine_shape = (*interface.tile_shape[:2], 2)
        for name, cursor_data in interface.cursor_screen_data.items():
            tile_h, tile_w = interface.tile_shape[:2]
            total_coarse_positions = 0
            self.decoder_data[name] = {}
            coarse_shapes = {}
            for screen_name, screen_data in cursor_data.items():
                screen_h, screen_w, screen_c = screen_data['shape']
                assert screen_h % tile_h == 0
                assert screen_w % tile_w == 0
                coarse_h = screen_h // tile_h
                coarse_w = screen_w // tile_w
                screen_coarse_positions = coarse_h * coarse_w
                self.decoder_data[name][screen_name] = {
                    'start':total_coarse_positions,
                    'end':total_coarse_positions+screen_coarse_positions,
                    #'coarse_shape':(coarse_h, coarse_w),
                    #'fine_shape':(tile_h, tile_w, screen_c),
                }
                coarse_shapes[screen_name] = (coarse_h, coarse_w)
                total_coarse_positions += screen_coarse_positions
            
            fine_positions = tile_h * tile_w * screen_c
            
            decoders[name] = CoarseToFineCursorDecoder(
                config, coarse_shapes, fine_positions)
        '''
        '''
        for name, cursor_screen_data in interface.cursor_screen_data.items():
            for screen_name, screen_data in cursor_screen_data.items():
                cursor_decoder = CoarseToFineCursorDecoder(
                    config,
                    *screen_data['shape'],
                    *interface.tile_shape[:2],
                    2,
                )
                decoders['%s,%s'%(name, screen_name)] = cursor_decoder
        '''
        '''
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
                {name:(b-a)
                 for name, (a, b) in interface.noncursor_name_range.items()
                }
            ),
        )
        
        self.decoders = ModuleDict(decoders)
        '''
        
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
            
            '''
            for head_name, head_x in name_x.items():
                sb, *other_dims = head_x.shape
                padded_x = torch.zeros(
                    max_seq,
                    b,
                    *other_dims,
                    dtype=head_x.dtype,
                    device=tile_x.device,
                )
                padded_x[readout_i, readout_b] = head_x
                head_xs[head_name] = padded_x
            '''
            
            '''
            for head_name, head_x in r_x.items():
                #start, end = self.interface.name_range[head_name]
                # if this is a cursor, we need to unpack a bit
                if name != 'noncursor':
                    maps = {}
                    total_elements = 0
                    import pdb
                    pdb.set_trace()
                    for screen_name, screen_x in head_x.items():
                        sb, ch, cw = screen_x.shape[:3]
                        
                        fh, fw, fc = self.fine_shape
                        screen_x = screen_x.view(sb, ch, cw, fh, fw, fc)
                        screen_x = screen_x.permute(0,1,3,2,4,5)
                        screen_x = screen_x.reshape(sb, ch*fh, cw*fw, fc)
                        maps[screen_name] = screen_x
                        total_elements += ch*fh*cw*fw*fc
                    
                    head_x = torch.zeros(
                        sb, total_elements+1, device=screen_x.device)
                    cursor_space = self.action_space.subspaces[head_name]
                    cursor_space.ravel_maps(maps, out=head_x)
                    head_x = head_x[:,1:]
                
                head_xs[head_name] = head_x
            '''
        
        flat_x = torch.zeros(sb, self.action_space.n, device=device)
        self.action_space.ravel_subspace(head_xs, out=flat_x)
        
        out_x = torch.zeros(max_seq, b, self.action_space.n, device=device)
        out_x[readout_i, readout_b] = flat_x
        
        '''
        import pdb
        pdb.set_trace()
        
        action_x = torch.zeros(
            sb,
            self.action_space.subspace_span.total,
            device=tile_x.device,
        )
        self.action_space.ravel_subspace(head_xs, out=action_x)
        
        padded_action_x = torch.zeros(
            max_seq,
            b,
            self.action_space.subspace_span.total,
            device=tile_x.device,
        )
        padded_action_x[readout_t, readout_b] = action_x
        '''
        #return head_xs
        return out_x
    
    def tensor_to_actions(self, x, mode='sample'):
        '''
        coarse_namespan = NameSpan()
        course_x = []
        for name in self.embedding.readout_layout.keys():
            if name in self.embedding.cursor_fine_layout.keys():
                s,b,*shape = x[name]['coarse_x'].shape
                coarse_namespan.add_names(**{name:shape})
                course_x.append(x[name]['coarse_x'].view(s,b,-1))
            else:
                for key in x[name]:
                    s,b,*shape = x[name][key].shape
                    coarse_namespan.add_names(**{key:shape})
                    course_x.append(x[name][key].view(s,b,-1))
        coarse_x = torch.cat(course_x, dim=-1)
        if mode == 'sample':
            distribution = Categorical(coarse_x)
            coarse_actions = distribution.sample().cpu().numpy()
        elif mode == 'argmax':
            coarse_actions = torch.argmax(coarse_x, dim=-1)
        
        fine_actions = []
        for a in coarse_actions:
            name, i = coarse_namespan.unravel(a)
            if name in self.embedding.cursor_fine_layout.keys():
                screen, cy, cx = self.decoder[name].coarse_span.unravel(a)
                if mode == 'sample':
                    
                    distribution = Categorical(x[name]['fine_x'])
            else:
                fine_action = self.action_space.ravel(name, i)
            fine_actions.append(fine_action)
        '''
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
                    #a = action_space.extract_subspace(seq_action_p, name)
                    #cursor_space = self.action_space.subspaces[name]
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
                    #n, y, x, c = action_subspace.unravel(action_index)
                    n, *yxc = action_subspace.unravel(action_index)
                    if n != 'NO_OP':
                        y, x, c = yxc
                        action_image = images[n]
                        color = self.embedding.cursor_colors[action_name][c]
                        #action_image[y,x] = color
                        draw_crosshairs(action_image, y, x, 5, color)

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
