import os
import json


import numpy

import torch
import torch.nn as nn

from avarice.data.numpy_torch import torch_to_numpy

from ltron.constants import (
    MAX_SNAPS_PER_BRICK, NUM_SHAPE_CLASSES, NUM_COLOR_CLASSES)
from ltron.visualization.drawing import (
    write_text,
    draw_crosshairs,
    draw_square,
    draw_line,
    heatmap_overlay,
    stack_images_horizontal,
)
from ltron.gym.components import ViewpointActions

from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.transformer import (
    TransformerConfig, Transformer, init_weights)
from ltron_torch.models.cursor_decoder import CursorDecoder
from ltron_torch.models.insert_decoder import InsertDecoder
from ltron_torch.models.decoder import (
    DecoderConfig,
    DiscreteDecoder,
    CriticDecoder,
    ConstantDecoder,
)
from ltron_torch.models.equivalence import (
    equivalent_probs_categorical,
    equivalent_outcome_categorical
)
from avarice.model.parameter import NoWeightDecayParameter

class LtronVisualTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    DecoderConfig,
):
    embedding_dropout = 0.1
    strict_load = True
    
    dense_decoder_mode = 'dpt'
    dpt_blocks = [2,5,8,11]
    dpt_channels = 256

class LtronVisualTransformer(nn.Module):
    def __init__(self,
        config,
        observation_space,
        action_space,
        checkpoint=None,
    ):
        # Module super
        super().__init__()
        
        # save config, observation and action space
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        
        # build the embeddings
        self.image_embedding = AutoEmbedding(
            config, observation_space['image'])
        if 'target_image' in set(observation_space.keys()):
            self.target_image_embedding = AutoEmbedding(
                config, observation_space['target_image'])
        else:
            if 'assembly' in set(observation_space.keys()):
                self.assembly_embedding = AutoEmbedding(
                    config, observation_space['assembly'])
            if 'target_assembly' in set(observation_space.keys()):
                self.target_assembly_embedding = AutoEmbedding(
                    config, observation_space['target_assembly'])
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        if 'action_primitives' in set(observation_space.keys()):
            if 'phase' in set(observation_space['action_primitives'].keys()):
                self.phase_embedding = AutoEmbedding(
                    config, observation_space['action_primitives']['phase'])
        
        #self.pre_encoder_norm = nn.LayerNorm(config.channels)
        
        # build the decoder token
        # adding two randns here to simulate the embedding itself plus
        # a learned positional encoding
        self.decoder_token = NoWeightDecayParameter(
            torch.randn(1, 1, config.channels) +
            torch.randn(1, 1, config.channels))
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # pre decoder layernorm
        self.predecoder_norm = nn.LayerNorm(config.channels)
        
        # build the mode decoder
        self.mode_decoder = DiscreteDecoder(
            config, action_space['action_primitives']['mode'].n)
        self.mode_decoder_names = (
            action_space['action_primitives']['mode'].names)
        
        # build the primitive decoders
        primitive_decoders = {}
        for name, subspace in action_space['action_primitives'].items():
            if name == 'mode' or name not in self.mode_decoder_names:
                continue
            elif name == 'insert':
                primitive_decoders[name] = InsertDecoder(config)
            elif name in (
                'pick_and_place', 'remove', 'done', 'assemble_step', 'phase',
            ):
                primitive_decoders[name] = ConstantDecoder(config, 1)
            elif name in ('viewpoint', 'rotate', 'translate'):
                primitive_decoders[name] = DiscreteDecoder(config, subspace.n)
                    #config, subspace.n-1, sample_offset=1)
        
        self.primitive_decoders = nn.ModuleDict(primitive_decoders)
        
        # build the cursor decoders
        #self.image_norm = nn.LayerNorm(config.channels)
        self.cursor_decoder = CursorDecoder(config)
        self.critic_decoder = CriticDecoder(config)
        
        # load checkpoint or initialize weights
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint, strict=config.strict_load)
        elif config.init_weights:
            self.apply(init_weights)
        
        self.forward_passes = 0
    
    def observation_to_kwargs(self, observation, info, done, model_output):
        device = next(iter(self.parameters())).device
        
        kwargs = {}
        
        kwargs['image_kwargs'] = self.image_embedding.observation_to_kwargs(
            observation['image'], info, done, model_output)
        
        if 'target_image' in observation:
            kwargs['target_image_kwargs'] = (
                self.target_image_embedding.observation_to_kwargs(
                    observation['target_image'], info, done, model_output))
        else:
            if 'assembly' in observation:
                kwargs['assembly_kwargs'] = (
                    self.assembly_embedding.observation_to_kwargs(
                        observation['assembly'], info, done, model_output))
            
            if 'target_assembly' in observation:
                embedding = self.target_assembly_embedding
                kwargs['target_assembly_kwargs'] = (
                    embedding.observation_to_kwargs(
                        observation['target_assembly'],
                        info, done, model_output))
        
        if 'pos_snap_render' in observation:
            pos_snaps = observation['pos_snap_render']
            kwargs['pos_snap_eq'] = torch.LongTensor(
                cursor_islands(pos_snaps)).to(device)
            
            neg_snaps = observation['neg_snap_render']
            kwargs['neg_snap_eq'] = torch.LongTensor(
                cursor_islands(neg_snaps)).to(device)
        
        # shape and color equivalence
        #if self.config.shape_class_labels is None:
        #num_shape_classes = NUM_SHAPE_CLASSES
        #else:
        #    num_shape_classes = len(self.config.shape_class_labels)+1
        #if self.config.color_class_labels is None:
        #num_color_classes = NUM_COLOR_CLASSES
        #else:
        #    num_color_classes = len(self.config.color_class_labels)+1
        
        if 'action_primitives' in observation:
            if 'phase' in observation['action_primitives']:
                phase = observation['action_primitives']['phase']
                kwargs['phase_kwargs'] = (
                    self.phase_embedding.observation_to_kwargs(
                        phase, info, done, model_output)
                )
        
        if ('target_assembly' in observation and 
            'insert' in self.primitive_decoders
        ):
            '''
            shapes = observation['target_assembly']['shape']
            colors = observation['target_assembly']['color']
            b, s = shapes.shape
            active_b, active_s = numpy.where(shapes)
            compact_shapes = shapes[active_b, active_s]
            _, compact_shapes_unique = numpy.unique(
                compact_shapes, return_inverse=True)
            zero_index = numpy.max(compact_shapes_unique) + 2
            shape_islands = numpy.zeros(
                (b, NUM_SHAPE_CLASSES), dtype=numpy.int64)
            shape_islands[active_b, compact_shapes] = compact_shapes_unique + 1
            shape_islands[:,0] = zero_index
            kwargs['shape_eq'] = torch.LongTensor(shape_islands).to(device)
            
            compact_colors = colors[active_b, active_s]
            _, compact_colors_unique = numpy.unique(
                compact_colors, return_inverse=True)
            zero_index = numpy.max(compact_colors_unique) + 2
            color_islands = numpy.zeros(
                (b, NUM_COLOR_CLASSES), dtype=numpy.int64)
            color_islands[active_b, compact_colors] = compact_colors_unique + 1
            color_islands[:,0] = zero_index
            kwargs['color_eq'] = torch.LongTensor(color_islands).to(device)
            '''
        
        return kwargs
       
    def forward(self,
        image_kwargs,
        target_image_kwargs=None,
        assembly_kwargs=None,
        target_assembly_kwargs=None,
        phase_kwargs=None,
        pos_snap_eq=None,
        neg_snap_eq=None,
        shape_eq=None,
        color_eq=None,
        sample=None,
        sample_max=False,
    ):
        
        # use the embedding to compute the tokens
        x = self.image_embedding(**image_kwargs)
        h = self.config.image_height // self.config.tile_height
        w = self.config.image_width // self.config.tile_width
        hw,b,c = x.shape
        
        if target_image_kwargs is not None:
            target_image_x = self.target_image_embedding(**target_image_kwargs)
            x = torch.cat((x, target_image_x), dim=0)
        
        if assembly_kwargs is not None:
            assembly_x = self.assembly_embedding(**assembly_kwargs)
            x = torch.cat((x, assembly_x), dim=0)
        
        if target_assembly_kwargs is not None:
            target_assembly_x = self.target_assembly_embedding(
                **target_assembly_kwargs)
            x = torch.cat((x, target_assembly_x), dim=0)
        
        # phase token
        if phase_kwargs is not None:
            phase_x = self.phase_embedding(**phase_kwargs)
            b,c = phase_x.shape
            x = torch.cat((x, phase_x.view(1,b,c)), dim=0)
        
        # decoder token
        decoder_token = self.decoder_token.expand(1,b,c)
        x = torch.cat((decoder_token, x), dim=0)
        
        # normalize the embedding
        # doing this in each embedding now in order to normalize
        # similar populations together
        #x = self.pre_encoder_norm(x)
        
        # embedding dropout
        x = self.embedding_dropout(x)
        
        # push the concatenated tokens through the transformer
        if self.config.dense_decoder_mode in ('dpt', 'dpt_sum'):
            output_layers = self.config.dpt_blocks
        elif self.config.dense_decoder_mode == 'simple':
            output_layers = None
        else:
            raise ValueError(
                'config.dense_decoder_mode "%s" should be (dpt,dpt_sum,simple)')
        x = self.encoder(x, output_layers=output_layers)
        decode_x = self.predecoder_norm(x[-1][0])
        if self.config.dense_decoder_mode in ('dpt', 'dpt_sum'):
            image_x = [x[i][1:hw+1] for i in self.config.dpt_blocks]
        elif self.config.dense_decoder_mode == 'simple':
            image_x = x[-1][1:hw+1]
        
        value = self.critic_decoder(decode_x)
        
        #decode_x = output['decode_x']
        #image_x = output['image_x']
        
        out_sample = {}
        log_prob = 0.
        entropy = 0.
        logits = {}
        
        # mode
        mode_sample = None if sample is None else (
            sample['action_primitives']['mode'])
        mode_sample, lp, e, x, m_logits = self.mode_decoder(
            decode_x, sample=mode_sample, sample_max=sample_max)
        out_sample['action_primitives'] = {'mode':mode_sample}
        log_prob = log_prob + lp
        entropy = entropy + e
        logits['action_primitives'] = {'mode':m_logits}
        
        # action primitives
        cursor_mask = torch.zeros(
            mode_sample.shape, dtype=torch.bool, device=mode_sample.device)
        do_release = torch.zeros_like(cursor_mask)
        for i, (name, decoder) in enumerate(self.primitive_decoders.items()):
            primitive_sample = None if sample is None else (
                sample['action_primitives'][name])
            
            if name == 'insert':
                s, lp, e, px, name_logits = decoder(
                    x,
                    sample=primitive_sample,
                    #sample_max=sample_max,
                    shape_eq=shape_eq,
                    color_eq=color_eq,
                )
            else:
                s, lp, e, px, name_logits = decoder(
                    x,
                    sample=primitive_sample,
                )
            
            out_sample['action_primitives'][name] = s
            
            mode_mask = mode_sample == i
            if name in ('remove', 'pick_and_place', 'rotate', 'translate'):
                cursor_mask |= mode_mask
            if name == 'pick_and_place':
                do_release |= mode_mask
            
            x = x * ~mode_mask.view(-1,1) + px * mode_mask.view(-1,1)
            log_prob = log_prob + lp * mode_mask
            entropy = entropy + e * mode_mask
            logits['action_primitives'][name] = name_logits
        
        # sample cursor
        if image_x is not None:
            cursor_sample = None if sample is None else (
                sample['cursor'])
            
            if self.config.dense_decoder_mode in ('dpt', 'dpt_sum'):
                _, b, c = image_x[0].shape
                h = self.config.image_height // self.config.tile_height
                w = self.config.image_width // self.config.tile_width
                image_x = [
                    ix.permute(1,2,0).reshape(b,c,h,w)
                    for ix in image_x]
            elif self.config.dense_decoder_mode == 'simple':
                _, b, c = image_x.shape
                #image_x = self.image_norm(image_x)
                h = self.config.image_height // self.config.tile_height
                w = self.config.image_width // self.config.tile_width
                image_x = image_x.permute(1,2,0).reshape(b,c,h,w)
            s, lp, e, cx, cursor_logits = self.cursor_decoder(
                x,
                image_x,
                pos_snap_eq=pos_snap_eq,
                neg_snap_eq=neg_snap_eq,
                do_release=do_release,
                sample=cursor_sample,
            )
            out_sample['cursor'] = s
            
            x = x + cx * cursor_mask.view(-1,1)
            log_prob = log_prob + lp * cursor_mask
            entropy = entropy + e * cursor_mask
            logits['cursor'] = cursor_logits
        else:
            b = mode_sample.shape[0]
            device = mode_sample.device
            out_sample['cursor'] = {
                'button' : torch.zeros((b,), dtype=torch.long).to(device),
                'click' : torch.zeros((b,2), dtype=torch.long).to(device),
                'release' : torch.zeros((b,2), dtype=torch.long).to(device),
            }
        
        return {
            'value' : value,
            'sample' : out_sample,
            'log_prob' : log_prob,
            'entropy' : entropy,
            'logits' : logits,
        }
    
    def expert(self, observation, info):
        return observation['expert'], observation['num_expert_actions']
    
    def log_prob(self, model_output):
        return model_output['log_prob']
    
    def entropy(self, model_output):
        return model_output['entropy']
    
    def compute_sample(self, sample_output):
        return sample_output['sample']
    
    def compute_action(self, model_output):
        return torch_to_numpy(model_output['sample'])
    
    def value(self, output):
        return output['value']
    
    def save_observation(self, observation, path, index):
        breakpoint()
    
    def visualize(self,
        observation,
        action,
        reward,
        model_output,
        next_image=None,
    ):
        images = observation['image']
        
        mode = action['action_primitives']['mode']
        mode_space = self.action_space['action_primitives']['mode']
        
        button_sample = model_output['sample']['cursor']['button'].view(-1,1,1)
        device = button_sample.device
        pos_islands = torch.LongTensor(
            cursor_islands(observation['pos_snap_render'])).to(device)
        neg_islands = torch.LongTensor(
            cursor_islands(observation['neg_snap_render'])).to(device)
        click_islands = (
            pos_islands * button_sample + neg_islands * (1-button_sample))
        release_islands = (
            pos_islands * (1-button_sample) + neg_islands * button_sample)
        
        if 'cursor' in model_output['logits']:
            click_logits = model_output['logits']['cursor']['click']
            release_logits = model_output['logits']['cursor']['release']
            b,h,w = click_logits.shape
            if self.config.sigmoid_screen_attention:
                click_heatmaps = torch.sigmoid(click_logits).view(b,h,w,1)
                release_heatmaps = torch.sigmoid(release_logits).view(b,h,w,1)
                click_eq_dist = equivalent_probs_categorical(
                    click_heatmaps, click_islands)
                release_eq_dist = equivalent_probs_categorical(
                    release_heatmaps, release_islands)
            else:
                click_heatmaps = torch.softmax(
                    click_logits.view(b,h*w), dim=-1).view(b,h,w,1)
                release_heatmaps = torch.softmax(
                    release_logits.view(b,h*w), dim=-1).view(b,h,w,1)
                click_eq_dist = equivalent_outcome_categorical(
                    click_logits,
                    click_islands,
                )
                release_eq_dist = equivalent_outcome_categorical(
                    release_logits,
                    release_islands,
                )
        else:
            b = action['action_primitives']['mode'].shape[0]
            h, w = 256,256
            click_heatmaps = torch.zeros((b,h,w,1))
            release_heatmaps = torch.zeros((b,h,w,1))
            click_eq_dist = TODO
            release_eq_dist = TODO
        
        visualizations = []
        
        for i, m in enumerate(mode):
            image = images[i]
            current_image = image.copy()
            current_image = write_text(current_image, 'Current Image')
            
            action_image = image.copy()
            
            click_heatmap_background = images[i].copy()
            click_heatmap = click_heatmaps[i].cpu().numpy()
            click_heatmap = heatmap_overlay(
                click_heatmap_background,
                click_heatmap,
                [255,255,255],
                background_scale=0.25,
                max_normalize=True,
            )

            release_heatmap_background = images[i].copy()
            release_heatmap = release_heatmaps[i].cpu().numpy()
            release_heatmap = heatmap_overlay(
                release_heatmap_background,
                release_heatmap,
                [255,255,255],
                background_scale=0.25,
                max_normalize=True,
            )
            
            click_p_map = click_eq_dist.probs[i][click_islands[i]].detach()
            click_min = torch.min(click_p_map)
            click_max = torch.max(click_p_map)
            click_p_map = (click_p_map - click_min) / (
                click_max - click_min + 1e-9)
            click_p_map = (click_p_map.cpu().numpy() * 255).round().astype(
                numpy.uint8)
            h, w = click_p_map.shape
            click_p_map = click_p_map.reshape(h,w,1).repeat(3, axis=2)
            
            release_p_map = release_eq_dist.probs[i][
                release_islands[i]].detach()
            release_min = torch.min(release_p_map)
            release_max = torch.max(release_p_map)
            release_p_map = (
                (release_p_map - release_min) /
                (release_max - release_min + 1e-9)
            )
            release_p_map = (release_p_map.cpu().numpy() * 255).round().astype(
                numpy.uint8)
            h, w = release_p_map.shape
            release_p_map = release_p_map.reshape(h,w,1).repeat(3, axis=2)
            
            mode_name = mode_space.names[m]
            mode_logits = model_output['logits']['action_primitives']['mode'][i]
            mode_prob = torch.softmax(mode_logits, dim=0).cpu()
            mode_lines = []
            for j, name in enumerate(mode_space.names):
                if mode_name == name:
                    prefix = '> '
                else:
                    prefix = ''
                if name == 'rotate':
                    r = action['action_primitives']['rotate'][i]
                    mode_lines.append(
                        '%s%s (%i): %.02f'%(prefix, name, r, mode_prob[j]))
                elif name == 'translate':
                    t = action['action_primitives']['translate'][i]
                    mode_lines.append(
                        '%s%s (%i): %.02f'%(prefix, name, r, mode_prob[j]))
                else:
                    mode_lines.append(
                        '%s%s: %.02f'%(prefix, name, mode_prob[j]))
            action_str = 'Action:\n' + '\n'.join(mode_lines)
            
            '''
            if mode_name == 'viewpoint':
                v = action['action_primitives']['viewpoint'][i]
                action_str = '%s.%s'%(mode_name, ViewpointActions(value=v).name)
            elif mode_name == 'remove':
                r = action['action_primitives']['remove'][i]
                ci, cs = observation['cursor']['click_snap'][i]
                cy, cx = action['cursor']['click'][i]
                action_str = (
                    '%s.%i\n'
                    'Click x/y: %i,%i\n'
                    'Click instance/snap: %i,%i\n'%(
                        mode_name, r, cx, cy, ci, cs)
                )
            elif mode_name == 'insert':
                SHAPE_NAMES = {
                    value:key for key, value in SHAPE_CLASS_LABELS.items()}
                s, c = action['action_primitives']['insert'][i]
                shape_name = SHAPE_NAMES.get(s, 'NONE')
                action_str = (
                    '%s.%i/%i\n'
                    'Shape: %s\n'
                    'Color: %i\n'%(
                        mode_name, s, c, shape_name, c))
            else:
                action_str = mode_name
            '''
            
            action_str += '\nReward: %.04f'%reward[i]
            action_image = write_text(action_image, action_str)

            if mode_name in ('remove', 'pick_and_place', 'rotate', 'translate'):
                click_polarity = action['cursor']['button'][i]
                click_yx = action['cursor']['click'][i]
                if click_polarity:
                    draw_crosshairs(action_image, *click_yx, 3, (255,0,0))
                    draw_crosshairs(click_heatmap, *click_yx, 3, (255,0,0))
                    draw_crosshairs(click_p_map, *click_yx, 3, (255,0,0))
                else:
                    draw_square(action_image, *click_yx, 3, (255,0,0))
                    draw_square(click_heatmap, *click_yx, 3, (255,0,0))
                    draw_square(click_p_map, *click_yx, 3, (255,0,0))

            if mode_name in ('pick_and_place',):
                release_polarity = not bool(click_polarity)
                release_yx = action['cursor']['release'][i]
                if release_polarity:
                    draw_crosshairs(action_image, *release_yx, 3, (0,0,255))
                    draw_crosshairs(release_heatmap, *release_yx, 3, (0,0,255))
                    draw_crosshairs(release_p_map, *release_yx, 3, (0,0,255))
                else:
                    draw_square(action_image, *release_yx, 3, (0,0,255))
                    draw_square(release_heatmap, *release_yx, 3, (0,0,255))
                    draw_square(release_p_map, *release_yx, 3, (0,0,255))
                draw_line(action_image, *click_yx, *release_yx, (255,0,255))
            
            image_row = [current_image]
            if 'target_image' in observation:
                target_image = observation['target_image'][i]
                target_image = write_text(target_image, 'Target Image')
                image_row.append(target_image)
            image_row.append(action_image)
            if next_image is not None:
                result_image = next_image[i]
                result_image = write_text(result_image, 'Result Image')
                image_row.append(result_image)
            image_row.extend([
                click_heatmap,
                release_heatmap,
                click_p_map,
                release_p_map,
            ])
            #breakpoint()
            visualization = stack_images_horizontal(image_row)
            visualizations.append(visualization)
        
        return visualizations

def cursor_islands(snaps):
    b,h,w,_ = snaps.shape
    islands = snaps[...,0] * MAX_SNAPS_PER_BRICK + snaps[...,1]
    _, islands = numpy.unique(islands, return_inverse=True)
    islands = islands.reshape(b, h, w)
    return islands
