import os
import json

import numpy

import torch
import torch.nn as nn

import tqdm

from gymnasium.spaces import Discrete

from splendor.image import save_image
from splendor.json_numpy import NumpyEncoder

from steadfast.hierarchy import flatten_hierarchy, hierarchy_getitem

from avarice.data.numpy_torch import torch_to_numpy

from ltron.bricks.brick_scene import BrickScene
from ltron.visualization.drawing import (
    draw_crosshairs, draw_box, write_text, stack_images_horizontal,
)

from ltron.constants import (
    MAX_SNAPS_PER_BRICK, NUM_SHAPE_CLASSES, NUM_COLOR_CLASSES)
from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.transformer import (
    TransformerConfig, Transformer, init_weights)
from ltron_torch.models.cursor_decoder import CursorDecoder
from ltron_torch.models.insert_decoder import InsertDecoder
from ltron_torch.models.auto_decoder import (
    AutoDecoderConfig,
    AutoDecoder,
    AutoActorCriticDecoder,
    CriticDecoder,
    DiscreteAutoDecoder,
    ConstantDecoder,
)

class LtronVisualTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    AutoDecoderConfig,
):
    embedding_dropout = 0.1
    strict_load = True

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
        #self.image_embedding = AutoEmbedding(
        #    config,
        #    observation_space['color_render'],
        #)
        #self.primitives_embedding = AutoEmbedding(
        #    config,
        #    observation_space['interface']['primitives'],
        #)
        #self.auto_embedding = AutoEmbedding(config, observation_space)
        self.image_embedding = AutoEmbedding(
            config, observation_space['image'])
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        #self.time_embedding = nn.Embedding(
        #    self.config.max_time_steps, config.channels)
        
        # build the decoder token
        #self.decoder_token = nn.Embedding(1, config.channels)
        
        # adding two randns here to simulate the embedding itself plus
        # a learned positional encoding
        self.decoder_token = nn.Parameter(
            torch.randn(1, 1, config.channels) +
            torch.randn(1, 1, config.channels))
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # pre decoder layernorm
        self.predecoder_norm = nn.LayerNorm(config.channels)
        
        # build the decoder
        self.mode_decoder = DiscreteAutoDecoder(
            config, action_space['action_primitives']['mode'])
        self.mode_decoder_names = (
            action_space['action_primitives']['mode'].names)
        
        primitive_decoders = {}
        for name, subspace in action_space['action_primitives'].items():
            if name == 'mode' or name not in self.mode_decoder_names:
                continue
            elif name == 'insert':
                primitive_decoders[name] = InsertDecoder(config)
            elif isinstance(subspace, Discrete) and subspace.n == 2:
                primitive_decoders[name] = ConstantDecoder(config, 1)
            else:
                primitive_decoders[name] = AutoDecoder(config, subspace)
        
        self.primitive_decoders = nn.ModuleDict(primitive_decoders)
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
        image_kwargs = self.image_embedding.observation_to_kwargs(
            observation['image'], info, done, model_output)
        #time_step = torch.LongTensor(observation['time']).to(device)
        #pos_eq = observation['interface']['pos_equivalence']
        #pos_eq = torch.LongTensor(pos_eq).to(device)
        #neg_eq = observation['interface']['neg_equivalence']
        #neg_eq = torch.LongTensor(neg_eq).to(device)
        
        if 'pos_snap_render' in observation:
            pos_snaps = observation['pos_snap_render']
            pos_islands = torch.LongTensor(cursor_islands(pos_snaps)).to(device)
            
            neg_snaps = observation['neg_snap_render']
            neg_islands = torch.LongTensor(cursor_islands(neg_snaps)).to(device)
        else:
            pos_islands = None
            neg_islands = None
        
        # shape and color equivalence
        if self.config.shape_class_labels is None:
            num_shape_classes = NUM_SHAPE_CLASSES
        else:
            num_shape_classes = len(self.config.shape_class_labels)+1
        if self.config.color_class_labels is None:
            num_color_classes = NUM_COLOR_CLASSES
        else:
            num_color_classes = len(self.config.color_class_labels)+1
        if 'target_assembly' in observation:
            shapes = observation['target_assembly']['shape']
            colors = observation['target_assembly']['color']
            b, s = shapes.shape
            active_b, active_s = numpy.where(shapes)
            compact_shapes = shapes[active_b, active_s]
            _, compact_shapes_unique = numpy.unique(
                compact_shapes, return_inverse=True)
            zero_index = numpy.max(compact_shapes_unique) + 2
            shape_islands = numpy.zeros(
                (b, num_shape_classes), dtype=numpy.int64)
            shape_islands[active_b, compact_shapes] = compact_shapes_unique + 1
            shape_islands[:,0] = zero_index
            shape_islands = torch.LongTensor(shape_islands).to(device)
            
            compact_colors = colors[active_b, active_s]
            _, compact_colors_unique = numpy.unique(
                compact_colors, return_inverse=True)
            zero_index = numpy.max(compact_colors_unique) + 2
            color_islands = numpy.zeros(
                (b, num_color_classes), dtype=numpy.int64)
            color_islands[active_b, compact_colors] = compact_colors_unique + 1
            color_islands[:,0] = zero_index
            color_islands = torch.LongTensor(color_islands).to(device)
        else:
            b = observation['image'].shape[0]
            shape_islands = torch.zeros(
                (b, num_shape_classes), dtype=torch.long, device=device)
            color_islands = torch.zeros(
                (b, num_color_classes), dtype=torch.long, device=device)
        
        return {
            'image_kwargs' : image_kwargs,
            #'time_step' : time_step,
            'pos_snap_eq' : pos_islands,
            'neg_snap_eq' : neg_islands,
            'shape_eq' : shape_islands,
            'color_eq' : color_islands,
        }
    
    def forward(self,
        image_kwargs,
        #time_step,
        pos_snap_eq,
        neg_snap_eq,
        shape_eq,
        color_eq,
    ):
        
        # use the embedding to compute the tokens
        if True:
            x = self.image_embedding(**image_kwargs)
            h = self.config.image_height // self.config.tile_height
            w = self.config.image_width // self.config.tile_width
            hw,b,c = x.shape
        else:
            hw = None
        
            x = self.time_embedding(time_step)
            b,c = x.shape
            x = x.view(1,b,c)
        
        # concat tokens together here
        #decoder_token = self.decoder_token.weight.view(1,1,c).expand(1,b,c)
        decoder_token = self.decoder_token.expand(1,b,c)
        x = torch.cat((decoder_token, x), dim=0)
        
        # extract and concatenate all tokens
        #all_tokens = flatten_hierarchy(x)
        #s, b, c = all_tokens[0].shape
        #decoder_token = self.decoder_token.weight.view(1,1,c).expand(1,b,c)
        #all_tokens.append(decoder_token)
        #x = torch.cat(all_tokens, dim=0)
        
        # embedding dropout
        x = self.embedding_dropout(x)
        
        # push the concatenated tokens through the transformer
        x = self.encoder(x)[-1]
        decode_x = self.predecoder_norm(x[0])
        if hw is not None:
            image_x = x[1:hw+1]
        else:
            image_x = None
        
        value = self.critic_decoder(decode_x)
        
        # push the decoder tokens through the auto decoder
        #x = self.decoder(x, **decoder_kwargs)
        
        return {
            'decode_x' : decode_x,
            'image_x' : image_x,
            'pos_snap_eq' : pos_snap_eq,
            'neg_snap_eq' : neg_snap_eq,
            'shape_eq' : shape_eq,
            'color_eq' : color_eq,
            'value' : value,
        }
    
    def sample_log_prob(self, output, sample=None):
        decode_x = output['decode_x']
        image_x = output['image_x']
        
        out_sample = {}
        log_prob = 0.
        entropy = 0.
        logits = {}
        
        # mode
        mode_sample = None if sample is None else (
            sample['action_primitives']['mode'])
        mode_sample, lp, e, x, m_logits = self.mode_decoder(
            decode_x, sample=mode_sample)
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
                    shape_eq = output['shape_eq'],
                    color_eq = output['color_eq'],
                )
            else:
                s, lp, e, px, name_logits = decoder(
                    x,
                    sample=primitive_sample,
                )
            
            out_sample['action_primitives'][name] = s
            
            mode_mask = mode_sample == i
            if name in ('remove', 'pick_and_place', 'rotate'):
                cursor_mask |= mode_mask
            if name == 'pick_and_place':
                do_release |= mode_mask
            
            x = x * ~mode_mask.view(-1,1) + px * mode_mask.view(-1,1)
            log_prob = log_prob + lp * mode_mask
            entropy = entropy + e * mode_mask
            logits['action_primitives'][name] = name_logits
        
        #if self.forward_passes > 1024:
        #    breakpoint()
        #self.forward_passes += 1
        
        # sample cursor
        if image_x is not None:
            cursor_sample = None if sample is None else (
                sample['cursor'])
            _, b, c = image_x.shape
            h = self.config.image_height // self.config.tile_height
            w = self.config.image_width // self.config.tile_width
            image_x = image_x.permute(1,2,0).reshape(b,c,h,w)
            s, lp, e, cx, cursor_logits = self.cursor_decoder(
                x,
                image_x,
                pos_snap_eq=output['pos_snap_eq'],
                neg_snap_eq=output['neg_snap_eq'],
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
            'sample' : out_sample,
            'log_prob' : log_prob,
            'entropy' : entropy,
            'logits' : logits,
            #'click_logits' : c_logits,
            #'release_logits' : r_logits,
        }
    
    def sample_output_to_log_prob(self, sample_output):
        return sample_output['log_prob']
    
    def sample_output_to_entropy(self, sample_output):
        return sample_output['entropy']
    
    def sample_output_to_sample(self, sample_output):
        return sample_output['sample']
    
    def sample_output_to_action(self, sample_output):
        action = torch_to_numpy(sample_output['sample'])
        # all of the below is to support instances where you only want some
        # decoders
        #if 'viewpoint' not in action:
        #    action['action_primitives']['viewpoint'] = 0
        #if 'pick_and_place' not in action:
        #    action['action_primitives']['pick_and_place'] = 0
        #if 'rotate' not in action:
        #    action['action_primitives']['rotate'] = 0
        #if 'remove' not in action:
        #    action['action_primitives']['remove'] = 0
        #if 'insert' not in action['action_primitives']:
        #    action['action_primitives']['insert'] = numpy.array(
        #        [0,0],
        #        dtype=numpy.in64,
        #    )
        #mode_names = action['action_primitives']['mode'].names
        #
        #n = action['action_primitives']['mode']
        return action
    
    def value(self, output):
        return output['value']
    
    def loss(self, x, y, seq_pad):
        #if True:
        #    p = torch.softmax(x, dim=-1).detach()
        #    y = p * y
        loss = torch.sum(-torch.log_softmax(x, dim=-1) * y, dim=-1)

        # automatically normalize the labels? why?
        #y_sum = torch.sum(y, dim=-1)
        #y_sum[y_sum == 0] = 1 # avoid divide by zero
        #loss = loss / y_sum

        # average loss over valid entries
        s_i, b_i = get_seq_batch_indices(torch.LongTensor(seq_pad))
        loss = loss[s_i, b_i].mean()
        
        return loss

def cursor_islands(snaps):
    b,h,w,_ = snaps.shape
    islands = snaps[...,0] * MAX_SNAPS_PER_BRICK + snaps[...,1]
    _, islands = numpy.unique(islands, return_inverse=True)
    islands = islands.reshape(b, h, w)
    return islands
