import os
import json

import numpy

import torch
import torch.nn as nn

import tqdm

from splendor.image import save_image
from splendor.json_numpy import NumpyEncoder

from steadfast.hierarchy import flatten_hierarchy, hierarchy_getitem

from avarice.data.numpy_torch import torch_to_numpy

from ltron.bricks.brick_scene import BrickScene
from ltron.visualization.drawing import (
    draw_crosshairs, draw_box, write_text, stack_images_horizontal,
)

from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.transformer import (
    TransformerConfig, Transformer, init_weights)
from ltron_torch.models.cursor_decoder import CursorDecoder
from ltron_torch.models.auto_decoder import (
    AutoDecoderConfig,
    AutoDecoder,
    AutoActorCriticDecoder,
    AutoDecoder,
    CriticDecoder,
    DiscreteAutoDecoder,
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
        decode_mode='actor',
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
        
        # build the decoder token
        self.decoder_token = nn.Embedding(1, config.channels)
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # build the decoder
        self.mode_decoder = DiscreteAutoDecoder(
            config, action_space['interface']['primitives']['mode'])
        
        primitive_decoders = {}
        for name, subspace in action_space['interface']['primitives'].items():
            if name == 'mode':
                continue
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
    
    def observation_to_kwargs(self, observation, info, done, model_output):
        image_kwargs = self.image_embedding.observation_to_kwargs(
            observation['image'], info, done, model_output)
        device = next(iter(self.parameters())).device
        click_eq = observation['interface']['cursor']['click_equivalence']
        click_eq = torch.LongTensor(click_eq).to(device)
        release_eq = observation['interface']['cursor']['release_equivalence']
        release_eq = torch.LongTensor(release_eq).to(device)
        return {
            'image_kwargs' : image_kwargs,
            'click_eq' : click_eq,
            'release_eq' : release_eq,
        }
    
    '''
    def sample_action(self, output, observation, info):
        return self.decoder.sample_action(output, observation, info)
    
    def log_prob(self, output, action):
        return output['log_prob']
    
    def entropy(self, output):
        return output['entropy']
    '''
    
    def forward(self, image_kwargs, click_eq, release_eq):
        
        # use the embedding to compute the tokens
        x = self.image_embedding(**image_kwargs)
        h = self.config.image_height // self.config.tile_height
        w = self.config.image_width // self.config.tile_width
        hw,b,c = x.shape
        
        # concat tokens together here
        decoder_token = self.decoder_token.weight.view(1,1,c).expand(1,b,c)
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
        decode_x = x[0]
        image_x = x[1:hw+1]
        
        value = self.critic_decoder(decode_x)
        
        # push the decoder tokens through the auto decoder
        #x = self.decoder(x, **decoder_kwargs)
        
        return {
            'decode_x' : decode_x,
            'image_x' : image_x,
            'click_eq' : click_eq,
            'release_eq' : release_eq,
            'value' : value,
        }
    
    def sample_log_prob(self, output, sample=None):
        decode_x = output['decode_x']
        image_x = output['image_x']
        
        out_sample = {}
        log_prob = 0.
        entropy = 0.
        
        # mode
        mode_sample = None if sample is None else (
            sample['interface']['primitives']['mode'])
        mode_sample, lp, e, x = self.mode_decoder(decode_x, sample=mode_sample)
        log_prob = log_prob + lp
        entropy = entropy + e
        
        # action primitives
        out_sample['interface'] = {'primitives':{'mode':mode_sample}}
        cursor_mask = torch.zeros(
            mode_sample.shape, dtype=torch.bool, device=mode_sample.device)
        do_release = torch.zeros_like(cursor_mask)
        for i, (name, decoder) in enumerate(self.primitive_decoders.items()):
            primitive_sample = None if sample is None else (
                sample['interface']['primitives'][name])
            s, lp, e, px = decoder(decode_x, sample=primitive_sample)
            out_sample['interface']['primitives'][name] = s
            
            mode_mask = mode_sample == i
            if name == 'remove':
                cursor_mask |= mode_mask
            x = x + px * mode_mask.view(-1,1)
            log_prob = log_prob + lp * mode_mask
            entropy = entropy + e * mode_mask
        
        # sample cursor
        cursor_sample = None if sample is None else (
            sample['interface']['cursor'])
        _, b, c = image_x.shape
        h = self.config.image_height // self.config.tile_height
        w = self.config.image_width // self.config.tile_width
        image_x = image_x.permute(1,2,0).reshape(b,c,h,w)
        s, lp, e, cx = self.cursor_decoder(
            x,
            image_x,
            click_eq=output['click_eq'],
            release_eq=output['release_eq'],
            do_release=do_release,
            sample=cursor_sample,
        )
        out_sample['interface']['cursor'] = s
        
        x = x + cx * cursor_mask.view(-1,1)
        log_prob = log_prob + lp * cursor_mask
        entropy = entropy + e * cursor_mask
        
        return out_sample, log_prob, entropy
    
    def sample_to_action(self, sample):
        return torch_to_numpy(sample)
    
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
