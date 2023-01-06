import os
import json

import numpy

import torch
import torch.nn as nn

import tqdm

from splendor.image import save_image
from splendor.json_numpy import NumpyEncoder

from steadfast.hierarchy import flatten_hierarchy, hierarchy_getitem

from ltron.bricks.brick_scene import BrickScene

from ltron.visualization.drawing import (
    draw_crosshairs, draw_box, write_text, stack_images_horizontal,
)

from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.transformer import (
    TransformerConfig, Transformer, init_weights)
from ltron_torch.models.auto_decoder import (
    AutoDecoderConfig, AutoActorDecoder, AutoActorCriticDecoder)

class AutoTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    AutoDecoderConfig,
):
    channels = 768
    embedding_dropout = 0.1
    strict_load = True

class AutoTransformer(nn.Module):
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
        
        # build the token embedding
        self.embedding = AutoEmbedding(config, observation_space)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # build the decoder token
        self.decoder_token = nn.Embedding(1, config.channels)
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # build the decoder
        if decode_mode == 'actor':
            self.decoder = AutoActorDecoder(config, action_space)
        elif decode_mode == 'actor_critic':
            self.decoder = AutoActorCriticDecoder(config, action_space)
        
        # load checkpoint or initialize weights
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint, strict=config.strict_load)
        elif config.init_weights:
            self.apply(init_weights)
    
    def observation_to_kwargs(self, observation, info, done, model_output):
        return self.embedding.observation_to_kwargs(
            observation, info, done, model_output)
    
    def sample_action(self, output, observation, info):
        return self.decoder.sample_action(output, observation, info)
    
    def log_prob(self, output, action):
        return self.decoder.log_prob(output, action)
    
    def entropy(self, output):
        return self.decoder.entropy(output)
    
    def value(self, output):
        return self.decoder.value(output)
    
    def forward(self, *args, **kwargs):
        
        # use the embedding to compute the tokens
        x = self.embedding(*args, **kwargs)
        
        # extract and concatenate all tokens
        all_tokens = flatten_hierarchy(x)
        s, b, c = all_tokens[0].shape
        decoder_token = self.decoder_token.weight.view(1,1,c).expand(1,b,c)
        all_tokens.append(decoder_token)
        x = torch.cat(all_tokens, dim=0)
        
        # embedding dropout
        x = self.embedding_dropout(x)
        
        # push the concatenated tokens through the transformer
        x = self.encoder(x)[-1]
        
        # strip off the decoder tokens
        x = x[-1]
        
        # push the decoder tokens through the auto decoder
        x = self.decoder(x)
        
        return x
    
    '''
    def observation_to_label(self, batch, pad, supervision_mode):
        device = next(self.parameters()).device
        
        if supervision_mode == 'action':
            # pull the actions
            actions = torch.LongTensor(batch['action']).to(device)
            s, b = actions.shape[:2]
            
            # build the y tensor
            y = torch.zeros((s, b, self.action_space.n), device=device)
            ss = torch.arange(s).view(s,1).expand(s,b).reshape(-1).cuda()
            bb = torch.arange(b).view(1,b).expand(s,b).reshape(-1).cuda()
            y[ss,bb,actions.view(-1)] = 1
        
        elif (supervision_mode == 'expert_uniform_distribution' or
            supervision_mode == 'expert_all'
        ):
            # pull the expert labels
            labels = torch.LongTensor(batch['observation']['expert']).to(device)
            s, b = labels.shape[:2]
            
            # build the y tensor
            y = torch.zeros((s,b,self.action_space.n), device=device)
            ss, bb, nn = torch.where(labels)
            ll = labels[ss,bb,nn]
            y[ss,bb,ll] = 1
            if supervision_mode == 'expert_uniform_distribution':
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
    '''
    
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
    
    def visualize_episodes(
        self,
        epoch,
        episodes,
        visualization_episodes_per_epoch,
        visualization_directory,
    ):
        # iterate through each episode
        with tqdm.tqdm(total=visualization_episodes_per_epoch) as iterate:
            for seq, pad in episodes:
                b = pad.shape[0]
                
                for i in range(b):
                    seq_id = iterate.n
                    
                    # get the sequence length
                    seq_len = pad[i]
                    
                    # make the sequence directory
                    seq_path = os.path.join(
                        visualization_directory, 'seq_%06i'%seq_id)
                    if not os.path.exists(seq_path):
                        os.makedirs(seq_path)
                    
                    # build the summary
                    summary = {
                        'action':[],
                        'reward':seq['reward'][:pad[i],i],
                    }
                    
                    for key, value in seq['observation'].items():
                        if 'cursor' in key and 'selected' in value:
                            summary['%s_selected'%key] = (
                                value['selected'][:,i])
                    
                    # compute action probabilities
                    seq_action_p = seq['distribution'][:,i]
                    max_action_p = numpy.max(seq_action_p, axis=-1)
                    
                    # build the frames
                    for frame_id in range(seq_len):
                        frame_observation = index_hierarchy(index_hierarchy(
                            seq['observation'], frame_id), i)
                        distribution = seq['distribution'][frame_id,i]
                        name_distribution = self.action_space.unravel_vector(
                            distribution)
                        
                        # add the action to the summary
                        summary['action'].append(
                            self.action_space.unravel(seq['action'][frame_id,i])
                        )
                        
                        images = {}
                        for name, space in self.observation_space.items():
                            if isinstance(space, MaskedTiledImageSpace):
                                image = frame_observation[name]['image']
                                
                                '''
                                for map_name, maps in cursor_maps.items():
                                    if name in maps:
                                        colors = self.embedding.cursor_colors[
                                            map_name]
                                        m = maps[name][frame_id]
                                        for c in range(m.shape[-1]):
                                            mc = (
                                                m[..., c] /
                                                max_action_p[frame_id]
                                            )
                                            mc = mc.reshape(*mc.shape, 1)
                                            image = (
                                                image * (1.-mc) + mc*colors[c])
                                '''
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
                        
                        action = index_hierarchy(seq['action'], frame_id)[i]
                        action_name, *nyxc = self.action_space.unravel(action)
                        action_subspace = (
                            self.action_space.subspaces[action_name])
                        if isinstance(action_subspace, MultiScreenPixelSpace):
                            n, m, *yxc = nyxc
                            if n != 'NO_OP':
                                action_image = images[n + '_color_tiles']
                                decoder = self.decoder.decoders[action_name]
                                #color = decoder.color
                                if m == 'deselect':
                                    color = [0,255,0]
                                    action_image[0,:] = color
                                    action_image[-1,:] = color
                                    action_image[:,0] = color
                                    action_image[:,-1] = color
                                else:
                                    y, x, c = yxc
                                    if c:
                                        color = [0,0,255]
                                    else:
                                        color = [255,0,0]
                                    draw_crosshairs(
                                        action_image, y, x, 5, color)
                        
                        #for name in self.action_space.keys():
                        #    subspace = self.action_space.subspaces[name]
                            #if isinstance(subspace, MultiScreenPixelSpace):
                            #subshape = self.action_space.get_shape(name)
                            #for subname in subshape.keys():
                            action_distribution = name_distribution[action_name]
                            max_p = 0.
                            for subname in action_distribution.keys():
                                if subname == 'NO_OP':
                                    continue
                                screen_dist = name_distribution[
                                    action_name][subname]['screen']
                                max_screen_p = numpy.max(screen_dist)
                                max_p = max(max_p, max_screen_p)
                            
                            for subname in action_distribution.keys():
                                if subname == 'NO_OP':
                                    continue
                                screen_dist = name_distribution[
                                    action_name][subname]['screen']
                                neg_dist = screen_dist[...,0]
                                pos_dist = screen_dist[...,1]
                                #m = numpy.max(screen_dist) * 2.
                                #if m > 0.:
                                if max_p > 0.:
                                    neg_dist = neg_dist / (max_p * 2.)
                                    pos_dist = pos_dist / (max_p * 2.)
                                decoder = self.decoder.decoders[action_name]
                                for dist, color in [
                                    (neg_dist, [255,0,0]), (pos_dist, [0,0,255])
                                ]:
                                    h,w = dist.shape
                                    dist = dist.reshape(h,w,1)
                                    images[subname + '_color_tiles'] = (
                                        images[subname + '_color_tiles'] *
                                        (1. - dist) +
                                        numpy.array(color) * dist)
                        
                        if images:
                            out_image = stack_images_horizontal(
                                images.values(), spacing=4)
                            frame_path = os.path.join(
                                seq_path,
                                'frame_%04i_%06i_%04i.png'%(
                                    epoch, seq_id, frame_id),
                            )
                            action_string = str(
                                (action_name,) + tuple(nyxc)) + ' (%i)'%action
                            out_image = write_text(out_image, action_string)
                            save_image(out_image, frame_path)
                    
                    summary_path = os.path.join(seq_path, 'summary.json')
                    with open(summary_path, 'w') as f:
                        json.dump(summary, f, indent=2, cls=NumpyEncoder)
                    
                    iterate.update(1)
                    if iterate.n >= visualization_episodes_per_epoch:
                        break
                
                else:
                    continue
                
                break


