import os
import json

import numpy

import torch
from torch.nn import Module

import tqdm

from splendor.image import save_image
from splendor.json_numpy import NumpyEncoder

from ltron.hierarchy import index_hierarchy
from ltron.bricks.brick_scene import BrickScene
from ltron.gym.spaces import (
    MultiScreenPixelSpace, MaskedTiledImageSpace, AssemblySpace,
)
from ltron.visualization.drawing import (
    draw_crosshairs, draw_box, stack_images_horizontal,
)

from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    init_weights,
)
from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.auto_decoder import (
    AutoDecoderConfig, AutoDecoder)
from ltron_torch.models.padding import cat_multi_padded_seqs, decat_padded_seq

class AutoTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    AutoDecoderConfig,
):
    action_decoder_dropout = 0.1

class AutoTransformer(Module):
    def __init__(self,
        config,
        observation_space,
        action_space,
        checkpoint=None
    ):
        # Module super
        super().__init__()
        
        # save config, observation and action space
        self.config = config
        self.observation_space = observation_space
        self.action_space = action_space
        
        # build the decoder
        # (do this before the encoder because we need the readout_layout)
        self.decoder = AutoDecoder(config, action_space)
        
        # build the token embedding
        self.embedding = AutoEmbedding(
            config, observation_space, self.decoder.readout_layout)
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # load checkpoint or initialize weights
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint)
        elif config.init_weights:
            self.apply(init_weights)
    
    def zero_all_memory(self):
        self.encoder.zero_all_memory()
    
    def observation_to_tensors(self, batch, seq_pad):
        # get the auto_embedding's tensors
        x = self.embedding.observation_to_tensors(batch, seq_pad)
        
        # add seq_pad
        device = next(iter(self.parameters())).device
        x['seq_pad'] = torch.LongTensor(seq_pad).to(device)
        
        return x
    
    def forward(self, x, t, pad, seq_pad, use_memory=None):
        
        # use the embedding to compute the tokens, time steps and padding
        emb_x, emb_t, emb_pad = self.embedding(x, t, pad)
        
        # concatenate all tokens
        # readout tokens go first so we can pull them out easier
        order_x = [emb_x['readout']]
        order_t = [emb_t['readout']]
        order_pad = [emb_pad['readout']]
        name_seq = ['readout']
        
        # everything else goes after the readout tokens
        non_readout_pad = torch.zeros_like(emb_pad['readout'])
        for name in emb_x:
            if name == 'readout':
                continue
            order_x.append(emb_x[name])
            order_t.append(emb_t[name])
            order_pad.append(emb_pad[name])
            non_readout_pad = non_readout_pad + emb_pad[name]
            name_seq.append(name)
        
        # do the concatenation
        cat_x, cat_pad = cat_multi_padded_seqs(order_x, order_pad)
        cat_t, _ = cat_multi_padded_seqs(order_t, order_pad)
        
        # push the concatenated sequence through the transformer
        enc_x = self.encoder(cat_x, cat_t, cat_pad, use_memory=use_memory)[-1]
        
        # strip off the decoder tokens
        decat_x, _ = decat_padded_seq(
            enc_x, emb_pad['readout'], non_readout_pad
        )
        
        # push the decoder tokens through the auto decoder
        head_x = self.decoder(decat_x, x['readout'], t['readout'], seq_pad)
        
        return head_x
    
    def forward_rollout(self, terminal, *args, **kwargs):
        device = next(self.parameters()).device
        use_memory = torch.BoolTensor(~terminal).to(device)
        return self(*args, **kwargs, use_memory=use_memory)
    
    def tensor_to_actions(self, x, mode='sample'):
        s, b, a = x.shape
        assert s == 1
        if mode == 'sample':
            distribution = self.tensor_to_distribution(x)
            actions = distribution.sample().cpu().view(b).numpy()
        elif mode == 'max':
            actions = torch.argmax(x, dim=-1).cpu().view(b).numpy()

        return actions
    
    def tensor_to_distribution(self, x):
        return self.decoder.tensor_to_distribution(x)
    
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
                    
                    # compute action probabilities
                    seq_action_p = seq['distribution'][:,i]
                    max_action_p = numpy.max(seq_action_p, axis=-1)
                    
                    # build the frames
                    for frame_id in range(seq_len):
                        frame_observation = index_hierarchy(index_hierarchy(
                            seq['observation'], frame_id), i)
                        
                        # add the action to the summary
                        summary['action'].append(
                            self.action_space.unravel(seq['action'][frame_id,i])
                        )
                        
                        images = {}
                        for name, space in self.observation_space.items():
                            if isinstance(space, MaskedTiledImageSpace):
                                image = frame_observation[name]['image']
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
                            n, *yxc = nyxc
                            if n != 'NO_OP':
                                y, x, c = yxc
                                action_image = images[n]
                                color = (
                                    self.embedding.cursor_colors[action_name][c]
                                )
                                draw_crosshairs(action_image, y, x, 5, color)
                        
                        if images:
                            out_image = stack_images_horizontal(
                                images.values(), spacing=4)
                            frame_path = os.path.join(
                                seq_path,
                                'frame_%04i_%06i_%04i.png'%(
                                    epoch, seq_id, frame_id),
                            )
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
