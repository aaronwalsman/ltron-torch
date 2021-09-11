import random

import numpy

import torch
from torch.distributions import Categorical

from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.dataset.paths import get_dataset_info
from ltron.gym.reassembly_env import handspace_reassembly_template_action
from ltron.hierarchy import index_hierarchy

from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.gym_tensor import gym_space_to_tensors, default_tile_transform
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)

# ppo model ====================================================================

class ReassemblyPPOModel(ActorCriticModel[CategoricalDistr]):
    def __init__(self, config):
        self.model = build_reassembly_model(config)
    
    def forward(self, observations, memory, prev_actions, masks):
        # TODO: is_blind
        
        # chop frames into tiles
        if single_frame:
            # generate workspace tiles
            pad = numpy.ones(b, dtype=numpy.long)
            wx, wi, w_pad = batch_deduplicate_tiled_seqs(
                workspace_frame, pad, tw, th,
                background=prev_workspace_frame,
                s_start=seq_lengths,
            )
            wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
            wi[:,:,0] = 0
            
            # generate handspace tiles
            hx, hi, h_pad = batch_deduplicate_tiled_seqs(
                handspace_frame, pad, tw, th,
                background=prev_handspace_frame,
                s_start=seq_lengths,
            )
            hi = numpy.insert(hi, (0,1,1), -1, axis=-1)
            hi[:,:,0] = 0
            
        elif multi_frame:
            wx, wi, w_pad = batch_deduplicate_tiled_seqs(
            
            )
        
        # TODO[Klemen] check input dimensions of observation and trim to size
        # 5 vs. 4 dimensional multi-agent thing
        
        x_out, x_pad = self.model(
            tile_x, tile_i, tile_pad,
            token_x, token_i, token_pad,
            decoder_i, decoder_pad,
            terminal=terminal,
        )
        
        # separate x_out in to actions and values
        
        ac_out = ActorCriticOutput(
            distrbutions = x_act, values=x_val, extras={},
        )
        
        return ac_out, memory

# expert =======================================================================

class ReassemblyExpert:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.orders = [None for _ in range(batch_size)]
        self.eagle = False
    
    def __call__(self, observation, terminal, reward):
        
        actions = []
        for i, t in enumerate(terminal):
            obs = index_hierarchy(observation, i)
            r_obs = obs['reassembly']
            
            act = handspace_reassembly_template_action()
            
            # if terminal, generate a new order
            if t:
                self.orders[i] = generate_order(obs)
            
            reassembling = obs['reassembly']['reassembling']
            if not reassembling:
                
                # if there are no objects left, switch to reassembly
                if not numpy.any(r_obs['workspace_configuration']['class']):
                    act['reassembly']['start'] = 1
                    actions.append(act)
                    continue
                
                # figure out the next item to remove
                for instance_id in self.orders[i]:
                    if r_obs['workspace_configuration']['class'][instance_id]:
                        next_instance = instance_id
                
                # pick a snap pixel and remove it
                pos_snaps = obs['workspace_pos_snap_render']
                neg_snaps = obs['workspace_neg_snap_render']
                pos_y, pos_x = numpy.where(pos_snaps[:,:,0] == next_instance)
                neg_y, neg_x = numpy.where(neg_snaps[:,:,0] == next_instance)
                
                y = numpy.concatenate((pos_y, neg_y))
                x = numpy.concatenate((pos_x, neg_x))
                p = numpy.concatenate((
                    numpy.ones(pos_y.shape, dtype=numpy.long),
                    numpy.zeros(neg_y.shape, dtype=numpy.long)))
                
                r = random.randint(0, y.shape[0]-1)
                pick_y = y[r]
                pick_x = x[r]
                polarity = p[r]
                
                act['disassembly']['activate'] = True
                act['disassembly']['polarity'] = polarity
                act['disassembly']['direction'] = 0 # TMP
                act['disassembly']['pick'] = numpy.array(
                    (pick_y, pick_x), dtype=numpy.long)
                actions.append(act)
                
            else:
                reverse_order = list(reversed(self.orders[i]))
                num_bricks = numpy.sum(
                    r_obs['workspace_configuration']['class'] != 0)
                
                # if the reward is 1, then we are done
                if reward[i] == 1.:
                    act['reassembly']['end'] = 1
                    actions.append(act)
                    continue
                
                if num_bricks <= 1:
                    needs_rotate = False
                else:
                    DO_A_THING

                
                '''
                if num_bricks == len(reverse_order):
                    act['reassembly']['end'] = 1
                    actions.append(act)
                    continue
                elif num_bricks <= 1:
                    needs_rotate = False
                else:
                    needs_rotate = False # TMP for now
                '''
                if needs_rotate:
                    print('fix me up yo')
                    import pdb
                    pdb.set_trace()
                
                else:
                    instance_id = reverse_order[num_bricks]
                    class_id = r_obs['target_configuration']['class'][
                        instance_id]
                    color_id = r_obs['target_configuration']['color'][
                        instance_id]
                    
                    # pick up the correct brick if necessary
                    if r_obs['handspace_configuration']['class'][1] != class_id:
                        act['insert_brick']['class_id'] = class_id
                        act['insert_brick']['color_id'] = color_id
                        actions.append(act)
                        continue
                    
                    # place the brick if it is already picked up
                    else:
                        # if there is nothing here yet add the first instance
                        # using the place_at_origin flag
                        if num_bricks == 0:
                            pos_snaps = obs['handspace_pos_snap_render']
                            neg_snaps = obs['handspace_neg_snap_render']
                            pos_y, pos_x = numpy.where(pos_snaps[:,:,0] != 0)
                            neg_y, neg_x = numpy.where(neg_snaps[:,:,0] != 0)
                            
                            y = numpy.concatenate((pos_y, neg_y))
                            x = numpy.concatenate((pos_x, neg_x))
                            p = numpy.concatenate((
                                numpy.ones(pos_y.shape, dtype=numpy.long),
                                numpy.zeros(neg_y.shape, dtype=numpy.long)))
                            
                            r = random.randint(0, y.shape[0]-1)
                            pick_y = y[r]
                            pick_x = x[r]
                            polarity = p[r]
                            
                            act['pick_and_place']['activate'] = True
                            act['pick_and_place']['polarity'] = polarity
                            act['pick_and_place']['pick'] = numpy.array(
                                [pick_y, pick_x])
                            act['pick_and_place']['place_at_origin'] = True
                            actions.append(act)
                            continue
                        
                        else:
                            # get a list of edges that are supposed to exist
                            # between this brick and the rest of the scene
                            edge_ids = r_obs['target_configuration']['edges'][
                                'edge_index']
                            hand_edges = edge_ids[0] == instance_id
                            
                            existing_brick_ids = reverse_order[:num_bricks]
                            
                            expanded_edge_ids = edge_ids.reshape(
                                (*edge_ids.shape, 1))
                            scene_edges = numpy.logical_or.reduce(
                                expanded_edge_ids[1] == existing_brick_ids,
                                axis=1)
                            connecting_edges = hand_edges & scene_edges
                            
                            connecting_edges = edge_ids[:,connecting_edges]
                            
                            r = random.randint(0, connecting_edges.shape[1]-1)
                            hi, si, hs, ss = connecting_edges[:,r]
                            hand_hi = 1
                            scene_si = reverse_order.index(si)+1
                            
                            # pick
                            pos_hand_snaps = obs['handspace_pos_snap_render']
                            neg_hand_snaps = obs['handspace_neg_snap_render']
                            
                            pos_locations = pos_hand_snaps == [hand_hi, hs]
                            pos_y, pos_x = numpy.where(numpy.logical_and.reduce(
                                pos_locations, axis=2))
                            neg_locations = neg_hand_snaps == [hand_hi, hs]
                            neg_y, neg_x = numpy.where(numpy.logical_and.reduce(
                                neg_locations, axis=2))
                            polarity = numpy.concatenate((
                                numpy.ones(pos_y.shape[0], dtype=numpy.long),
                                numpy.zeros(neg_y.shape[0], dtype=numpy.long)
                            ))
                            yx = numpy.stack((
                                numpy.concatenate((pos_y, neg_y)),
                                numpy.concatenate((pos_x, neg_x)),
                            ))
                            
                            r = random.randint(0, yx.shape[1]-1)
                            pick_y, pick_x = yx[:,r]
                            polarity = polarity[r]
                            
                            # place
                            pos_work_snaps = obs['workspace_pos_snap_render']
                            neg_work_snaps = obs['workspace_neg_snap_render']
                            
                            pos_locations = pos_work_snaps == [scene_si, ss]
                            pos_y, pos_x = numpy.where(numpy.logical_and.reduce(
                                pos_locations, axis=2))
                            neg_locations = neg_work_snaps == [scene_si, ss]
                            neg_y, neg_x = numpy.where(numpy.logical_and.reduce(
                                neg_locations, axis=2))
                            yx = numpy.stack((
                                numpy.concatenate((pos_y, neg_y)),
                                numpy.concatenate((pos_x, neg_x)),
                            ))
                            
                            r = random.randint(0, yx.shape[1]-1)
                            place_y, place_x = yx[:,r]
                            
                            act['pick_and_place']['activate'] = True
                            act['pick_and_place']['polarity'] = polarity
                            act['pick_and_place']['pick'] = numpy.array(
                                [pick_y, pick_x])
                            act['pick_and_place']['place'] = numpy.array(
                                [place_y, place_x])
                            actions.append(act)
                            continue
        
        return actions


def generate_order(observation):
    return [4,3,2,1] # TMP


# build functions ==============================================================

def build_reassembly_model(config):
    print('-'*80)
    print('Building reassembly model')
    wh = config.workspace_image_height // config.tile_height
    ww = config.workspace_image_width  // config.tile_width
    hh = config.handspace_image_height // config.tile_height
    hw = config.handspace_image_width  // config.tile_width
    model_config = CompressedTransformerConfig(
        data_shape = (2, config.max_episode_length, wh, ww, hh, hw),
        causal_dim=1,
        include_tile_embedding=True,
        include_token_embedding=True,
        tile_h=config.tile_height,
        tile_w=config.tile_width,
        token_vocab=2,

        num_blocks=8, # who knows?

        decode_input=False,
        decoder_tokens=1,
        decoder_channels=
            6 + # mode
            # disassemble
            2 + # polarity
            2 + # direction
            config.workspace_map_height +
            config.workspace_map_width +
            # insert
            config.num_classes +
            config.num_colors +
            # pick and place
            2 + # polarity
            2 + # at origin
            config.handspace_map_height +
            config.handspace_map_width +
            config.workspace_map_height +
            config.workspace_map_width +
            # rotate
            2 + # polarity
            2 + # direction
            config.workspace_map_height +
            config.workspace_map_width,
    )
    return CompressedTransformer(model_config).cuda()


def build_optimizer(train_config, model):
    print('-'*80)
    print('Building Optimizer')
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)

    return optimizer


# input and output utilities ===================================================

def observations_to_tensors(train_config, observation, pad):

    wh = train_config.workspace_image_height
    ww = train_config.workspace_image_width
    hh = train_config.handspace_image_height
    hw = train_config.handspace_image_width
    th = train_config.tile_height
    tw = train_config.tile_width

    # make tiles ---------------------------------------------------------------
    wx, wi, w_pad = batch_deduplicate_tiled_seqs(
        observation['workspace_color_render'], pad, tw, th,
        background=102,
    )
    wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
    wi[:,:,0] = 0
    b = wx.shape[1]

    hx, hi, h_pad = batch_deduplicate_tiled_seqs(
        observation['handspace_color_render'], pad, tw, th,
        background=102,
    )
    hi = numpy.insert(hi, (0,1,1), -1, axis=-1)
    hi[:,:,0] = 0

    # move tiles to torch/cuda -------------------------------------------------
    wx = torch.FloatTensor(wx)
    hx = torch.FloatTensor(hx)
    w_pad = torch.LongTensor(w_pad)
    h_pad = torch.LongTensor(h_pad)
    tile_x, tile_pad = cat_padded_seqs(wx, hx, w_pad, h_pad)
    tile_x = default_tile_transform(tile_x).cuda()
    tile_pad = tile_pad.cuda()
    tile_i, _ = cat_padded_seqs(
        torch.LongTensor(wi), torch.LongTensor(hi), w_pad, h_pad)
    tile_i = tile_i.cuda()

    # make tokens --------------------------------------------------------------
    #batch_len = len_hierarchy(batch)
    batch_len = numpy.max(pad)
    token_x = torch.LongTensor(
        observation['reassembly']['reassembling']).cuda()
    token_i = torch.ones((batch_len,b,6), dtype=torch.long) * -1
    token_i[:,:,0] = 0
    token_i[:,:,1] = torch.arange(batch_len).unsqueeze(-1)
    token_i = token_i.cuda()
    token_pad = torch.LongTensor(pad).cuda()

    # make decoder indices and pad ---------------------------------------------
    decoder_i = (
        torch.arange(batch_len).unsqueeze(-1).expand(batch_len, b))
    decoder_pad = torch.LongTensor(pad).cuda()

    return (
        tile_x, tile_i, tile_pad,
        token_x, token_i, token_pad,
        decoder_i, decoder_pad,
    )


def unpack_logits(logits, num_classes, num_colors):
    
    mode_logits = {}
    modes = 6 # disassemble, insert, pick/place, rotate, start reassembly, end
    mode_logits['mode'] = logits[:,:,0:modes]
    
    disassemble_channels = 2+2+64+64
    disassemble_start = modes
    disassemble_stop = disassemble_start + disassemble_channels
    disassemble_logits = logits[:,:,disassemble_start:disassemble_stop]
    mode_logits['disassemble_polarity'] = disassemble_logits[:,:,:2]
    mode_logits['disassemble_direction'] = disassemble_logits[:,:,2:4]
    mode_logits['disassemble_pick_y'] = disassemble_logits[:,:,4:4+64]
    mode_logits['disassemble_pick_x'] = disassemble_logits[:,:,4+64:4+128]
    
    insert_brick_channels = num_classes + num_colors
    insert_brick_start = disassemble_stop
    insert_brick_stop = insert_brick_start + insert_brick_channels
    insert_brick_logits = logits[:,:,insert_brick_start:insert_brick_stop]
    mode_logits['insert_brick_class_id'] = insert_brick_logits[:,:,:num_classes]
    mode_logits['insert_brick_color_id'] = insert_brick_logits[:,:,num_classes:]
    
    pick_and_place_channels = 2+2+24+24+64+64
    pick_and_place_start = insert_brick_stop
    pick_and_place_stop = pick_and_place_start + pick_and_place_channels
    pick_and_place_logits = logits[:,:,pick_and_place_start:pick_and_place_stop]
    mode_logits['pick_and_place_polarity'] = pick_and_place_logits[:,:,:2]
    mode_logits['pick_and_place_at_origin'] = pick_and_place_logits[:,:,2:4]
    mode_logits['pick_and_place_pick_y'] = pick_and_place_logits[:,:,4:4+24]
    mode_logits['pick_and_place_pick_x'] = pick_and_place_logits[:,:,4+24:4+48]
    mode_logits['pick_and_place_place_y'] = pick_and_place_logits[
        :,:,4+48:4+112]
    mode_logits['pick_and_place_place_x'] = pick_and_place_logits[
        :,:,4+112:4+176]
    
    rotate_channels = 2+2+64+64
    rotate_start = pick_and_place_stop
    rotate_stop = rotate_start + rotate_channels
    rotate_logits = logits[:,:,rotate_start:rotate_stop]
    mode_logits['rotate_polarity'] = rotate_logits[:,:,:2]
    mode_logits['rotate_direction'] = rotate_logits[:,:,2:4]
    mode_logits['rotate_pick_x'] = rotate_logits[:,:,4:4+64]
    mode_logits['rotate_pick_y'] = rotate_logits[:,:,4+64:4+128]
    
    return mode_logits


def sample_or_max(logits, mode):
    if mode == 'sample':
        distribution = Categorical(logits=logits)
        return distribution.sample()
    elif mode == 'max':
        return torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError


def logits_to_actions(logits, num_classes, num_colors, mode='sample'):
    s, b = logits.shape[:2]
    logits = unpack_logits(logits, num_classes, num_colors)
    
    action_mode = sample_or_max(logits['mode'].view(-1,6), mode).cpu().numpy()
    print(action_mode)
    
    disassemble_polarity = sample_or_max(
        logits['disassemble_polarity'].view(-1,2), mode).cpu().numpy()
    
    #direction_logits = logits['disassemble_direction'].view(-1,2)
    #direction_distribution = Categorical(logits=direction_logits)
    #direction = direction_distribution.sample().cpu().numpy()
    disassemble_direction = sample_or_max(
        logits['disassemble_direction'].view(-1,2), mode).cpu().numpy()
    
    #pick_y_logits = logits['disassemble_pick_y'].view(-1,64)
    #pick_y_distribution = Categorical(logits=pick_y_logits)
    #pick_y = pick_y_distribution.sample().cpu().numpy()
    disassemble_pick_y = sample_or_max(
        logits['disassemble_pick_y'].view(-1,64), mode).cpu().numpy()
    
    #pick_x_logits = logits['disassemble_pick_x'].view(-1,64)
    #pick_x_distribution = Categorical(logits=pick_x_logits)
    #pick_x = pick_x_distribution.sample().cpu().numpy()
    disassemble_pick_x = sample_or_max(
        logits['disassemble_pick_x'].view(-1,64), mode).cpu().numpy()
    
    disassemble_pick = numpy.stack(
        (disassemble_pick_y, disassemble_pick_x), axis=-1)
    
    num_classes = logits['insert_brick_class_id'].shape[-1]
    insert_brick_class_id = sample_or_max(
        logits['insert_brick_class_id'].view(-1, num_classes),
        mode,
    ).cpu().numpy()
    num_colors = logits['insert_brick_color_id'].shape[-1]
    insert_brick_color_id = sample_or_max(
        logits['insert_brick_color_id'].view(-1, num_colors),
        mode,
    ).cpu().numpy()
    
    pick_and_place_polarity = sample_or_max(
        logits['pick_and_place_polarity'].view(-1, 2), mode).cpu().numpy()
    pick_and_place_at_origin = sample_or_max(
        logits['pick_and_place_at_origin'].view(-1,2), mode).cpu().numpy()
    pick_and_place_pick_y = sample_or_max(
        logits['pick_and_place_pick_y'].view(-1, 24), mode).cpu().numpy()
    pick_and_place_pick_x = sample_or_max(
        logits['pick_and_place_pick_x'].view(-1, 24), mode).cpu().numpy()
    pick_and_place_pick = numpy.stack(
        (pick_and_place_pick_y, pick_and_place_pick_x), axis=-1)
    pick_and_place_place_y = sample_or_max(
        logits['pick_and_place_place_y'].view(-1, 64), mode).cpu().numpy()
    pick_and_place_place_x = sample_or_max(
        logits['pick_and_place_place_x'].view(-1, 64), mode).cpu().numpy()
    pick_and_place_place = numpy.stack(
        (pick_and_place_place_y, pick_and_place_place_x), axis=-1)
    
    rotate_polarity = sample_or_max(
        logits['rotate_polarity'].view(-1, 2), mode).cpu().numpy()
    rotate_direction = sample_or_max(
        logits['rotate_direction'].view(-1, 2), mode).cpu().numpy()
    rotate_pick_y = sample_or_max(
        logits['rotate_pick_y'].view(-1, 64), mode).cpu().numpy()
    rotate_pick_x = sample_or_max(
        logits['rotate_pick_x'].view(-1, 64), mode).cpu().numpy()
    rotate_pick = numpy.stack(
        (rotate_pick_y, rotate_pick_x), axis=-1)
    
    # assemble actions
    actions = []
    for i in range(b):
        action = handspace_reassembly_template_action()
        action['disassembly'] = {
            'activate':action_mode[i] == 0,
            'polarity':disassemble_polarity[i],
            'direction':disassemble_direction[i],
            'pick':disassemble_pick[i],
        }
        action['insert_brick'] = {
            'class_id':(action_mode[i] == 1) * insert_brick_class_id[i],
            'color_id':(action_mode[i] == 1) * insert_brick_color_id[i],
        }
        action['pick_and_place'] = {
            'activate':action_mode[i] == 2,
            'polarity':pick_and_place_polarity[i],
            'pick':pick_and_place_pick[i],
            'place':pick_and_place_place[i],
            'place_at_origin':pick_and_place_at_origin[i],
        }
        action['rotate'] = {
            'activate':action_mode[i] == 3,
            'polarity':rotate_polarity[i],
            'direction':rotate_direction[i],
            'pick':rotate_pick[i],
        }
        action['reassembly'] = {
            'start':action_mode[i] == 4,
            'end':action_mode[i] == 5,
        }
        actions.append(action)
    
    return actions
