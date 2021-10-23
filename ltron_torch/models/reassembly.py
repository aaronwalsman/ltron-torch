import random

import numpy

import torch
from torch.distributions import Categorical

from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.dataset.paths import get_dataset_info
from ltron.hierarchy import index_hierarchy
from ltron.bricks.brick_type import BrickType

from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.gym_tensor import gym_space_to_tensors, default_tile_transform
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)

# ppo model ====================================================================

class ReassemblyPPOModel: #(ActorCriticModel[CategoricalDistr]):
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
    def __init__(self, batch_size, class_ids):
        self.batch_size = batch_size
        self.orders = [None for _ in range(batch_size)]
        self.class_ids = class_ids
        self.class_names = {value:key for key, value in class_ids.items()}
    
    def generate_order(self, obs):
        #return [4,3,2,1] # TMP 000001
        #return [4,3,1,2] # TMP 000002
        pass
    
    def select_from_pos_neg_maps(self, pos_map, neg_map):
        pos_y, pos_x = numpy.where(pos_map)
        neg_y, neg_x = numpy.where(neg_map)
        y = numpy.concatenate((pos_y, neg_y))
        x = numpy.concatenate((pos_x, neg_x))
        p = numpy.concatenate((
            numpy.ones(pos_y.shape, dtype=numpy.long),
            numpy.zeros(neg_y.shape, dtype=numpy.long),
        ))
        r = random.randint(0, y.shape[0]-1)
        pick_y = y[r]
        pick_x = x[r]
        pick_p = p[r]
        return pick_y, pick_x, pick_p
    
    def disassembly_action(self, obs, order):
        action = handspace_reassembly_template_action()
        workspace_config = obs['reassembly']['workspace_configuration']
        
        # if there is still stuff to remove, remove it
        if numpy.any(workspace_config['class']):
            # figure out the next instance to remove
            for instance_id in order:
                if workspace_config['class'][instance_id]:
                    next_instance = instance_id
            
            # pick a snap pixel belonging to the chosen instance and remove it
            pos_snaps = obs['workspace_pos_snap_render']
            neg_snaps = obs['workspace_neg_snap_render']
            pick_y, pick_x, polarity = self.select_from_pos_neg_maps(
                pos_snaps[:,:,0] == next_instance,
                neg_snaps[:,:,0] == next_instance,
            )
            action['disassembly']['activate'] = True
            action['disassembly']['polarity'] = polarity
            action['disassembly']['direction'] = 0 # TMP
            action['disassembly']['pick'] = numpy.array(
                (pick_y, pick_x), dtype=numpy.long)
            return action
        
        # if there is nothing left to remove, switch to reassembly
        else:
            action['reassembly']['start'] = 1
            return action
    
    def reassembly_action(self, obs, order):
        action = handspace_reassembly_template_action()
        # if the reward is 1, then we are done
        if reward[i] == 1.:
            action['reassembly']['end'] = 1
            return action
    
    def __call_new__(self, observation, terminal, reward):
        # What mode are we in?
        
        # If disassembly, disassemble.
        
        # If reassembly, is the current workspace in-progress correct?
        # (Is there an assignment of workspace bricks to target bricks such
        # that all neighborhood connections have the correct pose?)
        
        # If the workspace is in-progress correct, pick a new brick and add it
        
        # If the workspace is not in-progress correct, is there only one brick
        # out of place?
        
        # If so move it into place.
        
        # If not, do something drastic.  Or just fix one, then the other?
        
        ######
        # Or maybe the real question is: if ANYTHING is out of place, is it
        # possible to move one of the bricks into place such that it's on a
        # correct assembly path (one that doesn't cause collision problems for
        # later bricks)?  If so do it.  If not, start removing stuff.
        pass
    
    def __call__(self, observation, terminal, reward):
        
        actions = []
        for i, t in enumerate(terminal):
            obs = index_hierarchy(observation, i)
            workspace_config = obs['reassembly']['workspace_configuration']
            handspace_config = obs['reassembly']['handspace_configuration']
            target_config = obs['reassembly']['target_configuration']
            
            act = handspace_reassembly_template_action()
            
            # if terminal, generate a new order
            if t:
                self.orders[i] = self.generate_order(obs)
            
            reassembling = obs['reassembly']['reassembling']
            if not reassembling:
                
                act = self.disassembly_action(obs, self.orders[i])
                actions.append(act)
                '''
                # if there are no objects left, switch to reassembly
                if not numpy.any(workspace_config['class']):
                    act['reassembly']['start'] = 1
                    actions.append(act)
                    continue
                
                # figure out the next item to remove
                for instance_id in self.orders[i]:
                    if workspace_config['class'][instance_id]:
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
                '''
            
            else:
                reverse_order = list(reversed(self.orders[i]))
                num_bricks = numpy.sum(workspace_config['class'] != 0)
                
                # if the reward is 1, then we are done
                if reward[i] == 1.:
                    act['reassembly']['end'] = 1
                    actions.append(act)
                    continue
                
                # if there are 0 or 1 bricks, we don't need to rotate yet
                if num_bricks <= 1:
                    needs_rotate = False
                
                # if there are two or more bricks, check if we need to rotate
                # this is done by checking if the most recently added brick
                # has a correct relative relationship with all other bricks in
                # the scene that it's supposed to be connected to
                else:
                    #instance_id = reverse_order[num_bricks]
                    last_placed_brick = numpy.max(numpy.where(
                        workspace_config['class'])[0])
                    last_target_id = reverse_order[num_bricks-1]
                    edge_ids = target_config['edges']
                    last_edges = edge_ids[0] == last_target_id
                    
                    prev_target_ids = reverse_order[:num_bricks-1]
                    expanded_edge_ids = edge_ids.reshape(
                        (*edge_ids.shape, 1))
                    prev_edges = numpy.logical_or.reduce(
                        expanded_edge_ids[1] == prev_target_ids, axis=1)
                    connected_edges = last_edges & prev_edges
                    connected_target_ids = edge_ids[:,connected_edges][1]
                    connected_target_ids = numpy.unique(connected_target_ids)
                    
                    target_poses = target_config['pose']
                    last_target_transform = target_poses[last_target_id]
                    prev_target_transforms = target_poses[connected_target_ids]
                    target_offsets = [
                        numpy.linalg.inv(last_target_transform) @
                        prev_target_transform
                        for prev_target_transform in prev_target_transforms
                    ]
                    
                    workspace_poses = workspace_config['pose']
                    last_workspace_transform = workspace_poses[num_bricks]
                    connected_workspace_ids = [
                        reverse_order.index(target_id)+1
                        for target_id in connected_target_ids
                    ]
                    prev_workspace_transforms = workspace_poses[
                        connected_workspace_ids]
                    workspace_offsets = [
                        numpy.linalg.inv(last_workspace_transform) @
                        prev_work_transform
                        for prev_work_transform in prev_workspace_transforms
                    ]
                    
                    matching_offsets = [
                        numpy.allclose(workspace_offset, target_offset)
                        for workspace_offset, target_offset
                        in zip(workspace_offsets, target_offsets)
                    ]
                    
                    needs_rotate = not all(matching_offsets)
                
                if needs_rotate:
                    # first find the right snap
                    # it should be one connected to a correct neighbor
                    target_edges = target_config['edges']
                    workspace_edges = workspace_config['edges']
                    recent_edges = workspace_edges[0] == num_bricks
                    recent_edges = workspace_edges[:,recent_edges]
                    
                    target_edges = recent_edges.copy()
                    for i in range(target_edges.shape[1]):
                        target_edges[0,i] = reverse_order[target_edges[0,i]-1]
                        target_edges[1,i] = reverse_order[target_edges[1,i]-1]
                    
                    target_edges = target_edges.reshape(4, -1, 1)
                    matching_edges = (
                        target_edges == target_config['edges'].reshape(4,1,-1))
                    all_match = numpy.logical_and.reduce(matching_edges, axis=0)
                    any_match = numpy.logical_or.reduce(all_match, axis=1)
                    assert numpy.sum(any_match) == 1, (
                        'precisely one edge should exist on the current model')
                    matching_edge = numpy.where(any_match)[0][0]
                    picked_snap = recent_edges[2, matching_edge]
                    
                    pos_snaps = obs['workspace_pos_snap_render']
                    neg_snaps = obs['workspace_neg_snap_render']
                    #pos_y, pos_x = numpy.where(pos_snaps[:,:,0] != 0)
                    #neg_y, neg_x = numpy.where(neg_snaps[:,:,0] != 0)
                    pos_y, pos_x = numpy.where(
                        (pos_snaps[:,:,0] == num_bricks) &
                        (pos_snaps[:,:,1] == picked_snap)
                    )
                    neg_y, neg_x = numpy.where(
                        (neg_snaps[:,:,0] == num_bricks) &
                        (neg_snaps[:,:,1] == picked_snap)
                    )
                    
                    ys = numpy.concatenate((pos_y, neg_y))
                    xs = numpy.concatenate((pos_x, neg_x))
                    polarities = numpy.concatenate((
                        numpy.ones(pos_y.shape[0], dtype=numpy.long),
                        numpy.zeros(neg_y.shape[0], dtype=numpy.long),
                    ))
                    
                    r = random.randint(0, ys.shape[0]-1)
                    y = ys[r]
                    x = xs[r]
                    p = polarities[r]
                    
                    # then figure out which direction to rotate
                    brick_class = workspace_config['class'][num_bricks]
                    brick_type = BrickType(
                        self.class_names[brick_class])
                    instance_transform = workspace_config['pose'][num_bricks]
                    snap_transform = brick_type.snaps[picked_snap].transform
                    inv_snap_transform = numpy.linalg.inv(snap_transform)
                    instance_snap_transform = (
                        instance_transform @
                        snap_transform
                    )
                    #inv_instance_snap_transform = numpy.linalg.inv(
                    #    instance_snap_transform)
                    connected_brick = recent_edges[1, matching_edge]
                    connected_transform = workspace_config['pose'][
                        connected_brick]
                    inv_connected_transform = numpy.linalg.inv(
                        connected_transform)
                    
                    ry0 = numpy.array([
                        [ 0, 0, 1, 0],
                        [ 0, 1, 0, 0],
                        [-1, 0, 0, 0],
                        [ 0, 0, 0, 1],
                    ])
                    ry1 = numpy.array([
                        [ 0, 0,-1, 0],
                        [ 0, 1, 0, 0],
                        [ 1, 0, 0, 0],
                        [ 0, 0, 0, 1],
                    ])
                    ry2 = numpy.array([
                        [-1, 0, 0, 0],
                        [ 0, 1, 0, 0],
                        [ 0, 0,-1, 0],
                        [ 0, 0, 0, 1],
                    ])
                    offset_to_r0 = (
                        inv_connected_transform @
                        instance_snap_transform @
                        ry0 @
                        inv_snap_transform
                        #inv_instance_snap_transform @
                        #instance_transform
                    )
                    offset_to_r1 = (
                        inv_connected_transform @
                        instance_snap_transform @
                        ry1 @
                        inv_snap_transform
                        #inv_instance_snap_transform @
                        #instance_transform
                    )
                    offset_to_r2 = (
                        inv_connected_transform @
                        instance_snap_transform @
                        ry2 @
                        inv_snap_transform
                        #inv_instance_snap_transform @
                        #instance_transform
                    )
                    
                    target_connected_brick = reverse_order[connected_brick-1]
                    connected_brick_pose = target_config['pose'][
                        target_connected_brick]
                    
                    target_offset = (
                        numpy.linalg.inv(connected_brick_pose) @
                        target_config['pose'][reverse_order[num_bricks-1]]
                    )
                    
                    if numpy.allclose(offset_to_r0, target_offset):
                        direction = 1
                    elif numpy.allclose(offset_to_r1, target_offset):
                        direction = 0
                    elif numpy.allclose(offset_to_r2, target_offset):
                        direction = random.randint(0,1)
                    else:
                        assert False, 'this should never happen (yet)'
                    
                    act['rotate']['activate'] = True
                    act['rotate']['pick'] = numpy.array([y,x])
                    act['rotate']['polarity'] = p
                    act['rotate']['direction'] = direction
                    actions.append(act)
                    continue
                
                else:
                    target_id = reverse_order[num_bricks]
                    class_id = target_config['class'][target_id]
                    color_id = target_config['color'][target_id]
                    
                    # pick up the correct brick if necessary
                    if handspace_config['class'][1] != class_id:
                        act['insert_brick']['class_id'] = class_id
                        act['insert_brick']['color_id'] = color_id
                        actions.append(act)
                        continue
                    
                    # place the brick if it is already picked up
                    else:
                        # if there is nothing here yet add the first instance
                        # using the place_at_origin flag
                        if num_bricks == 0:
                            # Need to pick a snap that's
                            # y-up so the model ends up aligned properly
                            # I need the brick type for this though.
                            brick_class = handspace_config['class'][1]
                            brick_type = BrickType(
                                self.class_names[brick_class])
                            snap_ys = [
                                snap.transform[:3,1]
                                for snap in brick_type.snaps
                            ]
                            pose_y = target_config['pose'][target_id][:3,1]
                            alignment = [-snap_y @ pose_y for snap_y in snap_ys]
                            aligned_snaps = [
                                i for i, a in enumerate(alignment) if a > 0.99]
                            picked_snap = random.choice(aligned_snaps)
                            
                            pos_snaps = obs['handspace_pos_snap_render']
                            neg_snaps = obs['handspace_neg_snap_render']
                            #pos_y, pos_x = numpy.where(pos_snaps[:,:,0] != 0)
                            #neg_y, neg_x = numpy.where(neg_snaps[:,:,0] != 0)
                            pos_y, pos_x = numpy.where(
                                (pos_snaps[:,:,0] != 0) &
                                (pos_snaps[:,:,1] == picked_snap))
                            neg_y, neg_x = numpy.where(
                                (neg_snaps[:,:,0] != 0) &
                                (neg_snaps[:,:,1] == picked_snap))
                            
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
                            edge_ids = target_config['edges']
                            hand_edges = edge_ids[0] == target_id
                            
                            existing_brick_ids = reverse_order[:num_bricks]
                            
                            expanded_edge_ids = edge_ids.reshape(
                                (*edge_ids.shape, 1))
                            scene_edges = numpy.logical_or.reduce(
                                expanded_edge_ids[1] == existing_brick_ids,
                                axis=1)
                            connecting_edges = hand_edges & scene_edges
                            
                            connecting_edges = edge_ids[:,connecting_edges]
                            
                            r = random.randint(
                                0, connecting_edges.shape[1]-1)
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
            8 + # mode
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
    modes = 8 # remove, insert, pick/place, r1, r2, r3, start reassembly, end
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
    # OUT OF DATE
    s, b = logits.shape[:2]
    logits = unpack_logits(logits, num_classes, num_colors)
    
    action_mode = sample_or_max(logits['mode'].view(-1,8), mode).cpu().numpy()
    
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
