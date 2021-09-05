import random

import numpy

from ltron.gym.reassembly_env import handspace_reassembly_template_action
from ltron.hierarchy import index_hierarchy

class ReassemblyExpert:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.orders = [None for _ in range(batch_size)]
        self.eagle = False
    
    def __call__(self, observation, terminal):
        
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
                #print('-'*80)
                #print('target')
                #print(r_obs['target_configuration']['pose'])
                #print('workspace')
                #print(r_obs['workspace_configuration']['pose'])
                
                reverse_order = list(reversed(self.orders[i]))
                num_bricks = numpy.sum(
                    r_obs['workspace_configuration']['class'] != 0)
                
                # is it time to pick add a new brick to the scene?
                if num_bricks == len(reverse_order):
                    act['reassembly']['end'] = 1
                    actions.append(act)
                    continue
                elif num_bricks <= 1:
                    needs_rotate = False
                else:
                    needs_rotate = False
                
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
                            act['pick_and_place']['pick'] = (pick_y, pick_x)
                            act['pick_and_place']['place_at_origin'] = True
                            actions.append(act)
                            continue
                        
                        else:
                            # get a list of edges that are supposed to exist
                            # between this brick and the rest of the scene
                            edge_ids = r_obs['target_configuration']['edges'][
                                'edge_index']
                            hand_edges = edge_ids[0] == instance_id
                            #source_edges = edge_ids[0] == instance_id
                            #dest_edges = edge_ids[1] == instance_id
                            #hand_edges = source_edges | dest_edges
                            
                            existing_brick_ids = reverse_order[:num_bricks]
                            #print(instance_id, existing_brick_ids)
                            # looks good
                            
                            expanded_edge_ids = edge_ids.reshape(
                                (*edge_ids.shape, 1))
                            scene_edges = numpy.logical_or.reduce(
                                expanded_edge_ids[1] == existing_brick_ids,
                                axis=1)
                            #source_edges = numpy.logical_or.reduce(
                            #    expanded_edge_ids[0] == existing_brick_ids,
                            #    axis=1)
                            #dest_edges = numpy.logical_or.reduce(
                            #    expanded_edge_ids[1] == existing_brick_ids,
                            #    axis=1)
                            #scene_edges = source_edges | dest_edges
                            connecting_edges = hand_edges & scene_edges
                            
                            connecting_edges = edge_ids[:,connecting_edges]
                            
                            r = random.randint(0, connecting_edges.shape[1]-1)
                            hi, si, hs, ss = connecting_edges[:,r]
                            hand_hi = 1
                            scene_si = reverse_order.index(si)+1
                            #print(hi, si, hs, ss)
                            # looks good
                            
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
                            #polarity = numpy.concatenate((
                            #    numpy.ones(pos_y.shape[0], dtype=numpy.long),
                            #    numpy.zeros(neg_y.shape[0], dtype=numpy.long)
                            #))
                            yx = numpy.stack((
                                numpy.concatenate((pos_y, neg_y)),
                                numpy.concatenate((pos_x, neg_x)),
                            ))
                            #print(yx)
                            
                            r = random.randint(0, yx.shape[1]-1)
                            place_y, place_x = yx[:,r]
                            
                            act['pick_and_place']['activate'] = True
                            act['pick_and_place']['polarity'] = polarity
                            act['pick_and_place']['pick'] = (pick_y, pick_x)
                            act['pick_and_place']['place'] = (place_y, place_x)
                            actions.append(act)
                            continue
        
        return actions

def generate_order(observation):
    return [4,3,2,1]
