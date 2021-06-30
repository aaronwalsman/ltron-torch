#!/usr/bin/env python
import random
import os

import tqdm

import numpy

def make_dataset(num_sequences):
    data_path = './simple_transformer_data'
    os.makedirs(data_path)
    for i in tqdm.tqdm(range(num_sequences)):
        target, brick_maps, actions = make_sequence()
        target_path = os.path.join(data_path, 'target_%06i.npy'%i)
        numpy.save(target_path, target)
        for j, (brick_map, action) in enumerate(zip(brick_maps, actions)):
            brick_map_path = os.path.join(
                data_path, 'observation_%06i_%04i.npy'%(i,j))
            numpy.save(brick_map_path, brick_map)
            
            action_path = os.path.join(
                data_path, 'action_%06i_%04i.npy'%(i,j))
            numpy.save(action_path, numpy.array(action))

def make_sequence():
    target_positions = random_positions()
    start_positions = random_positions()
    
    target_map = positions_to_brick_map(target_positions)
    sequence_maps = [positions_to_brick_map(start_positions)]
    
    actions = []
    for i in range(2):
        ty, tx, to = target_positions[i]
        sy, sx, so = start_positions[i]
        
        if ty != sy or tx != sx:
            actions.append((sy, sx, ty, tx, 0, 0, 1))
            sy = ty
            sx = tx
            start_positions[i] = (sy, sx, so)
            sequence_maps.append(positions_to_brick_map(start_positions))
        
        if to != so:
            actions.append((0, 0, 0, 0, sy, sx, 2))
            so = to
            start_positions[i] = (sy, sx, so)
            sequence_maps.append(positions_to_brick_map(start_positions))
    
    actions.append((0,0,0,0,0,0,0))
    
    return target_map, sequence_maps, actions

def random_positions():
    
    positions = []
    collisions = set()
    for i in range(2):
        while True:
            
            o = random.randint(0,1)
            y = random.randint(0, 6)
            x = random.randint(0, 6)
            new_locations = position_to_map_locations((y,x,o))
            if not any([location in collisions for location in new_locations]):
                break
        
        collisions |= set(new_locations)
        positions.append((y,x,o))
    
    return positions

def position_to_map_locations(position):
    y,x,o = position
    locations = [(y,x)]
    if o == 0:
        locations.append((y+1,x))
    else:
        locations.append((y,x+1))
    
    return locations

def positions_to_brick_map(positions):
    brick_map = numpy.zeros((8,8), dtype=numpy.long)
    
    for i, position in enumerate(positions, start=1):
        map_locations = position_to_map_locations(position)
        for y,x in map_locations:
            brick_map[y,x] = i
    
    return brick_map

if __name__ == '__main__':
    make_dataset(20000)
