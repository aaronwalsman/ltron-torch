import random
import collections
import os

import tqdm

import numpy

from ltron.gym.ltron_env import LtronEnv, async_ltron
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
    ColorRenderComponent,
    SnapRenderComponent,
)
from ltron.gym.components.viewpoint import RandomizedAzimuthalViewpointComponent

def snap_four_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    width=256,
    height=256,
    segmentation_width=64,
    segmentation_height=64,
    train=True,
):
    
    components = collections.OrderedDict()
    
    # dataset
    if train:
        reset_mode='uniform'
    else:
        reset_mode='single_pass'
    components['dataset'] = DatasetPathComponent(
        dataset,
        split=split,
        subset=subset,
        rank=rank,
        size=size,
        reset_mode=reset_mode,
        observe_episode_id=False,
    )
    
    # scene
    components['scene'] = SceneComponent(
        path_component=components['dataset'],
        renderable=True,
    )
    
    # max episode length
    components['episode_length'] = MaxEpisodeLengthComponent(1)
    
    # viewpoint
    components['viewpoint'] = RandomizedAzimuthalViewpointComponent(
        components['scene'],
        aspect_ratio=(width/height),
        randomize_frequency='reset',
    )
    
    # color render
    components['color_render'] = ColorRenderComponent(
        width,
        height,
        components['scene'],
        anti_alias=True,
    )
    
    # snap render
    components['snap_render_pos'] = SnapRenderComponent(
        segmentation_width,
        segmentation_height,
        components['scene'],
        polarity='+',
    )
    
    components['snap_render_neg'] = SnapRenderComponent(
        segmentation_width,
        segmentation_height,
        components['scene'],
        polarity='-',
    )
    
    # build/return the env
    env = LtronEnv(components, print_traceback=True)
    return env

def static_dataset(
    directory,
    start=0,
    split='train',
    num_processes=8,
    num_examples=60000,
):
    
    random.seed(1234)
    numpy.random.seed(1234)
    
    import PIL.Image as Image
    import splendor.masks as masks
    
    width = 256
    height = 256
    segmentation_width = 64
    segmentation_height = 64
    
    env = async_ltron(
        num_processes,
        snap_four_env,
        'snap_four',
        split,
        subset=None,
        width=width,
        height=height,
        segmentation_width=segmentation_width,
        segmentation_height=segmentation_height,
        train=True,
    )
    
    observation = env.reset()
    
    for i in tqdm.tqdm(range(num_examples//num_processes)):
        for j in range(num_processes):
            frame_index = i * num_processes + j
            
            color_path = os.path.join(
                directory, 'data', 'color_%06i.png'%frame_index)
            color_image = Image.fromarray(observation['color_render'][j])
            color_image.save(color_path)
            
            pos_snaps = observation['snap_render_pos'][j]
            pos_snap_path = os.path.join(
                directory, 'data', 'pos_snaps_%06i.npy'%frame_index)
            numpy.save(pos_snap_path, pos_snaps)
            
            neg_snaps = observation['snap_render_neg'][j]
            neg_snap_path = os.path.join(
                directory, 'data', 'neg_snaps_%06i.npy'%frame_index)
            numpy.save(neg_snap_path, neg_snaps)
            
            #pos_instance_ids = observation['snap_render_pos'][j,:,:,0]
            #pos_snap_ids = observation['snap_render_pos'][j,:,:,1]
            
            #neg_instance_ids = observation['snap_render_neg'][j,:,:,0]
            #neg_snap_ids = observation['snap_render_neg'][j,:,:,1]
            
            #pick_targets = (neg_instance_ids == 2) & (neg_snap_ids == 0)
            #pick_path = os.path.join(
            #    directory, 'data', 'pick_%06i.npy'%frame_index)
            #numpy.save(pick_path, pick_targets)
            
            #place_targets = (pos_instance_ids == 1) & (pos_snap_ids == 8)
            #place_path = os.path.join(
            #    directory, 'data', 'place_%06i.npy'%frame_index)
            #numpy.save(place_path, place_targets)
        
        actions = [{} for _ in range(num_processes)]
        observation, reward, terminal, info = env.step(actions)

if __name__ == '__main__':
    static_dataset('.')
