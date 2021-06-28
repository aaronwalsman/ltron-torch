import random
import math
import collections
import os
import json

import tqdm

import numpy

from pyquaternion import Quaternion

import ltron.settings as settings
from ltron.hierarchy import index_hierarchy
from ltron.gym.ltron_env import LtronEnv, async_ltron
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.render import (
    ColorRenderComponent,
    SnapRenderComponent,
)
from ltron.gym.components.viewpoint import (
    RandomizedAzimuthalViewpointComponent,
    CopyViewpointComponent,
)
from ltron.gym.components.labels import InstanceGraphComponent
from ltron.gym.components.spatial_info import InstancePoseComponent
from ltron.gym.components.manipulation.symbolic import (
    PickAndPlaceSymbolic,
    VectorOffsetSymbolic,
)

def conditional_snap_two_env(
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
        observe_episode_id=True,
    )
    
    # scene
    components['scene_x'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=('x',),
        renderable=True,
    )
    components['scene_y'] = SceneComponent(
        dataset_component=components['dataset'],
        path_location=('y',),
        renderable=True,
    )
    
    # max episode length
    components['episode_length'] = MaxEpisodeLengthComponent(7)
    
    # viewpoint
    components['viewpoint_x'] = RandomizedAzimuthalViewpointComponent(
        components['scene_x'],
        aspect_ratio=(width/height),
        randomize_frequency='reset',
    )
    components['copy_viewpoint'] = CopyViewpointComponent(
        components['scene_x'],
        components['scene_y'],
    )
    
    # pick-and-place
    components['pick_and_place'] = PickAndPlaceSymbolic(
        components['scene_x'],
        max_instances=3,
        max_snaps=9,
    )
    components['vector_offset'] = VectorOffsetSymbolic(
        components['scene_x'],
        max_instances=3,
        max_snaps=9,
        space='local',
    )
    
    # color render
    components['color_render_x'] = ColorRenderComponent(
        width,
        height,
        components['scene_x'],
        anti_alias=True,
    )
    components['color_render_y'] = ColorRenderComponent(
        width,
        height,
        components['scene_y'],
        anti_alias=True,
    )
    
    # snap render
    components['snap_render_pos'] = SnapRenderComponent(
        segmentation_width,
        segmentation_height,
        components['scene_x'],
        polarity='+',
    )
    
    components['snap_render_neg'] = SnapRenderComponent(
        segmentation_width,
        segmentation_height,
        components['scene_x'],
        polarity='-',
    )
    
    # graph
    components['graph_x'] = InstanceGraphComponent(
        num_classes=2,
        max_instances=3,
        max_snaps=9,
        max_edges=4,
        dataset_component=components['dataset'],
        scene_component=components['scene_x'],
    )
    components['graph_y'] = InstanceGraphComponent(
        num_classes=2,
        max_instances=3,
        max_snaps=9,
        max_edges=4,
        dataset_component=components['dataset'],
        scene_component=components['scene_y'],
    )
    
    # pose
    components['pose_x'] = InstancePoseComponent(3, components['scene_x'])
    components['pose_y'] = InstancePoseComponent(3, components['scene_y'])
    
    # build/return the env
    env = LtronEnv(components, print_traceback=True)
    return env

def static_dataset(
    directory,
    start=0,
    split='train',
    num_processes=8,
    num_sequences=60000,
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
        conditional_snap_two_env,
        'conditional_snap_two',
        split,
        subset=None,
        width=width,
        height=height,
        segmentation_width=segmentation_width,
        segmentation_height=segmentation_height,
        train=True,
    )
    
    observation = env.reset()
    
    for i in tqdm.tqdm(range(num_sequences*7//num_processes)):
        actions = [{} for _ in range(num_processes)]
        for j in range(num_processes):
            episode_id = observation['dataset']['episode_id'][j]
            episode_id = episode_id * num_processes + j
            step_id = observation['episode_length'][j]
            
            color_x_path = os.path.join(
                directory,
                'frames',
                'color_x_%06i_%04i.png'%(episode_id, step_id),
            )
            color_image_x = Image.fromarray(observation['color_render_x'][j])
            color_image_x.save(color_x_path)
            
            snap_pos_path = os.path.join(
                directory,
                'frames',
                'snap_pos_%06i_%04i.npz'%(episode_id, step_id),
            )
            snap_render_pos = observation['snap_render_pos'][j]
            numpy.savez_compressed(snap_pos_path, snap_render_pos)
            
            snap_neg_path = os.path.join(
                directory,
                'frames',
                'snap_neg_%06i_%04i.npz'%(episode_id, step_id),
            )
            snap_render_neg = observation['snap_render_neg'][j]
            numpy.savez_compressed(snap_neg_path, snap_render_neg)
            
            target_graph = index_hierarchy(observation['graph_y'], j)
            current_graph = index_hierarchy(observation['graph_x'], j)
            
            if step_id == 0:
                color_y_path = os.path.join(
                    directory,
                    'frames',
                    'color_y_%06i.png'%(episode_id),
                )
                color_image_y = Image.fromarray(
                    observation['color_render_y'][j])
                color_image_y.save(color_y_path)
                
                target_graph_path = os.path.join(
                    directory,
                    'frames',
                    'target_graph_%06i.npy'%(episode_id),
                )
                numpy.save(target_graph_path, target_graph)
                
                poses_y = index_hierarchy(observation['pose_y'], j)
                pose_y_path = os.path.join(
                    directory,
                    'frames',
                    'poses_y_%06i.npy'%(episode_id),
                )
                numpy.save(pose_y_path, poses_y)
            
            slope_target_poses = observation['pose_y'][j]
            slope_current_poses = observation['pose_x'][j]
            
            instance_labels = observation['graph_y']['instances']['label'][j]
            slope_instances = numpy.where(instance_labels == 2)[0]
            slope_target_distances = slope_target_poses[slope_instances][:,2,3]
            order = sorted(zip(slope_target_distances, slope_instances))
            
            target_edges = target_graph['edges']['edge_index']
            target_edge_set = set(
                tuple(target_edges[:,k])
                for k in range(target_edges.shape[1])
            )
            
            current_edges = current_graph['edges']['edge_index']
            current_edge_set = set(
                tuple(current_edges[:,k])
                for k in range(current_edges.shape[1])
            )
            
            for _, instance_id in order:
                target_pose = slope_target_poses[instance_id]
                current_pose = slope_current_poses[instance_id]
                
                # if the poses already match, move on to the next object
                if numpy.allclose(target_pose, current_pose):
                    continue
                
                # look for matching snaps, if a snap matches, we need to rotate
                current_instance_edges = set(
                    edge for edge in current_edge_set
                    if edge[0] == instance_id or edge[1] == instance_id)
                target_instance_edges = set(
                    edge for edge in target_edge_set
                    if edge[0] == instance_id or edge[1] == instance_id)
                matching_edges = current_instance_edges & target_instance_edges
                if len(matching_edges):
                    # rotate action goes here
                    actions[j]['pick_and_place'] = {
                        'pick':(0,0),
                        'place':(0,0),
                    }
                    i_a, i_b, s_a, s_b = list(matching_edges)[0]
                    rotate_pos = Quaternion(axis=(0,1,0), angle=math.pi/2.)
                    result_pos = current_pose @ rotate_pos.transformation_matrix
                    offset_pos = numpy.linalg.inv(target_pose) @ result_pos
                    rotate_neg = Quaternion(axis=(0,-1,0), angle=math.pi/2.)
                    result_neg = current_pose @ rotate_neg.transformation_matrix
                    offset_neg = numpy.linalg.inv(target_pose) @ result_neg
                    if (numpy.trace(offset_pos[:3,:3]) >
                        numpy.trace(offset_neg[:3,:3])
                    ):
                        direction = [0,1,0]
                    else:
                        direction = [0,-1,0]
                    actions[j]['vector_offset'] = {
                        'pick':(i_b, s_b),
                        'direction':direction,
                        'motion':1
                    }
                    break
                else:
                    # if no edges match, then we need to drag-and-drop
                    picked_edge = random.choice(list(target_instance_edges))
                    instance_a, instance_b, snap_a, snap_b = picked_edge
                    actions[j]['pick_and_place'] = {
                        'pick':(instance_b, snap_b),
                        'place':(instance_a, snap_a)
                    }
                    actions[j]['vector_offset'] = {
                        'pick':(0, 0),
                        'direction':(0,1,0),
                        'motion':0,
                    }
                    break
            else:
                actions[j]['pick_and_place'] = {'pick':(0,0), 'place':(0,0)}
                actions[j]['vector_offset'] = {
                    'pick':(0,0), 'direction':(0,1,0), 'motion':0}
            
            action_path = os.path.join(
                directory,
                'frames',
                'action_%06i_%04i.npy'%(episode_id, step_id),
            )
            numpy.save(action_path, actions[j])
        
        observation, reward, terminal, info = env.step(actions)
        
    split_spot = num_sequences * 7 // 8
    
    name = 'conditional_snap_two_frames'
    dataset_info = {
        'splits':{
            'all':{
                'color_x':
                    '{%s}/frames/color_x_*_0000.png'%(name),
                'color_y':
                    '{%s}/frames/color_y_*.png'%(name),
                'snap_pos':
                    '{%s}/frames/snap_pos_*_0000.npz'%(name),
                'snap_neg':
                    '{%s}/frames/snap_neg_*_0000.npz'%(name),
                'action':
                    '{%s}/frames/action_*_0000.npy'%(name),
                'target_graph':
                    '{%s}/frames/target_graph_*.npy'%(name),
                'poses_y':
                    '{%s}/frames/poses_y_*.npy'%(name),
            }
            'train':{
                'color_x':
                    '{%s}/frames/color_x_*_0000.png[:%i]'%(name, split_spot),
                'color_y':
                    '{%s}/frames/color_y_*.png[:%i]'%(name, split_spot),
                'snap_pos':
                    '{%s}/frames/snap_pos_*_0000.npz[:%i]'%(name, split_spot),
                'snap_neg':
                    '{%s}/frames/snap_neg_*_0000.npz[:%i]'%(name, split_spot),
                'action':
                    '{%s}/frames/action_*_0000.npy[:%i]'%(name, split_spot),
                'target_graph':
                    '{%s}/frames/target_graph_*.npy[:%i]'%(name, split_spot),
                'poses_y':
                    '{%s}/frames/poses_y_*.npy[:%i]'%(name, split_spot),
            }
            'train':{
                'color_x':
                    '{%s}/frames/color_x_*_0000.png[%i:]'%(name, split_spot),
                'color_y':
                    '{%s}/frames/color_y_*.png[%i:]'%(name, split_spot),
                'snap_pos':
                    '{%s}/frames/snap_pos_*_0000.npz[%i:]'%(name, split_spot),
                'snap_neg':
                    '{%s}/frames/snap_neg_*_0000.npz[%i:]'%(name, split_spot),
                'action':
                    '{%s}/frames/action_*_0000.npy[%i:]'%(name, split_spot),
                'target_graph':
                    '{%s}/frames/target_graph_*.npy[%i:]'%(name, split_spot),
                'poses_y':
                    '{%s}/frames/poses_y_*.npy[%i:]'%(name, split_spot),
            }
        }
    }
    
    dataset_path = os.path.join(directory, 'conditional_snap_two_frames.json')
    with open(dataset_path, 'w') as f:
        json.dump(dataset_info, f)

if __name__ == '__main__':
    collection_dir = os.path.join(
        settings.paths['collections'], 'conditional_snap_two_frames')
    assert not os.path.exists(collection_dir)
    frames_dir = os.path.join(collection_dir, 'frames')
    os.makedirs(frames_dir)
    static_dataset(collection_dir, num_sequences = 16384)
