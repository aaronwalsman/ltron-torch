import math
import collections

from ltron.gym.ltron_env import LtronEnv
from ltron.gym.components.scene import SceneComponent
from ltron.gym.components.episode import MaxEpisodeLengthComponent
from ltron.gym.components.dataset import DatasetPathComponent
from ltron.gym.components.labels import InstanceListComponent
from ltron.gym.components.spatial_info import InstancePoseComponent
from ltron.gym.components.viewpoint import (
    RandomizedAzimuthalViewpointComponent,
    ControlledAzimuthalViewpointComponent,
)
from ltron.gym.components.colors import RandomizeColorsComponent
from ltron.gym.components.visibility import PixelVisibilityComponent
from ltron.gym.components.render import (
    ColorRenderComponent, SegmentationRenderComponent)
from ltron.gym.components.dense_map import DenseMapComponent

def pose_estimation_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    width=256,
    height=256,
    segmentation_width=64,
    segmentation_height=64,
    controlled_viewpoint=False,
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
    dataset_info = components['dataset'].dataset_info
    
    # scene
    components['scene'] = SceneComponent(
        path_component=components['dataset'],
        renderable=True,
    )
    
    # max episode length
    components['episode_length'] = MaxEpisodeLengthComponent(
        dataset_info['max_instances_per_scene'])
    
    # color randomization
    if train:
        components['color_randomization'] = RandomizeColorsComponent(
            dataset_info['all_colors'],
            components['scene'],
            randomize_frequency='step',
        )
    
    # segmentation_component
    segmentation_component = SegmentationRenderComponent(
        segmentation_width,
        segmentation_height,
        scene_component=components['scene'],
    )
    # do not register this yet, we want it to come after the visiblity component
    
    # visibility action space
    components['visibility'] = PixelVisibilityComponent(
        width=width,
        height=height,
        scene_component=components['scene'],
        segmentation_component=segmentation_component,
        terminate_when_all_hidden=True
    )
    
    # viewpoint
    if controlled_viewpoint:
        components['viewpoint'] = ControlledAzimuthalViewpointComponent(
            components['scene'],
            azimuth_steps=24,
            elevation_range=(math.radians(-45), math.radians(45)),
            elevation_steps=7,
            distance_range=(200,350),
            distance_steps=4,
            start_position = 'uniform',
        )
    else:
        components['viewpoint'] = RandomizedAzimuthalViewpointComponent(
            components['scene'],
            distance=(0.8, 1.2),
            aspect_ratio=(width/height),
            randomize_frequency='reset'
        )
    
    # color render
    components['color_render'] = ColorRenderComponent(
        width,
        height,
        components['scene'],
        anti_alias=True,
    )
    
    # segmentation render
    components['segmentation_render'] = segmentation_component
    
    # instance labels
    num_classes = max(dataset_info['class_ids'].values())+1
    components['class_labels'] = InstanceListComponent(
        num_classes,
        dataset_info['max_instances_per_scene'],
        components['dataset'],
        components['scene'],
    )
    
    # dense instance labels
    components['dense_class_labels'] = DenseMapComponent(
        components['class_labels'],
        components['segmentation_render'],
        instance_data_key = ('label',),
    )
    
    # instance poses
    components['pose_labels'] = InstancePoseComponent(
        dataset_info['max_instances_per_scene'],
        components['scene'],
    )
    
    # dense instance poses
    components['dense_pose_labels'] = DenseMapComponent(
        components['pose_labels'],
        components['segmentation_render'],
    )
    
    # build the env
    env = LtronEnv(components, print_traceback=True)
    
    return env
