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

def camera_test_env(
    dataset,
    split,
    subset=None,
    rank=0,
    size=1,
    width=256,
    height=256,
):
    
    components = collections.OrderedDict()
    
    # dataset
    reset_mode='uniform'
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
    components['episode_length'] = MaxEpisodeLengthComponent(6)
    
    # viewpoint
    components['viewpoint'] = ControlledAzimuthalViewpointComponent(
        components['scene'],
        azimuth_steps=12,
        elevation_range=(math.radians(-60), math.radians(60)),
        elevation_steps=5,
        distance_range=(200,350),
        distance_steps=4,
        start_position='uniform',
        observe_camera_parameters=False,
        observe_camera_matrix=False,
    )
    
    # color render
    components['color_render'] = ColorRenderComponent(
        width,
        height,
        components['scene'],
        anti_alias=True,
    )
    
    # instance labels
    num_classes = max(dataset_info['shape_ids'].values())+1
    components['class_label'] = InstanceListComponent(
        num_classes,
        dataset_info['max_instances_per_scene'],
        components['dataset'],
        components['scene'],
    )
    
    # build the env
    env = LtronEnv(components, print_traceback=True)
    
    return env
