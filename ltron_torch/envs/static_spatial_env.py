#!/usr/bin/env python
import random
import argparse
import os

import numpy

from PIL import Image

from ltron.gym.ltron_env import async_ltron
from ltron_torch.envs.spatial_env import pose_estimation_env

parser = argparse.ArgumentParser()

parser.add_argument('--output', type=str, default='.')
parser.add_argument('--resolution', type=str, default='256x256')
parser.add_argument('--label-resolution', type=str, default='32x32')
parser.add_argument('--num-sequences', type=int, default=None)

args = parser.parse_args()
width, height = args.resolution.lower().split('x')
width = int(width)
height = int(height)
label_width, label_height = args.label_resolution.lower().split('x')
label_width = int(label_width)
label_height = int(label_height)

if __name__ == '__main__':
    env = async_ltron(
        1,
        pose_estimation_env,
        dataset='random_stack_redux',
        split='all',
        subset=args.num_sequences,
        width=width,
        height=height,
        segmentation_width=label_width,
        segmentation_height=label_height,
        controlled_viewpoint=False,
        train=False
    )
    
    all_finished = False
    
    step_observations = env.reset()
    while not all_finished:
        actions = [{} for _ in range(env.num_envs)]
        select_actions = numpy.zeros(
            (env.num_envs, label_height, label_width),
            dtype=numpy.bool)
        episode_ids = step_observations['dataset']['episode_id']
        episode_step = step_observations['episode_length']
        for i, ep in enumerate(episode_ids):
            if step_observations['scene']['valid_scene_loaded'][i]:
                print('saving episode: %i frame %i'%(ep, episode_step[i]))
                image = Image.fromarray(step_observations['color_render'][i])
                color_path = os.path.join(
                    args.output,
                    'color_%04i_%04i.png'%(ep, episode_step[i]),
                )
                image.save(color_path)
                
                label_path = os.path.join(
                    args.output,
                    'label_%04i_%04i.npy'%(ep, episode_step[i]),
                )
                numpy.save(
                    label_path,
                    step_observations['dense_class_labels'][i]
                )
                
                pose_path = os.path.join(
                    args.output,
                    'pose_%04i_%04i.npy'%(ep, episode_step[i]),
                )
                numpy.save(pose_path, step_observations['dense_pose_labels'][i])
                
                foreground = (
                    step_observations['dense_class_labels'][i,:,:,0] != 0
                )
                ys, xs = numpy.where(foreground)
                selected_pixel = random.randint(0, ys.shape[0]-1)
                y = ys[selected_pixel]
                x = xs[selected_pixel]
                select_actions[i][y,x] = 1
                actions[i]['visibility'] = select_actions[i]
        
        (step_observations,
         step_rewards,
         step_terminal,
         step_info) = env.step(actions)
        
        all_finished = (
            not numpy.any(step_observations['scene']['valid_scene_loaded']))
