import numpy

import tqdm

import gymnasium as gym

from splendor.image import save_image

from steadfast.hierarchy import hierarchy_getitem

from ltron.gym.envs.stepped_break_and_make_env import (
    SteppedBreakAndMakeEnvConfig
)

def eval_stepped_break_and_make(
    config=None,
    total_steps=10000,
    eval_expert=False,
):
    if config is None:
        config = SteppedBreakAndMakeEnvConfig.from_commandline()
    config.train_repeat=False
    config.eval_repeat=False
    env = gym.wrappers.AutoResetWrapper(gym.make(
        'LTRON/SteppedBreakAndMakeWithExpert-v1', config=config, train=True))
    
    o,i = env.reset(seed=123456)
    done = True
    
    scene_index = 0
    step_index = 0
    scene_instances = 0
    running_reward = 0.
    returns = []
    truncated = []
    imperfect_scenes = []
    file_names = []
    
    iterate = tqdm.tqdm(range(total_steps))
    with iterate:
        while not env.components['loader'].finished:
            if eval_expert:
                if done:
                    assembly = o['assembly']
                    scene_instances = numpy.sum(assembly['shape'] != 0)
                    file_names.append(env.components['loader'].file_name)
                save_image(
                    o['image'],
                    './tmp_%06i_%03i.png'%(scene_index, step_index),
                )
                expert_actions = o['expert']
                num_expert_actions = o['num_expert_actions']
                action_index = numpy.random.randint(num_expert_actions)
                a = hierarchy_getitem(expert_actions, action_index)
                o,r,t,u,i = env.step(a)
                
                running_reward += r
                step_index += 1
                
                if u:
                    truncated.append(scene_index)
                
                done = t | u
                if done:
                    returns.append(running_reward)
                    if running_reward < 3. * scene_instances:
                        imperfect_scenes.append(scene_index)
                        breakpoint()
                    #print('Finished %i: %.04f'%(scene_index, running_reward))
                    running_reward = 0.
                    scene_index += 1
                    step_index = 0
                    iterate.update(1)
    
    breakpoint()

if __name__ == '__main__':
    config = SteppedBreakAndMakeEnvConfig()
    config.train_dataset = 'rcb'
    config.train_split = '3_4_test'
    config.train_shuffle = False
    config.include_viewpoint = False
    config.include_translate = True
    config.max_time_steps = 100
    eval_stepped_break_and_make(config, total_steps=10000, eval_expert=True)
