import os

import numpy

import tqdm

from steadfast.hierarchy import hierarchy_getitem, stack_numpy_hierarchies

from avarice.vector_env import make_vector_env

from ltron_torch.train.ltron_interactive_trainer import (
    LtronInteractiveTrainerConfig
)

def make_dataset(name='2_2_new', n=50000):
    if not os.path.exists(f'./{name}'):
        os.makedirs(f'./{name}')
    config = LtronInteractiveTrainerConfig.from_commandline()
    
    env = make_vector_env(
        config.train_env,
        config.parallel_envs,
        {'config':config, 'train':True},
        seed=12345,
        async_envs=config.async_envs,
    )
    
    observation, info = env.reset()
    
    saved_episodes = 0
    
    current_episodes = [None for _ in range(config.parallel_envs)]
    done = [True for _ in range(config.parallel_envs)]
    
    progress = tqdm.tqdm(total=n)
    
    while saved_episodes < n:
        valid, action, _, _ = observation['expert']
        for i, d in enumerate(done):
            if d:
                if current_episodes[i] is not None:
                    joined_observations = stack_numpy_hierarchies(
                        *current_episodes[i]['observations'])
                    joined_actions = stack_numpy_hierarchies(
                        *current_episodes[i]['actions'])
                    joined_rewards = stack_numpy_hierarchies(
                        *current_episodes[i]['rewards'])
                    joined_episode = {
                        'observations':joined_observations,
                        'actions':joined_actions,
                        'rewards':joined_rewards,
                    }
                    #episodes.append(joined_episode)
                    file_name = f'./{name}/{name}_{saved_episodes}.npz'
                    numpy.savez_compressed(file_name, episode=joined_episode)
                    saved_episodes += 1
                    progress.update(1)
                current_episodes[i] = {
                    'observations':[],
                    'actions':[],
                    'rewards':[],
                }
            o = hierarchy_getitem(observation, i)
            a = hierarchy_getitem(action, i)
            current_episodes[i]['observations'].append(o)
            current_episodes[i]['actions'].append(a)
            
        observation, reward, terminal, truncated, info = env.step(action)
        done = terminal | truncated
        for i, r in enumerate(reward):
            current_episodes[i]['rewards'].append(r)

if __name__ == '__main__':
    make_dataset()
