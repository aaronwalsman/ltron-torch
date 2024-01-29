from avarice.vector_env import make_vector_env

from ltron_torch.train.ltron_interactive_trainer import (
    LtronInteractiveTrainerConfig
)

def make_dataset(n=50000):
    config = LtronInteractiveTrainerConfig.from_commandline()
    
    env = make_vector_env(
        config.train_env,
        config.parallel_envs,
        {'config':config, 'train':True},
        seed=12345,
        async_envs=config.async_envs,
    )
    
    observation, info = env.reset()
    
    episodes = []
    
    current_episodes = [None for _ in range(config.parallel_envs)]
    done = [True for _ in range(config.parallel_envs)]
    
    while len(episodes) < n:
        breakpoint()
        action = SOMETHING
        for i, d in enumerate(done):
            if d:
                if current_episodes[i] is not None:
                    episodes.append(joined_episode)
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
        for i, r in enumerate(reward):
            current_episodes[i]['reward'].append(r)

if __name__ == '__main__':
    make_dataset()
