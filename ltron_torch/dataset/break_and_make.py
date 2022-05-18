from ltron.gym.envs.multiscreen_edit_env import (
    MultiScreenEditEnv,
    MultiScreenEditEnvConfig,
)

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron

from ltron_torch.dataset.tar_dataset import generate_tar_dataset

class LTRONDatasetConfig(MultiScreenEditEnvConfig):
    name = 'random_construction_6b_6c_2i_episode_train'
    total_episodes = 50000
    shards = 1
    save_episode_frequency = 256
    
    parallel_envs = 4
    async_ltron = True

def ltron_break_and_make_dataset(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = LTRONDatasetConfig.from_commandline()
    
    if config.async_ltron:
        vector_env = async_ltron
    else:
        vector_env = sync_ltron
    env = vector_env(
        config.parallel_envs,
        MultiScreenEditEnv,
        config,
        print_traceback=True,
    )
    
    generate_tar_dataset(
        config.name,
        config.total_episodes,
        shards=config.shards,
        save_episode_frequency=config.save_episode_frequency,
        path='.',
        env=env,
        model=None,
        expert_probability=1.,
        store_activations=False,
    )
