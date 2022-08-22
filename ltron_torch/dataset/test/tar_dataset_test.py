from ltron.gym.envs.ltron_env import async_ltron
from ltron.gym.envs.multiscreen_edit_env import (
    MultiScreenEditEnv,
    MultiScreenEditEnvConfig,
)

from ltron_torch.dataset.tar_dataset import generate_tar_dataset

config = MultiScreenEditEnvConfig(
    dataset='random_construction',
    split='6b_6c_2i_ldraw_train',
    dataset_sample_mode='single_pass',
    max_episode_length=8,
)

if __name__ == '__main__':
    env = async_ltron(
        4,
        MultiScreenEditEnv,
        config,
        print_traceback=True,
    )
    
    generate_tar_dataset(
        'dataset',
        1000,
        env=env,
        expert_probability=1.,
        store_activations=False,
    )
