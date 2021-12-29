import numpy

from torch.utils.data import Dataset, DataLoader

from gym.vector.async_vector_env import AsyncVectorEnv

from ltron.config import Config
from ltron.dataset.paths import get_dataset_paths
from ltron.gym.envs.blocks_env import BlocksEnvConfig, BlocksEnv

from ltron_torch.dataset.collate import pad_stack_collate

class BlocksBehaviorCloningConfig(BlocksEnvConfig):
    dataset='blocks'
    
    train_split = 'train_episodes'
    train_subset = None
    
    test_envs = 4
    
    batch_size = 4
    loader_workers = 4
    shuffle = True

class BlocksSequenceDataset(Dataset):
    def __init__(self, dataset, split, subset):
        paths = get_dataset_paths(dataset, split, subset=subset)
        self.episode_paths = paths['episodes']
    
    def __len__(self):
        return len(self.episode_paths)
    
    def __getitem__(self, i):
        path = self.episode_paths[i]
        data = numpy.load(path, allow_pickle=True)['episode'].item()
        
        return data

def build_sequence_train_loader(config):
    dataset = BlocksSequenceDataset(
        config.dataset,
        config.train_split,
        config.train_subset,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.loader_workers,
        collate_fn=pad_stack_collate,
        shuffle=config.shuffle,
    )
    
    return loader

def build_test_env(config):
    def constructor():
        return BlocksEnv(config)
    constructors = [constructor for i in range(config.test_envs)]
    vector_env = AsyncVectorEnv(constructors, context='spawn')
    
    return vector_env
