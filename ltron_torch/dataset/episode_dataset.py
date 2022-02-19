import numpy

from torch.utils.data import Dataset, DataLoader

from gym.vector.async_vector_env import AsyncVectorEnv

from ltron.config import Config
from ltron.dataset.paths import get_dataset_paths
from ltron.hierarchy import index_hierarchy

from ltron_torch.dataset.collate import pad_stack_collate

class EpisodeDatasetConfig(Config):
    dataset='omr_clean'
    
    split = 'train_episodes'
    subset = None
    
    batch_size = 2
    loader_workers = 4
    shuffle = True

class EpisodeDataset(Dataset):
    def __init__(self, config):
        self.config = config
        paths = get_dataset_paths(
            config.dataset, config.split, subset=config.subset)
        self.episode_paths = paths['episodes']
    
    def __len__(self):
        return len(self.episode_paths)
    
    def __getitem__(self, i):
        path = self.episode_paths[i]
        data = numpy.load(path, allow_pickle=True)['episode'].item()
        
        return data

def build_episode_loader(config, dataset):
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.loader_workers,
        collate_fn=pad_stack_collate,
        shuffle=config.shuffle,
    )
    
    return loader
