import math
import io
import os
import tarfile

import numpy

from torch.utils.data import Dataset, DataLoader

from gym.vector.async_vector_env import AsyncVectorEnv

from ltron.config import Config
from ltron.hierarchy import index_hierarchy
from ltron.dataset.paths import get_sources

from ltron_torch.dataset.collate import pad_stack_collate
from ltron_torch.train.epoch import rollout_epoch

class TarDataset(Dataset):
    def __init__(self, tar_paths, subset=None):
        self.tar_paths = tar_paths
        self.tar_files = None
        _, self.names = get_tarfiles_and_names(self.tar_paths, subset=subset)
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, i):
        if self.tar_files is None:
            self.tar_files, _ = get_tarfiles_and_names(self.tar_paths)
        
        tar_path, name = self.names[i]
        data = self.tar_files[tar_path].extractfile(name)
        data = numpy.load(data, allow_pickle=True)
        #data = data['episode'].item()
        data = data['seq'].item()
        
        return data

def make_tar_dataset_and_loader(config, shuffle=False):
    sources = get_sources(config.dataset, config.split)
    dataset = TarDataset(sources, subset=config.subset)
    loader = build_episode_loader(
        dataset,
        config.batch_size,
        config.workers,
        shuffle=shuffle,
    )
    
    return dataset, loader

def get_tarfiles_and_names(tar_paths, subset=None):
    tar_files = {tp:tarfile.open(tp, 'r') for tp in tar_paths}
    names = []
    for tar_path, tar_file in tar_files.items():
        names.extend([(tar_path, name) for name in tar_file.getnames()])
    
    names = names[:subset]
    return tar_files, names
    

def build_episode_loader(dataset, batch_size, workers, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=pad_stack_collate,
        shuffle=shuffle,
    )
    
    return loader

'''
class TarDatasetConfig(Config):
    dataset = 'random_construction'
    split = 'train'
    
    total_episodes = 50000
    shards = 1
    save_episode_frequency = 256
    
    path = '.'
'''

def generate_tar_dataset(
    name,
    total_episodes,
    shards=1,
    shard_start=0,
    save_episode_frequency=256,
    path='.',
    **kwargs,
):
    
    episodes_per_shard = math.ceil(total_episodes/shards)
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    new_shards = []
    for shard in range(shards):
        shard_name = '%s_%04i.tar'%(name, shard+shard_start)
        shard_path = os.path.expanduser(os.path.join(path, shard_name))
        new_shards.append(shard_path)
        print('Making Shard %s'%shard_path)
        shard_tar = tarfile.open(shard_path, 'w')
        shard_seqs = 0
        #rollout_passes = math.ceil(episodes_per_shard/save_episode_frequency)
        #for rollout_pass in range(1, rollout_passes+1):
        while shard_seqs < total_episodes:
            pass_episodes = min(
                episodes_per_shard-shard_seqs, save_episode_frequency)
            pass_name = name + ' (%i-%i/$i)'%(
                shard_seqs, shard_seqs+pass_episodes, total_episodes)
            episodes = rollout_epoch(
                pass_name,
                pass_episodes,
                **kwargs,
            )
            print('Adding Sequences To Shard')
            save_ids = None
            if episodes.num_finished_seqs() > pass_episodes:
                save_ids = list(episodes.finished_seqs)[:pass_episodes]
            episodes.save(
                shard_tar,
                finished_only=True,
                seq_ids=save_ids,
                seq_offset=shard_seqs,
            )
            #shard_seqs += episodes.num_finished_seqs()
            shard_seqs += pass_episodes
    
    return new_shards
