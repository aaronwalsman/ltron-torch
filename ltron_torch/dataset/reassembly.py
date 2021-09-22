import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron.gym.ltron_env import async_ltron, sync_ltron
from ltron.gym.reassembly_env import handspace_reassembly_env
from ltron.gym.rollout_storage import RolloutStorage
from ltron.experts.reassembly import ReassemblyExpert
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    auto_pad_stack_numpy_hierarchies,
)

from ltron_torch.config import Config
#from ltron_torch.models.reassembly import ReassemblyExpert


# config =======================================================================

class ReassemblyDatasetConfig(Config):
    save_path='.'
    num_seqs=16
    num_envs=1
    dataset='random_six'
    train_split='simple_single'
    train_subset=None
    
    workspace_image_width=256
    workspace_image_height=256
    workspace_map_width=64
    workspace_map_height=64
    handspace_image_width=96
    handspace_image_height=96
    handspace_map_width=24
    handspace_map_height=24
    
    max_episode_length=32


# data generation ==============================================================

def generate_offline_dataset(dataset_config):
    
    print('-'*80)
    print('Building expert')
    dataset_info = get_dataset_info(dataset_config.dataset)
    expert = ReassemblyExpert(
        dataset_config.num_envs,
        dataset_info['class_ids'],
        dataset_info['color_ids'],
    )
    
    print('-'*80)
    print('Building environment')
    train_env = build_train_env(dataset_config)
    
    print('-'*80)
    print('Initializing storage')
    rollout_storage = RolloutStorage(dataset_config.num_envs)
    observation = train_env.reset()
    terminal = numpy.ones(dataset_config.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(dataset_config.num_envs)
    
    print('-'*80)
    print('Rolling out expert sequences')
    progress = tqdm.tqdm(total=dataset_config.num_seqs)
    with progress:
        while True:
            rollout_storage.start_new_seqs(terminal)
            labels = expert(observation, terminal, reward)
                
            rollout_storage.append_batch(
                observation=observation,
                label=stack_numpy_hierarchies(*labels),
            )
            observation, reward, terminal, info = train_env.step(labels)
            progress.update(numpy.sum(terminal))
            if rollout_storage.num_finished_seqs() >= dataset_config.num_seqs:
                break
    
    print('-'*80)
    print('Saving sequences')
    rollout_storage.save(dataset_config.save_path, finished_only=True)


# dataset ======================================================================

class SeqDataset(Dataset):
    def __init__(self, dataset, split, subset):
        dataset_paths = get_dataset_paths(dataset, split)
        self.rollout_paths = dataset_paths['rollouts']
    
    def __len__(self):
        return len(self.rollout_paths)
    
    def __getitem__(self, i):
        path = self.rollout_paths[i]
        data = numpy.load(path, allow_pickle=True)['seq'].item()
        return data

def seq_data_collate(seqs):
    seqs, pad = auto_pad_stack_numpy_hierarchies(
        *seqs, pad_axis=0, stack_axis=1)
    return seqs, pad


# build functions ==============================================================

def build_train_env(dataset_config):
    print('-'*80)
    print('Building train env')
    train_env = async_ltron(
        dataset_config.num_envs,
        handspace_reassembly_env,
        dataset=dataset_config.dataset,
        split=dataset_config.train_split,
        subset=dataset_config.train_subset,
        workspace_image_width=dataset_config.workspace_image_width,
        workspace_image_height=dataset_config.workspace_image_height,
        handspace_image_width=dataset_config.handspace_image_width,
        handspace_image_height=dataset_config.handspace_image_height,
        workspace_map_width=dataset_config.workspace_map_width,
        workspace_map_height=dataset_config.workspace_map_height,
        handspace_map_width=dataset_config.handspace_map_width,
        handspace_map_height=dataset_config.handspace_map_height,
        max_episode_length=dataset_config.max_episode_length,
        train=True,
        check_collisions=False, # TMP
        randomize_viewpoint=False,
    )
    
    return train_env


def build_test_env(dataset_config):
    print('-'*80)
    print('Building test env')
    test_env = async_ltron(
        dataset_config.num_envs,
        handspace_reassembly_env,
        dataset=dataset_config.dataset,
        split=dataset_config.test_split,
        subset=dataset_config.test_subset,
        workspace_image_width=dataset_config.workspace_image_width,
        workspace_image_height=dataset_config.workspace_image_height,
        handspace_image_width=dataset_config.handspace_image_width,
        handspace_image_height=dataset_config.handspace_image_height,
        workspace_map_width=dataset_config.workspace_map_width,
        workspace_map_height=dataset_config.workspace_map_height,
        handspace_map_width=dataset_config.handspace_map_width,
        handspace_map_height=dataset_config.handspace_map_height,
        max_episode_length=dataset_config.max_episode_length,
        train=True,
        check_collisions=False, # TMP
        randomize_viewpoint=False,
    )
    
    return test_env


def build_seq_train_loader(config):
    print('-'*80)
    print('Building sequence data loader')
    dataset = SeqDataset(
        config.dataset,
        config.train_split,
        config.train_subset,
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.loader_workers,
        collate_fn=seq_data_collate,
    )
    
    return loader
