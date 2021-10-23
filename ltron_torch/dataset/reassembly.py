import os

import numpy

import torch
from torch.utils.data import Dataset, DataLoader

import tqdm

import ltron.settings as settings
from ltron.exceptions import LtronException
from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron.gym.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.reassembly_env import (
    reassembly_env, reassembly_template_action)
from ltron.gym.rollout_storage import RolloutStorage
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    auto_pad_stack_numpy_hierarchies,
)
from ltron.planner.plannerd import Roadmap, RoadmapPlanner
from ltron.bricks.brick_scene import BrickScene

from ltron_torch.config import Config


# config =======================================================================

class ReassemblyDatasetConfig(Config):
    num_seqs=16
    num_envs=1
    dataset='random_six'
    collection='random_six'
    train_split='simple_single'
    train_subset=None
    start = 0
    end = None
    file_override=None
    
    workspace_image_width=256
    workspace_image_height=256
    workspace_map_width=64
    workspace_map_height=64
    handspace_image_width=96
    handspace_image_height=96
    handspace_map_width=24
    handspace_map_height=24
    
    max_episode_length=32
    
    error_handling='record'


# data generation ==============================================================

def generate_offline_dataset(dataset_config):
    
    dataset_info = get_dataset_info(dataset_config.dataset)
    class_ids = dataset_info['class_ids']
    color_ids = dataset_info['color_ids']
    dataset_paths = get_dataset_paths(
        dataset_config.dataset,
        dataset_config.train_split,
        subset=dataset_config.train_subset,
    )
    
    #print('-'*80)
    #print('Initializing storage')
    #rollout_storage = RolloutStorage(dataset_config.num_envs)
    #observation = train_env.reset()
    #terminal = numpy.ones(dataset_config.num_envs, dtype=numpy.bool)
    #reward = numpy.zeros(dataset_config.num_envs)
    
    rollout_path = os.path.join(
        settings.collections[dataset_config.collection], 'rollouts')
    
    print('-'*80)
    print('Planning plans')
    num_paths = len(dataset_paths['mpd'])
    if dataset_config.end is None:
        end = num_paths
    else:
        end = dataset_config.end
    iterate = tqdm.tqdm(range(dataset_config.start, end))
    complete_seqs = 0
    problems = {}
    for i in iterate:
        env = reassembly_env(
            dataset=dataset_config.dataset,
            split=dataset_config.train_split,
            subset=dataset_config.train_subset,
            rank=i,
            size=num_paths,
            max_episode_length=512,
            randomize_colors=False,
            check_collisions=True,
            print_traceback=True,
            dataset_reset_mode='single_pass',
            train=True,
        )
        
        observation_seq = []
        action_seq = []
        
        # get the full config and state ----------------------------------------
        start_observation = env.reset()
        full_config = start_observation['workspace_config']['config']
        full_state = env.get_state()
        
        # build the disassembly goal (empty) config ----------------------------
        max_instances = full_config['class'].shape[0]
        max_edges = full_config['edges'].shape[1]
        empty_config = {
            'class' : numpy.zeros(max_instances, dtype=numpy.long),
            'color' : numpy.zeros(max_instances, dtype=numpy.long),
            'pose' : numpy.zeros((max_instances, 4, 4)),
            'edges' : numpy.zeros((4, max_edges), dtype=numpy.long),
        }
        
        # disassemble ----------------------------------------------------------
        disassembly_roadmap = Roadmap(env, empty_config, class_ids, color_ids)
        disassembly_planner = RoadmapPlanner(disassembly_roadmap, full_state)
        
        def dump_scene(basename):
            scene = BrickScene()
            class_ids = env.components['workspace_scene'].class_ids
            color_ids = env.components['workspace_scene'].color_ids
            scene.set_configuration(full_config, class_ids, color_ids)
            if not isinstance(basename, str):
                basename = basename.__class__.__name__
            scene.export_ldraw('./%s_%06i.mpd'%(basename, i))
        
        try:
            found_disassembly = disassembly_planner.plan(timeout=2)
            disassembly_path = disassembly_planner.greedy_path()
        except Exception as e:
            if dataset_config.error_handling == 'record':
                dump_scene(e)
                continue
            elif dataset_config.error_handling == 'pdb':
                import pdb
                pdb.set_trace()
            elif dataset_config.error_handling == 'raise':
                raise
            elif dataset_config.error_handling == 'continue':
                continue
        
        # get the disassembly observations and actions -------------------------
        (disassembly_observations,
         disassembly_actions) = disassembly_roadmap.get_observation_action_seq(
            disassembly_path)
        observation_seq.extend(disassembly_observations)
        action_seq.extend(disassembly_actions)
        
        # switch to reassembly -------------------------------------------------
        action = reassembly_template_action()
        action['reassembly']['start'] = True
        action_seq.append(action)
        _, reward, terminal, info = env.step(action)
        
        empty_state = env.get_state()
        
        # reassemble -----------------------------------------------------------
        reassembly_roadmap = Roadmap(env, full_config, class_ids, color_ids)
        reassembly_planner = RoadmapPlanner(reassembly_roadmap, empty_state)
        
        try:
            found_reassembly = reassembly_planner.plan(timeout=2)
            if found_reassembly:
                reassembly_path = reassembly_planner.greedy_path()
            else:
                dump_scene('no_path_found')
                continue
        except Exception as e:
            if dataset_config.error_handling == 'record':
                dump_scene(e)
                continue
            elif dataset_config.error_handling == 'pdb':
                import pdb
                pdb.set_trace()
            elif dataset_config.error_handling == 'raise':
                raise
            elif dataset_config.error_handling == 'continue':
                continue
        
        complete_seqs += 1
        
        # get the reassembly observations and actions --------------------------
        (reassembly_observations,
         reassembly_actions) = reassembly_roadmap.get_observation_action_seq(
            reassembly_path)
        observation_seq.extend(reassembly_observations)
        action_seq.extend(reassembly_actions)
        
        # make the final action ------------------------------------------------
        action = reassembly_template_action()
        action['reassembly']['end'] = True
        action_seq.append(action)
        
        iterate.set_description('Complete: %i/%i'%(complete_seqs, i+1))
        
        observation_seq = stack_numpy_hierarchies(*observation_seq)
        action_seq = stack_numpy_hierarchies(*action_seq)
        
        path = os.path.join(rollout_path, 'rollout_%06i.npz'%i)
        rollout = {'observations':observation_seq, 'actions':action_seq}
        numpy.savez_compressed(path, rollout=rollout)

# dataset ======================================================================

class SeqDataset(Dataset):
    def __init__(self, dataset, split, subset):
        dataset_paths = get_dataset_paths(dataset, split)
        self.rollout_paths = dataset_paths['rollouts']
    
    def __len__(self):
        return len(self.rollout_paths)
    
    def __getitem__(self, i):
        path = self.rollout_paths[i]
        data = numpy.load(path, allow_pickle=True)['rollout'].item()
        #data['observations'] = stack_numpy_hierarchies(*data['observations'])
        #data['actions'] = stack_numpy_hierarchies(*data['actions'])
        return data

def seq_data_collate(seqs):
    #import pdb
    #pdb.set_trace()
    seqs, pad = auto_pad_stack_numpy_hierarchies(
        *seqs, pad_axis=0, stack_axis=1)
    return seqs, pad


# build functions ==============================================================

def build_train_env(dataset_config):
    print('-'*80)
    print('Building train env')
    train_env = async_ltron(
        dataset_config.num_envs,
        reassembly_env,
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
        reassembly_env,
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
