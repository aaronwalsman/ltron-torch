#!/usr/bin/env python
import time
import os

import numpy

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

from PIL import Image

from ltron.dataset.paths import get_dataset_info
from ltron.gym.envs.break_and_make_env import break_and_make_template_action
from ltron.gym.rollout_storage import RolloutStorage
from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    len_hierarchy,
    index_hierarchy,
)
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.config import Config
from ltron_torch.gym_tensor import gym_space_to_tensors, default_tile_transform
from ltron_torch.train.optimizer import build_optimizer
from ltron_torch.dataset.break_and_make import (
    build_train_env,
    build_test_env,
    build_seq_train_loader,
)
from ltron_torch.models.break_and_make_resnet import BreakAndMakeResnet
from ltron_torch.models.break_and_make_lstm import (
    build_model as build_lstm_model,
)
from ltron_torch.envs.break_and_make_lstm_interface import (
    BreakAndMakeLSTMInterface,
)
from ltron_torch.train.behavior_cloning import behavior_cloning

# config definitions ===========================================================

class BehaviorCloningReassemblyConfig(Config):
    epochs=10
    batch_size=4
    num_envs=16
    loader_workers=8
    test_rollout_steps_per_epoch=256
    max_episode_length=64
    
    #skip_reassembly=False
    #insert_only=False
    task = 'break_and_make'
    
    optimizer='adamw'
    learning_rate=3e-4
    weight_decay=0.1
    betas=(0.9, 0.95)
    
    workspace_image_width=256
    workspace_image_height=256
    workspace_map_width=64
    workspace_map_height=64
    handspace_image_width=96
    handspace_image_height=96
    handspace_map_width=24
    handspace_map_height=24
    tile_width=16
    tile_height=16
    
    randomize_viewpoint=True,
    randomize_colors=True,
    
    dataset='random_six'
    train_split='simple_single_seq'
    train_subset=None
    test_split='simple_single'
    test_subset=None
    
    resnet_name='resnet50'
    
    test_frequency=1
    checkpoint_frequency=10
    visualization_frequency=1
    visualization_seqs=10
    
    def set_dependent_variables(self):
        dataset_info = get_dataset_info(self.dataset)
        self.num_classes = max(dataset_info['class_ids'].values()) + 1
        self.num_colors = max(dataset_info['color_ids'].values()) + 1
        
        self.test_batch_rollout_steps_per_epoch = (
            self.test_rollout_steps_per_epoch // self.num_envs
        )

# train functions ==============================================================

def train(train_config):
    print('='*80)
    print('Break and Make Setup')
    model = build_lstm_model(train_config)
    optimizer = build_optimizer(model, train_config)
    train_loader = build_seq_train_loader(train_config)
    test_env = build_test_env(train_config)
    interface = BreakAndMakeLSTMInterface(train_config)
    
    # run behavior cloning
    behavior_cloning(
        train_config, model, optimizer, train_loader, test_env, interface)
