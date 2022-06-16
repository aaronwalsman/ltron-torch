#!/usr/bin/env python
import time
import os

import numpy

import torch
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
#from torch.utils.tensorboard import SummaryWriter

import tqdm

from splendor.image import save_image

from conspiracy.log import Log

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.rollout_storage import RolloutStorage
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    len_hierarchy,
    index_hierarchy,
)
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import make_padding_mask
from ltron_torch.train.reassembly_labels import make_reassembly_labels
from ltron_torch.train.epoch import (
    rollout_epoch,
    train_epoch,
    evaluate_epoch,
    visualize_epoch,
)

# config definition ============================================================

class BehaviorCloningConfig(Config):
    epochs = 10
    
    batch_size = 4
    workers = 4
    
    num_test_envs = 4
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    
    test_episodes_per_epoch = 1024
    visualization_episodes_per_epoch = 16
    
    checkpoint_directory = './checkpoint'

# train functions ==============================================================

def behavior_cloning(
    config,
    train_loader,
    test_env,
    model,
    optimizer,
    scheduler,
    start_epoch = 1,
    success_reward_value=0.,
    train_loss_log=None,
    test_reward_log=None,
    test_success_log=None,
):
    
    print('='*80)
    print('Begin Behavior Cloning')
    train_start = time.time()
    
    print('-'*80)
    print('Building Logs')
    if train_loss_log is None:
        train_loss_log = Log()
    if test_reward_log is None:
        test_reward_log = Log()
    if test_success_log is None:
        test_success_log = Log()
    
    for epoch in range(start_epoch, config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        # figure out what we're doing this epoch
        this_epoch = lambda freq : freq and epoch % freq == 0
        train_this_epoch = this_epoch(config.train_frequency)
        checkpoint_this_epoch = this_epoch(config.checkpoint_frequency)
        test_this_epoch = this_epoch(config.test_frequency)
        visualize_this_epoch = this_epoch(config.visualization_frequency)
        
        # train
        if train_this_epoch:
            train_epoch(
                'Train',
                model,
                optimizer,
                scheduler,
                train_loader,
                train_loss_log,
                grad_norm_clip=config.grad_norm_clip,
                supervision_mode='action',
                plot=True,
            )
        
        # save checkpoint
        if checkpoint_this_epoch:
            save_checkpoint(
                config,
                epoch,
                model,
                optimizer,
                scheduler,
                train_loss_log,
                test_reward_log,
                test_success_log,
            )
        
        test_freq = config.test_frequency
        test = test_freq and epoch % test_freq == 0
        vis_freq = config.visualization_frequency
        visualize = vis_freq and epoch % vis_freq == 0
        
        if test or visualize:
            test_episodes = rollout_epoch(
                'test',
                config.test_episodes_per_epoch,
                test_env,
                model,
                True,
                True,
                rollout_mode='max',
                expert_probability=0.
            )
        
        if test:
            evaluate_epoch(
                'test',
                test_episodes.batch_seq_iterator(1, finished_only=True),
                model,
                success_reward_value,
                test_reward_log,
                test_success_log,
            )
        
        if visualize:
            visualize_epoch(
                'test',
                epoch,
                test_episodes.batch_seq_iterator(1, finished_only=True),
                config.visualization_episodes_per_epoch,
                model,
            )
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))


# train subfunctions ===========================================================

def save_checkpoint(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    train_loss_log,
    test_reward_log,
    test_success_log,
):
    if not os.path.exists(config.checkpoint_directory):
        os.makedirs(config.checkpoint_directory)
    
    path = os.path.join(
        config.checkpoint_directory, 'checkpoint_%04i.pt'%epoch)
    print('-'*80)
    print('Saving checkpoint to: %s'%path)
    checkpoint = {
        'epoch' : epoch,
        'config' : config.as_dict(),
        'model' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'scheduler' : scheduler.state_dict(),
        'train_loss_log' : train_loss_log.get_state(),
        'test_reward_log' : test_reward_log.get_state(),
        'test_success_log' : test_success_log.get_state(),
    }
    torch.save(checkpoint, path)
