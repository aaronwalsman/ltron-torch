#!/usr/bin/env python
import time
import os
import tarfile

import numpy

import tqdm

import torch
import torch.autograd as autograd
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

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

from ltron_torch.train.reassembly_labels import make_reassembly_labels
from ltron_torch.train.epoch import (
    rollout_epoch,
    train_epoch,
    evaluate_epoch,
    visualize_epoch,
)
from ltron_torch.dataset.tar_dataset import (
    TarDataset, build_episode_loader, generate_tar_dataset,
)

# config definitions ===========================================================

class DAggerConfig(Config):
    epochs = 10
    passes_per_epoch = 3
    recent_epochs_to_save = 8
    
    batch_size = 8
    workers = 4
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    
    train_episodes_per_epoch = 4096
    test_episodes_per_epoch = 1024
    visualization_episodes_per_epoch = 16
    
    checkpoint_directory = './checkpoint'
    
    expert_probability_start = 1.
    expert_probability_end = 0.
    expert_decay_start = 1
    expert_decay_end = 50
    
    supervision_mode = 'expert_uniform_distribution'

# train functions ==============================================================

def dagger(
    config,
    train_env,
    test_env,
    model,
    optimizer,
    scheduler,
    start_epoch=1,
    success_reward_value=0.,
    train_loss_log=None,
    train_reward_log=None,
    train_success_log=None,
    test_reward_log=None,
    test_success_log=None,
):
    
    print('='*80)
    print('Begin DAgger')
    train_start = time.time()
    
    print('-'*80)
    print('Building Logs')
    if train_loss_log is None:
        train_loss_log = Log()
    if train_reward_log is None:
        train_reward_log = Log()
    if train_success_log is None:
        train_success_log = Log()
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
        
        # rollout training episodes
        if train_this_epoch or visualize_this_epoch:
            
            # compute the expert probability for this epoch
            if epoch <= config.expert_decay_start:
                expert_probability = config.expert_probability_start
            elif epoch >= config.expert_decay_end:
                expert_probability = config.expert_probability_end
            else:
                t = ((epoch - config.expert_decay_start) /
                     (config.expert_decay_end - config.expert_decay_start))
                expert_probability = (
                    t * config.expert_probability_end +
                    (1-t) * config.expert_probability_start
                )
            
            # rollout training episodes
            if config.recent_epochs_to_save:
                r = epoch % config.recent_epochs_to_save
                #scratch_path = './data_scratch/recent_%04i.tar'%r
                scratch_path = './data_scratch'
                new_shards = generate_tar_dataset(
                    'train',
                    config.train_episodes_per_epoch,
                    env=train_env,
                    model=model,
                    expert_probability=expert_probability,
                    shards=1,
                    shard_start=r,
                    path=scratch_path,
                )
                train_episodes = TarDataset(new_shards)
                
            else:
                #save_episodes = None
                train_episodes = rollout_epoch(
                    'train',
                    config.train_episodes_per_epoch,
                    train_env,
                    model,
                    True,
                    True,
                    'sample',
                    expert_probability=expert_probability,
                    #save_episodes=save_episodes,
                )
            
            # evaluate training episodes
            evaluate_epoch(
                'train',
                train_episodes,
                model,
                success_reward_value,
                train_reward_log,
                train_success_log,
            )
        
        # train
        if train_this_epoch:
            train_dagger_epoch(
                config,
                epoch,
                model,
                optimizer,
                scheduler,
                train_episodes,
                train_loss_log,
            )
        
        # visualize training episodes
        if visualize_this_epoch:
            visualize_epoch(
                'train',
                epoch,
                train_episodes,
                config.visualization_episodes_per_epoch,
                model,
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
                train_reward_log,
                train_success_log,
                test_reward_log,
                test_success_log,
            )
        
        # rollout test episodes
        if test_this_epoch or visualize_this_epoch:
            test_episodes = rollout_epoch(
                'test',
                config.test_episodes_per_epoch,
                test_env,
                model,
                True,
                visualize_this_epoch,
                rollout_mode='max',
                expert_probability=0.,
            )
        
        # evaluate test episodes
        if test_this_epoch:
            evaluate_epoch(
                'test',
                test_episodes,
                model,
                success_reward_value,
                test_reward_log,
                test_success_log,
            )
        
        # visualize training episodes
        if visualize_this_epoch:
            visualize_epoch(
                'test',
                epoch,
                test_episodes,
                config.visualization_episodes_per_epoch,
                model,
            )
        
        # keep time
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))

# train subfunctions ===========================================================

def train_dagger_epoch(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    episodes,
    train_loss_log,
):
    # create dataset and loader if training on recent data
    if config.recent_epochs_to_save:
        scratch_files = [
            './data_scratch/train_%04i.tar'%i
            for i in range(config.recent_epochs_to_save)
        ]
        scratch_files = [s for s in scratch_files if os.path.exists(s)]
        assert len(scratch_files)
            
        dataset = TarDataset(scratch_files)
        loader = build_episode_loader(
            dataset, config.batch_size, config.workers, shuffle=True)
    
    # randomly iterate through the completed episodes
    for i in range(1, config.passes_per_epoch+1):
        # make the dataset iterable
        if config.recent_epochs_to_save:
            data = loader
        else:
            data = episodes.batch_seq_iterator(config.batch_size, shuffle=True)
        
        train_epoch(
            'Pass %i'%i,
            model,
            optimizer,
            scheduler,
            data,
            train_loss_log,
            grad_norm_clip=config.grad_norm_clip,
            supervision_mode=config.supervision_mode,
            plot=(i==config.passes_per_epoch),
        )

def save_checkpoint(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    train_loss_log,
    train_reward_log,
    train_success_log,
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
        'train_reward_log' : train_reward_log.get_state(),
        'train_success_log' : train_success_log.get_state(),
        'test_reward_log' : test_reward_log.get_state(),
        'test_success_log' : test_success_log.get_state(),
    }
    torch.save(checkpoint, path)
