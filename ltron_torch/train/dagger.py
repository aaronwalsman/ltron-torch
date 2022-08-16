import time
import os
import tarfile

import tqdm

import torch

from conspiracy.log import Log

from ltron.config import Config
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

# config definition ============================================================

class DAggerConfig(Config):
    epochs = 10
    passes_per_epoch = 3
    recent_epochs_to_save = 8
    
    save_episode_frequency = 256
    
    batch_size = 8
    workers = 4
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    evaluate_train = True
    
    train_episodes_per_epoch = 4096
    test_episodes_per_epoch = 1024
    visualization_episodes_per_epoch = 16
    
    checkpoint_directory = './checkpoint'
    
    expert_probability_start = 1.
    expert_probability_end = 0.
    expert_decay_start = 1
    expert_decay_end = 50
    
    supervision_mode = 'expert_uniform_distribution'
    
    shuffle_train = True

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
    train_agreement_log=None,
    learning_rate_log=None,
    train_reward_log=None,
    train_success_log=None,
    test_reward_log=None,
    test_success_log=None,
):
    
    print('='*80)
    print('Begin DAgger')
    print('-'*80)
    print('Epochs: %i'%config.epochs)
    print('Passes Per Epoch: %i'%config.passes_per_epoch)
    print('Recent Epochs To Save: %i'%config.recent_epochs_to_save)
    print('Batch Size: %i'%config.batch_size)
    print('Workers: %i'%config.workers)
    print('Train Frequency: %i'%config.train_frequency)
    print('Test Frequency: %i'%config.test_frequency)
    print('Checkpoint Frequency: %i'%config.checkpoint_frequency)
    print('Visualization Frequency: %i'%config.visualization_frequency)
    print('Train Episodes Per Epoch: %i'%config.train_episodes_per_epoch)
    print('Test Episodes Per Epoch: %i'%config.test_episodes_per_epoch)
    print('Visualization Episodes Per Epoch: %i'%
        config.visualization_episodes_per_epoch)
    print('Expert Probability Start: %.04f'%config.expert_probability_start)
    print('Expert Probability End: %.04f'%config.expert_probability_end)
    print('Expert Decay Start: %i'%config.expert_decay_start)
    print('Expert Decay End: %i'%config.expert_decay_end)
    
    train_start = time.time()
    
    print('-'*80)
    print('Building Logs')
    if train_loss_log is None:
        train_loss_log = Log()
    if train_agreement_log is None:
        train_agreement_log = Log()
    if learning_rate_log is None:
        learning_rate_log = Log()
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
            
            if config.recent_epochs_to_save:
                e = epoch - 1
                shard_index = e % config.recent_epochs_to_save
                #scratch_path = './data_scratch/recent_%04i.tar'%r
                scratch_path = './data_scratch'
                additional_tar_paths = [
                    '%s/train_%04i.tar'%(scratch_path, i)
                    for i in range(config.recent_epochs_to_save)
                    if i != shard_index and i < e
                    and os.path.exists('%s/train_%04i.tar'%(scratch_path, i))
                ]
            else:
                scratch_path = None
                additional_tar_paths = []
                shard_index = 0
            
            train_episodes = (
                (len(additional_tar_paths)+1) * config.train_episodes_per_epoch)
            train_loader = rollout_epoch(
                'train',
                config.train_episodes_per_epoch,
                train_env,
                model=model,
                rollout_mode='sample',
                expert_probability=expert_probability,
                batch_size=config.batch_size,
                workers=config.workers,
                dataset_length=train_episodes,
                shuffle=config.shuffle_train,
                tar_path=scratch_path,
                additional_tar_paths=additional_tar_paths,
                shards=1,
                start_shard=shard_index,
                save_episode_frequency=config.save_episode_frequency,
            )
            
            # evaluate training episodes
            if config.evaluate_train:
                evaluate_epoch(
                    'train',
                    train_loader,
                    model,
                    success_reward_value,
                    #loader_length=train_loader_length,
                    reward_log=train_reward_log,
                    success_log=train_success_log,
                )
        
        # train
        if train_this_epoch:
            for i in range(1, config.passes_per_epoch+1):
                train_epoch(
                    'Pass %i'%i,
                    model,
                    optimizer,
                    scheduler,
                    train_loader,
                    #loader_length=train_loader_length,
                    loss_log=train_loss_log,
                    agreement_log=train_agreement_log,
                    learning_rate_log=learning_rate_log,
                    grad_norm_clip=config.grad_norm_clip,
                    supervision_mode=config.supervision_mode,
                    plot=(i==config.passes_per_epoch),
                )
        
        # visualize training episodes
        if visualize_this_epoch:
            visualize_epoch(
                'train',
                epoch,
                train_loader,
                config.visualization_episodes_per_epoch,
                model,
            )
        
        # save checkpoint
        if checkpoint_this_epoch:
            save_checkpoint(
                config=config,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loss_log=train_loss_log,
                train_agreement_log=train_agreement_log,
                learning_rate_log=learning_rate_log,
                train_reward_log=train_reward_log,
                train_success_log=train_success_log,
                test_reward_log=test_reward_log,
                test_success_log=test_success_log,
            )
        
        # rollout test episodes
        if test_this_epoch or visualize_this_epoch:
            test_loader = rollout_epoch(
                'test',
                config.test_episodes_per_epoch,
                test_env,
                model,
                rollout_mode='max',
                expert_probability=0.,
                batch_size=config.batch_size,
                workers=config.workers,
                shuffle=False,
            )
        
        # evaluate test episodes
        if test_this_epoch:
            evaluate_epoch(
                'test',
                test_loader,
                model,
                success_reward_value,
                reward_log=test_reward_log,
                success_log=test_success_log,
            )
        
        # visualize training episodes
        if visualize_this_epoch:
            visualize_epoch(
                'test',
                epoch,
                test_loader,
                config.visualization_episodes_per_epoch,
                model,
            )
        
        # keep time
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
    train_agreement_log,
    learning_rate_log,
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
        'train_agreement_log' : train_agreement_log.get_state(),
        'learning_rate_log' : learning_rate_log.get_state(),
        'train_reward_log' : train_reward_log.get_state(),
        'train_success_log' : train_success_log.get_state(),
        'test_reward_log' : test_reward_log.get_state(),
        'test_success_log' : test_success_log.get_state(),
    }
    torch.save(checkpoint, path)
