#!/usr/bin/env python
import time
import os

import numpy

import tqdm

import torch
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
#from torch.utils.tensorboard import SummaryWriter

from splendor.image import save_image

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
from ltron_torch.train.optimizer import clip_grad
from ltron_torch.train.epoch import (
    rollout_epoch,
    train_epoch,
    evaluate_epoch,
)

# config definitions ===========================================================

class BehaviorCloningConfig(Config):
    epochs = 10
    
    batch_size = 4
    
    num_test_envs = 4
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    
    test_episodes_per_epoch = 1024
    
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
    success_reward_value=0.
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
        
        this_epoch = lambda freq : freq and epoch % freq == 0
        train_this_epoch = this_epoch(config.train_frequency)
        checkpoint_this_epoch = this_epoch(config.checkpoint_frequency)
        test_this_epoch = this_epoch(config.test_frequency)
        visualize_this_epoch = this_epoch(config.visualization_frequency)
        
        if train_this_epoch:
            train_epoch(
                model,
                optimizer,
                scheduler,
                train_loader,
                train_loss_log,
                grad_norm_clip=config.grad_norm_clip,
                supervision_mode='action',
            )
            #chart = train_log.plot_grid(
            #    topline=True, legend=True, minmax_y=True, height=40, width=72)
            #print(chart)
            chart = plot_logs(
                {'train_loss':train_loss_log},
                border='line',
                legend=True,
                colors={'train_loss':2},
                min_max_y=True,
                x_range=(0.2,1.),
            )
            print(chart)
        
        checkpoint_freq = config.checkpoint_frequency
        if checkpoint_freq and epoch % checkpoint_freq == 0:
            save_checkpoint(
                config, epoch, model, optimizer, scheduler, train_log, test_log)
        
        test_freq = config.test_frequency
        test = test_freq and epoch % test_freq == 0
        vis_freq = config.visualization_frequency
        visualize = vis_freq and epoch % vis_freq == 0
        
        if test or visualize:
            episodes = rollout_epoch(
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
            #test_episodes(config, epoch, episodes, interface, test_log)
            evaluate_epoch(
                'test',
                episodes,
                model,
                success_reward_value,
                test_reward_log,
                test_success_log,
            )
            test_log.step()
            chart = test_log.plot_sequential(
                legend=True, minmax_y=True, height=60, width=160)
            print(chart)
        
        if visualize:
            visualize_episodes(config, epoch, episodes, interface)
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))


# train subfunctions ===========================================================

'''
def train_epoch(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    loader,
    interface,
    train_log,
):
    print('-'*80)
    print('Training')
    model.train()
    
    for batch, pad in tqdm.tqdm(loader):
        
        # convert observations to model tensors
        x = interface.observation_to_tensors(batch, pad)
        y = interface.action_to_tensors(batch, pad)
        
        if hasattr(interface, 'augment'):
            x, y = interface.augment(x, y)
        
        # forward
        x = model(**x)
        
        # loss
        loss = interface.loss(x, y, pad, train_log)
        
        # train
        optimizer.zero_grad()
        loss.backward()
        clip_grad(config, model)
        scheduler.step()
        optimizer.step()
        
        train_log.step()
'''

'''
def test_episodes(config, epoch, episodes, interface, test_log):
    print('-'*80)
    print('Testing')
    
    avg_terminal_reward = 0.
    reward_bins = [0 for _ in range(11)]
    for seq_id in episodes.finished_seqs:
        seq = episodes.get_seq(seq_id)
        reward = seq['reward'][-1]
        avg_terminal_reward += reward
        reward_bin = int(reward * 10)
        reward_bins[reward_bin] += 1
    
    n = episodes.num_finished_seqs()
    if n:
        avg_terminal_reward /= n
    
    print('Average Terminal Reward: %f'%avg_terminal_reward)
    test_log.log(terminal_reward=avg_terminal_reward)
    
    if n:
        bin_percent = [b/n for b in reward_bins]
        for i, (p, c) in enumerate(zip(bin_percent, reward_bins)):
            low = i * 0.1
            print('%.01f'%low)
            print('|' * round(p * 40) + ' (%i)'%c)
    
    #avg_reward = 0.
    #entries = 0
    #for seq_id in range(episodes.num_seqs()):
    #    seq = episodes.get_seq(seq_id)
    #    avg_reward += numpy.sum(seq['reward'])
    #    entries += seq['reward'].shape[0]
    #
    #avg_reward /= entries
    #print('Average Reward: %f'%avg_reward)
    #log.add_scalar('val/reward', avg_reward, clock[0])
    
    if hasattr(interface, 'test_episodes'):
        interface.test_episodes(episodes, test_log)
    
    #return episodes
'''

def visualize_episodes(config, epoch, episodes, interface):
    #frequency = config.visualization_frequency
    #if frequency and epoch % frequency == 0:
    print('-'*80)
    print('Generating Visualizations')
    
    visualization_directory = os.path.join(
        './visualization',
        #os.path.split(log.log_dir)[-1],
        'epoch_%04i'%epoch,
    )
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)
    
    interface.visualize_episodes(epoch, episodes, visualization_directory)

def save_checkpoint(
    config, epoch, model, optimizer, scheduler, train_log, test_log):
    #checkpoint_directory = os.path.join(
    #    './checkpoint', os.path.split(log.log_dir)[-1])
    
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
        'train_log' : train_log.get_state(),
        'test_log' : test_log.get_state(),
    }
    torch.save(checkpoint, path)
