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

from conspiracy.log import Log, plot_logs, plot_logs_grid

from ltron.config import Config
from ltron.dataset.paths import get_dataset_info
from ltron.gym.rollout_storage import RolloutStorage
from ltron.hierarchy import (
    stack_numpy_hierarchies,
    len_hierarchy,
    index_hierarchy,
)
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import make_padding_mask, get_seq_batch_indices
from ltron_torch.train.reassembly_labels import make_reassembly_labels
from ltron_torch.train.optimizer import clip_grad
from ltron_torch.train.rollout import rollout_epoch
from ltron_torch.dataset.tar_dataset import TarDataset, build_episode_loader

# config definitions ===========================================================

class DAggerConfig(Config):
    epochs = 10
    passes_per_epoch = 3
    recent_epochs_to_save = 8
    
    task = 'break_and_make'
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    
    train_episodes_per_epoch = 4096
    test_episodes_per_epoch = 1024
    visualization_episodes_per_epoch = 16
    
    batch_size = 8
    workers = 4
    
    checkpoint_directory = './checkpoint'
    
    expert_probability_start = 1.
    expert_probability_end = 0.
    expert_decay_start = 1
    expert_decay_end = 50


# train functions ==============================================================

def dagger(
    config,
    train_env,
    test_env,
    #interface,
    model,
    optimizer,
    scheduler,
    start_epoch=1,
    success_value=0.,
    train_loss_log=None,
    train_reward_log=None,
    train_success_log=None,
    test_reward_log=None,
    test_success_log=None,
):
    
    print('='*80)
    print('Begin DAgger')
    
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
    
    train_start = time.time()
    
    for epoch in range(start_epoch, config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        this_epoch = lambda f, e : f and e % f == 0
        train_this_epoch = this_epoch(config.train_frequency, epoch)
        checkpoint_this_epoch = this_epoch(config.checkpoint_frequency, epoch)
        test_this_epoch = this_epoch(config.test_frequency, epoch)
        visualize_this_epoch = this_epoch(config.visualization_frequency, epoch)
        
        # rollout training episodes
        if train_this_epoch or visualize_this_epoch:
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
                r = epoch % config.recent_epochs_to_save
                save_episodes = './data_scratch/recent_%04i.tar'%r
            else:
                save_episodes = None
            train_episodes = rollout_epoch(
                epoch,
                config.train_episodes_per_epoch,
                train_env,
                model,
                True,
                visualize_this_epoch,
                'sample',
                expert_probability=expert_probability,
                save_episodes=save_episodes,
            )
            
            evaluate_episodes(
                'train',
                config,
                epoch,
                train_episodes,
                model,
                success_value,
                train_reward_log,
                train_success_log,
                color=3,
            )
        
        if train_this_epoch:
            train_epoch(
                config,
                epoch,
                model,
                optimizer,
                scheduler,
                train_episodes,
                train_loss_log,
                train_reward_log,
            )
            
        if visualize_this_epoch:
            visualize_episodes(
                config, epoch, train_episodes,
                model,
                'train')
        
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
        
        if test_this_epoch or visualize_this_epoch:
            test_episodes = rollout_epoch(
                epoch,
                config.test_episodes_per_epoch,
                test_env,
                model,
                #interface,
                True,
                visualize_this_epoch,
                rollout_mode='max',
                expert_probability=0.,
            )
        
        if test_this_epoch:
            evaluate_episodes(
                'test',
                config,
                epoch,
                test_episodes,
                #interface,
                model,
                success_value,
                test_reward_log,
                test_success_log,
                color=4,
            )
        
        if visualize_this_epoch:
            visualize_episodes(config, epoch, test_episodes,
                #interface,
                model,
                'test')
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))


# train subfunctions ===========================================================

def train_epoch(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    episodes,
    train_loss_log,
    train_reward_log,
):
    print('-'*80)
    print('Training')
    model.train()
    
    # randomly iterate through the completed episodes
    for i in range(1, config.passes_per_epoch+1):
        print('Pass: %i'%i)
        if config.recent_epochs_to_save:
            scratch_files = [
                './data_scratch/recent_%04i.tar'%i
                for i in range(config.recent_epochs_to_save)
            ]
            scratch_files = [s for s in scratch_files if os.path.exists(s)]
            assert len(scratch_files)
                
            dataset = TarDataset(scratch_files)
            loader = build_episode_loader(
                dataset, config.batch_size, config.workers, shuffle=True)
            iterate = tqdm.tqdm(loader)
        else:
            iterate = tqdm.tqdm(
                episodes.batch_seq_iterator(config.batch_size, shuffle=True))
        
        running_loss = None
        for batch, pad in iterate:
            
            # convert observations to model tensors
            device = next(model.parameters()).device
            #x = interface.observation_to_tensors(batch, pad, device)
            x = model.observation_to_tensors(batch, pad, device)
            #y = interface.label_actions(batch, pad)
            #y = interface.observation_to_single_label(batch, pad, device)
            #y = model.observation_to_single_label(batch, pad, device)
            y = model.observation_to_label_distribution(batch, pad, device)
            '''
            for yy in y.view(-1):
                yy = yy.detach().cpu().numpy()
                n, i = model.action_space.unravel(yy)
                if n in model.embedding.cursor_names:
                    s, *yxc = (
                        model.action_space.subspaces[n].screen_span.unravel(i))
                    print(n, s, yxc)
                else:
                    print(yy, n, i)
            '''
            #if hasattr(interface, 'augment'):
            #    x, y = interface.augment(x, y)
            
            # forward
            x = model(**x)
            
            # loss
            #s, b = x.shape[:2]
            #loss = cross_entropy(x.view(s*b,-1), y.view(-1), reduction='none')
            #s_i, b_i = get_seq_batch_indices(torch.LongTensor(pad))
            #loss = loss.view(s,b)[s_i, b_i].mean()
            #print(torch.max(x).detach().cpu().numpy(), torch.min(x).detach().cpu().numpy())
            loss = torch.sum(-torch.log_softmax(x, dim=-1) * y, dim=-1)
            s_i, b_i = get_seq_batch_indices(torch.LongTensor(pad))
            loss = loss[s_i, b_i].mean()
            
            # train
            optimizer.zero_grad()
            loss.backward()
            #import pdb
            #pdb.set_trace()
            clip_grad(config, model)
            scheduler.step()
            optimizer.step()
            
            # update logs and tqdm
            train_loss_log.log(float(loss))
            if running_loss is None:
                running_loss = float(loss)
            else:
                running_loss = running_loss * 0.9 + float(loss) * 0.1
            iterate.set_description('loss: %.04f'%running_loss)
    
    chart = plot_logs(
        {'train_loss':train_loss_log},
        #border='top_line',
        border='line',
        legend=True,
        colors={'train_loss':2},
        min_max_y=True,
        x_range=(0.2,1.),
    )
    print(chart)
    
    #chart = train_loss_log.plot_grid(
    #    topline=True, legend=True, minmax_y=True, height=40, width=72)
    #print(chart)

def evaluate_episodes(
    name,
    config,
    epoch,
    episodes,
    model,
    success_value,
    reward_log,
    success_log,
    color=4,
):
    print('-'*80)
    print('Evaluating Episodes: %s'%name)
    
    avg_terminal_reward = 0.
    total_success = 0.
    for seq_id in episodes.finished_seqs:
        seq = episodes.get_seq(seq_id)
        reward = seq['reward'][-1]
        avg_terminal_reward += reward
        if reward >= success_value:
            total_success += 1
    
    n = episodes.num_finished_seqs()
    if n:
        avg_terminal_reward /= n
        avg_success = total_success/n
    
    print('Average Terminal Reward: %f'%avg_terminal_reward)
    reward_log.log(avg_terminal_reward)
    
    print('Average Success: %f (%i/%i)'%(avg_success, total_success, n))
    success_log.log(avg_success)
    
    chart = plot_logs_grid(
        [[{'%s_reward'%name:reward_log}, {'%s_success'%name:success_log}]],
        border='line',
        legend=True,
        #colors={'%s_reward'%name:color},
        colors='auto',
        min_max_y=True,
        x_range=(0.,1.),
    )
    print(chart)

def visualize_episodes(config, epoch, episodes,
    #interface,
    model, suffix):
    print('-'*80)
    print('Generating Visualizations (%s)'%suffix)
    
    visualization_directory = os.path.join(
        './visualization',
        #os.path.split(log.log_dir)[-1],
        'epoch_%04i_%s'%(epoch, suffix),
    )
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)
    
    #interface.visualize_episodes(
    model.visualize_episodes(
        epoch,
        episodes,
        config.visualization_episodes_per_epoch,
        visualization_directory,
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
