#!/usr/bin/env python
import time
import os

import numpy

import tqdm

import torch
from torch.distributions import Categorical
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter

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

# config definitions ===========================================================

class BehaviorCloningConfig(Config):
    start_epoch = 1
    epochs = 10
    batch_size = 4
    num_test_envs = 4
    
    train_frequency = 1
    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1
    
    test_rollout_steps_per_epoch = 2048
    
    def set_dependents(self):
        self.test_batch_rollout_steps_per_epoch = (
            self.test_rollout_steps_per_epoch // self.num_test_envs
        )


# train functions ==============================================================

def behavior_cloning(
    config,
    model,
    optimizer,
    scheduler,
    train_loader,
    test_env,
    interface,
):
    
    print('='*80)
    print('Begin Behavior Cloning')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    train_start = time.time()
    
    for epoch in range(config.start_epoch, config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_pass(
            config,
            epoch,
            model,
            optimizer,
            scheduler,
            train_loader,
            interface,
            log,
            clock,
        )
        save_checkpoint(config, epoch, model, optimizer, scheduler, log, clock)
        episodes = test_epoch(
            config, epoch, test_env, model, interface, log, clock)
        if hasattr(interface, 'visualize_episodes'):
            visualize_episodes(config, epoch, episodes, interface, log, clock)
        
        train_end = time.time()
        print('='*80)
        print('Train elapsed: %0.02f seconds'%(train_end-train_start))


# train subfunctions ===========================================================

def train_pass(
    config,
    epoch,
    model,
    optimizer,
    scheduler,
    loader,
    interface,
    log,
    clock,
):
    frequency = config.train_frequency
    if frequency and epoch % frequency == 0:
        print('-'*80)
        print('Training')
        model.train()
        
        for batch, pad in tqdm.tqdm(loader):
            
            observations = batch['observations']
            actions = batch['actions']
            
            # convert observations to model tensors ----------------------------
            x = interface.observation_to_tensors(observations, pad)
            
            # forward ----------------------------------------------------------
            x = model(*x)
            
            # loss -------------------------------------------------------------
            loss = interface.loss(x, pad, actions, log, clock)
            
            # train ------------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            clip_grad(config, model)
            scheduler.step()
            optimizer.step()
            
            clock[0] += 1

def test_epoch(config, epoch, test_env, model, interface, log, clock):
    frequency = config.test_frequency
    if frequency and epoch % frequency == 0:
        print('-'*80)
        print('Testing')
        episodes = rollout_epoch(
            config, test_env, model, interface, 'test', log, clock)
        
        avg_terminal_reward = 0.
        for seq_id in episodes.finished_seqs:
            seq = episodes.get_seq(seq_id)
            avg_terminal_reward += seq['reward'][-1]
        
        if episodes.num_finished_seqs():
            avg_terminal_reward /= episodes.num_finished_seqs()
        
        print('Average Terminal Reward: %f'%avg_terminal_reward)
        log.add_scalar('val/term_reward', avg_terminal_reward, clock[0])
        
        avg_reward = 0.
        entries = 0
        for seq_id in range(episodes.num_seqs()):
            seq = episodes.get_seq(seq_id)
            avg_reward += numpy.sum(seq['reward'])
            entries += seq['reward'].shape[0]
        
        avg_reward /= entries
        print('Average Reward: %f'%avg_reward)
        log.add_scalar('val/reward', avg_reward, clock[0])
        
        if hasattr(interface, 'eval_episodes'):
            interface.eval_episodes(episodes, log, clock)
        
        return episodes
    
    else:
        return None

def rollout_epoch(config, env, model, interface, train_mode, log, clock):
    print('-'*80)
    print('Rolling out episodes')
    
    # initialize storage for observations, actions and rewards
    if train_mode == 'test':
        num_envs = config.num_test_envs
    elif train_mode == 'train':
        num_envs = config.num_train_envs
    observation_storage = RolloutStorage(num_envs)
    action_reward_storage = RolloutStorage(num_envs)
    
    # put the model in eval mode
    model.eval()
    device = next(model.parameters()).device
    
    # use the train mode to determine the number of steps and rollout mode
    if train_mode == 'train':
        steps = config.train_batch_rollout_steps_per_epoch
        rollout_mode = 'sample'
    elif train_mode == 'test':
        steps = config.test_batch_rollout_steps_per_epoch
        rollout_mode = 'max'
    b = num_envs
    
    # reset
    observation = env.reset()
    terminal = numpy.ones(num_envs, dtype=numpy.bool)
    reward = numpy.zeros(num_envs)
    
    with torch.no_grad():
        if hasattr(model, 'initialize_memory'):
            memory = model.initialize_memory(b)
        else:
            memory = None
        
        for step in tqdm.tqdm(range(steps)):
            # prep -------------------------------------------------------------
            # start new sequences if necessary
            action_reward_storage.start_new_seqs(terminal)
            observation_storage.start_new_seqs(terminal)
            
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # move observations to torch and cuda
            pad = numpy.ones(b, dtype=numpy.long)
            observation = stack_numpy_hierarchies(observation)
            x = interface.observation_to_tensors(observation, pad)
            
            # compute actions --------------------------------------------------
            if hasattr(model, 'initialize_memory'):
                if hasattr(interface, 'rollout_forward'):
                    x = interface.rollout_forward(terminal, *x, memory=memory)
                else:
                    x = model(*x, memory=memory)
                memory = x['memory']
            else:
                if hasattr(interface, 'forward_rollout'):
                    x = interface.forward_rollout(terminal, *x)
                else:
                    x = model(*x)
            actions = interface.tensor_to_actions(x, env, mode=rollout_mode)
            
            # step -------------------------------------------------------------
            observation, reward, terminal, info = env.step(actions)
            
            # reset memory -----------------------------------------------------
            if hasattr(model, 'reset_memory'):
                model.reset_memory(memory, terminal)
            
            # storage ----------------------------------------------------------
            action_reward_storage.append_batch(
                action=stack_numpy_hierarchies(*actions),
                reward=reward,
            )
    
    episodes = observation_storage | action_reward_storage
    
    return episodes

def visualize_episodes(config, epoch, episodes, interface, log, clock):
    frequency = config.visualization_frequency
    if frequency and epoch % frequency == 0:
        print('-'*80)
        print('Generating Visualizations')
        
        visualization_directory = os.path.join(
            './visualization',
            os.path.split(log.log_dir)[-1],
            'epoch_%04i'%epoch,
        )
        if not os.path.exists(visualization_directory):
            os.makedirs(visualization_directory)
        
        interface.visualize_episodes(epoch, episodes, visualization_directory)

def save_checkpoint(config, epoch, model, optimizer, scheduler, log, clock):
    frequency = config.checkpoint_frequency
    if frequency and epoch % frequency == 0:
        checkpoint_directory = os.path.join(
            './checkpoint', os.path.split(log.log_dir)[-1])
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        
        path = os.path.join(checkpoint_directory, 'checkpoint_%04i.pt'%epoch)
        print('-'*80)
        print('Saving checkpoint to: %s'%path)
        checkpoint = {
            'config' : config.as_dict(),
            'model' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict(),
        }
        torch.save(checkpoint, path)
