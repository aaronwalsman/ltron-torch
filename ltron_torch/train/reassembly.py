#!/usr/bin/env python
import random
import time
import os

import numpy

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical

from ltron.dataset.paths import get_dataset_info
from ltron.gym.ltron_env import async_ltron
from ltron.gym.reassembly_env import (
    handspace_reassembly_env, handspace_reassembly_template_action)
from ltron.gym.rollout_storage import RolloutStorage
from ltron.compression import batch_deduplicate_tiled_seqs
from ltron.hierarchy import stack_numpy_hierarchies

from ltron_torch.models.padding import cat_padded_seqs
from ltron_torch.config import Config
from ltron_torch.gym_tensor import gym_space_to_tensors, default_tile_transform
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)

class ReassemblyTrainConfig(Config):
    epochs=10
    training_passes_per_epoch=8
    batch_size=16
    num_envs=16
    rollout_steps_per_epoch=2048*4
    max_episode_length=32
    
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
    
    dataset='random_six'
    train_split='simple_single'
    train_subset=None
    
    test_frequency=None
    checkpoint_frequency=1
    
    def set_dependent_variables(self):
        self.batch_rollout_steps_per_epoch = (
            self.rollout_steps_per_epoch // self.num_envs
        )

def train_reassembly(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    
    print('-'*80)
    print('Building Model')
    wh = train_config.workspace_image_height // train_config.tile_height
    ww = train_config.workspace_image_width // train_config.tile_width
    hh = train_config.handspace_image_height // train_config.tile_height
    hw = train_config.handspace_image_width // train_config.tile_width
    model_config = CompressedTransformerConfig(
        data_shape = (2, train_config.max_episode_length, wh, ww, hh, hw),
        tile_h=train_config.tile_height,
        tile_w=train_config.tile_width,
        causal_dim=1,
        
        num_blocks=8, # who knows?
        
        input_mode='tile',
        
        decode_input=False,
        decoder_tokens=1,
        decoder_channels=
            train_config.workspace_map_height +
            train_config.workspace_map_width +
            2 +
            2,
    )
    model = CompressedTransformer(model_config).cuda()
    
    print('-'*80)
    print('Building Optimizer')
    optimizer_config = OptimizerConfig()
    optimizer = adamw_optimizer(model, optimizer_config)
    
    dataset_info = get_dataset_info(train_config.dataset)
    
    print('-'*80)
    print('Building Train Env')
    train_env = async_ltron(
        train_config.num_envs,
        handspace_reassembly_env,
        dataset=train_config.dataset,
        split=train_config.train_split,
        subset=train_config.train_subset,
        workspace_image_width=train_config.workspace_image_width,
        workspace_image_height=train_config.workspace_image_height,
        handspace_image_width=train_config.handspace_image_width,
        handspace_image_height=train_config.handspace_image_height,
        workspace_map_width=train_config.workspace_map_width,
        workspace_map_height=train_config.workspace_map_height,
        handspace_map_width=train_config.handspace_map_width,
        handspace_map_height=train_config.handspace_map_height,
        max_episode_length=train_config.max_episode_length,
    )
    
    test_env = None
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        #train_epoch(
        #    train_config, epoch, train_env, model, optimizer, log, clock)
        episodes = rollout_epoch(
            train_config, epoch, train_env, model, log, clock)
        train_epoch(train_config, epoch, model, optimizer, episodes, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        test_epoch(train_config, epoch, test_env, model, log, clock)
        
        epoch_end = time.time()
        print('Elapsed: %.02f seconds'%(epoch_end-epoch_start))
        
def train_epoch(train_config, epoch, train_env, model, optimizer, log, clock):
    print('-'*80)
    print('Train')
    
    rollout_data_labels = rollout(
        train_config, epoch, train_env, model, log, clock)
    train_on_rollouts(
        train_config, epoch, model, rollout_data_labels, log, clock)

def rollout_epoch(train_config, epoch, env, model, log, clock):
    print('- '*40)
    print('Rolling out episodes')
    
    # initialize storage for observations, actions and rewards
    observation_storage = RolloutStorage(train_config.num_envs)
    action_reward_storage = RolloutStorage(train_config.num_envs)
    
    # tell the model to keep track of rollout memory
    model.eval()
    
    # reset and get first observation
    b = train_config.num_envs
    wh = train_config.workspace_image_height
    ww = train_config.workspace_image_width
    hh = train_config.handspace_image_height
    hw = train_config.handspace_image_width
    th = train_config.tile_height
    tw = train_config.tile_width
    prev_workspace_frame = numpy.ones((b, wh, ww, 3), dtype=numpy.uint8) * 102
    prev_handspace_frame = numpy.ones((b, hh, hw, 3), dtype=numpy.uint8) * 102
    
    # reset
    observation = env.reset()
    terminal = numpy.ones(train_config.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(train_config.num_envs)
    
    with torch.no_grad():
        for step in tqdm.tqdm(
            range(train_config.batch_rollout_steps_per_epoch)
        ):
            # start new sequences if necessary
            action_reward_storage.start_new_seqs(terminal)
            observation_storage.start_new_seqs(terminal)
            
            # get sequence lengths before adding the new observation
            seq_lengths = numpy.array(
                observation_storage.get_current_seq_lens())
                
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # generate tile sequences from the workspace
            workspace_frame = observation['workspace_color_render'].reshape(
                1, b, wh, ww, 3)
            pad = numpy.ones(b, dtype=numpy.long)
            (wx, wi, w_pad) = batch_deduplicate_tiled_seqs(
                workspace_frame, pad, tw, th,
                background=prev_workspace_frame,
                s_start=seq_lengths,
            )
            prev_workspace_frame = workspace_frame
            num_workspace = wx.shape[0]
            wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
            wi[:,:,0] = 0
            
            # generate tile sequences from the handspace
            handspace_frame = observation['handspace_color_render'].reshape(
                1, b, hh, hw, 3)
            (hx, hi, h_pad) = batch_deduplicate_tiled_seqs(
                handspace_frame, pad, tw, th,
                background=prev_handspace_frame,
                s_start=seq_lengths,
            )
            prev_handspace_frame = handspace_frame
            num_handspace = hx.shape[0]
            hi = numpy.insert(hi, (0,1,1), -1, axis=-1)
            hi[:,:,0] = 0
            
            # move data to torch and cuda
            wx = torch.FloatTensor(wx)
            hx = torch.FloatTensor(hx)
            w_pad = torch.LongTensor(w_pad)
            h_pad = torch.LongTensor(h_pad)
            x, tile_pad = cat_padded_seqs(wx, hx, w_pad, h_pad)
            x = default_tile_transform(x).cuda()
            tile_pad = tile_pad.cuda()
            s, b = x.shape[:2]
            i, _ = cat_padded_seqs(
                torch.LongTensor(wi), torch.LongTensor(hi), w_pad, h_pad)
            i = i.cuda()
            t = torch.LongTensor(seq_lengths).cuda()
            terminal_tensor = torch.BoolTensor(terminal).cuda()
            
            # compute action logits and sample actions
            if i.numel():
                print(torch.max(i[:,:,1]))
            else:
                print('empty')
            action_logits = model(x, i, tile_pad, t, terminal_tensor)
            
            polarity_logits = action_logits[:,:,:2].view(-1,2)
            polarity_distribution = Categorical(logits=polarity_logits)
            polarity = polarity_distribution.sample().cpu().numpy()
            
            direction_logits = action_logits[:,:,2:4].view(-1,2)
            direction_distribution = Categorical(logits=direction_logits)
            direction = direction_distribution.sample().cpu().numpy()
            
            pick_y_logits = action_logits[:,:,4:64+4].view(-1,64)
            pick_y_distribution = Categorical(logits=pick_y_logits)
            pick_y = pick_y_distribution.sample().cpu().numpy()
            
            pick_x_logits = action_logits[:,:,64+4:128+4].view(-1,64)
            pick_x_distribution = Categorical(logits=pick_x_logits)
            pick_x = pick_x_distribution.sample().cpu().numpy()
            
            pick = numpy.stack((pick_y, pick_x), axis=-1)
            
            # assemble actions
            actions = []
            for i in range(b):
                action = handspace_reassembly_template_action()
                action['disassembly'] = {
                    'activate':True,
                    'polarity':polarity[i],
                    'direction':direction[i],
                    'pick':pick[i],
                }
                actions.append(action)
            
            # send actions to the environment
            observation, reward, terminal, info = env.step(actions)
            
            # store actions and rewards
            action_reward_storage.append_batch(
                action=stack_numpy_hierarchies(*actions),
                reward=reward,
            )
    
    return observation_storage | action_reward_storage

def train_epoch(train_config, epoch, model, optimizer, episodes, log, clock):
    print('- '*40)
    print('Training on episodes')
    
    batch_iterator = episodes.batch_seq_iterator(
        train_config.batch_size, shuffle=True)
    for batch in tqdm.tqdm(batch_iterator):
        
        # train the model on the batch
        model.train(batch)

def test_epoch(train_config, epoch, test_env, model, log):
    frequency = train_config.test_frequency
    if frequency is not None and epoch % frequency == 0:
        rollout_data = rollout(train_config, epoch, test_env, model, log)

def save_checkpoint(train_config, epoch, model, optimizer):
    frequency = train_config.checkpoint_frequency
    if frequency is not None and epoch % frequency == 0:
        checkpoint_directory = os.path.join(
            './checkpoint', os.path.split(log.log_dir)[-1])
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)
        
        print('-'*80)
        model_path = os.path.join(
            checkpoint_directory, 'model_%04i.pt'%epoch)
        print('Saving model to: %s'%model_path)
        torch.save(optimizer.state_dict(), model_path)
        
        optimizer_path = os.path.join(
            checkpoint_directory, 'optimizer_%04i.pt'%epoch)
        print('Saving optimizer to: %s'%optimizer_path)
        torch.save(optimizer.state_dict(), optimizer_path)
