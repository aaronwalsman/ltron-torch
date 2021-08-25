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
    max_sequence_length = 32
    wh = train_config.workspace_image_height // train_config.tile_height
    ww = train_config.workspace_image_width // train_config.tile_width
    hh = train_config.handspace_image_height // train_config.tile_height
    hw = train_config.handspace_image_width // train_config.tile_width
    model_config = CompressedTransformerConfig(
        data_shape = (2, max_sequence_length, wh, ww, hh, hw),
        tile_h=train_config.tile_height,
        tile_w=train_config.tile_width,
        
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
        
        train_epoch(
            train_config, epoch, train_env, model, optimizer, log, clock)
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

def rollout(train_config, epoch, env, model, log, clock):
    print('- '*40)
    print('Rolling out episodes')
    
    # initialize storage for observations, actions and rewards
    observation_storage = RolloutStorage(train_config.num_envs)
    action_reward_storage = RolloutStorage(train_config.num_envs)
    
    # tell the model to keep track of rollout memory
    model.eval()
    
    # reset and get first observation
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
            
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # package data for the model
            seq_lengths = numpy.array(
                observation_storage.get_current_seq_lens())
            workspace_storage = (
                observation_storage['observation', 'workspace_color_render'])
            workspace_frames, padding_mask = workspace_storage.get_current_seqs(
                start=-2)
            
            handspace_storage = (
                observation_storage['observation', 'handspace_color_render'])
            handspace_frames, padding_mask = handspace_storage.get_current_seqs(
                start=-2)
            
            (wx, wi, w_mask) = batch_deduplicate_tiled_seqs(
                workspace_frames,
                padding_mask,
                train_config.tile_width,
                train_config.tile_height,
                background=102,
                s_start=seq_lengths,
            )
            num_workspace = wx.shape[0]
            
            (hx, hi, h_mask) = batch_deduplicate_tiled_seqs(
                handspace_frames,
                padding_mask,
                train_config.tile_width,
                train_config.tile_height,
                background=102,
                s_start=seq_lengths,
            )
            num_handspace = hx.shape[0]
            
            '''
            STOP
            
            Need to figure out the whole input space thing.
            The decoder tokens should be integer embeddings, not tiles.
            So I can't just tack dummy entries onto the input.
            Do I break the embedding out of the transformer model?
            Or do I go in and actually make it do a more complicated thing
            based on decoder tokens or whatever?
            '''
            
            x = numpy.concatenate((wx, hx), axis=0)
            x = default_tile_transform(x).cuda()
            s, b = x.shape[:2]
            i = numpy.ones((s, b, 6)) * -1
            i[:num_workspace, :, 0] = 0
            i[:num_workspace, :, [1,2,3]] = wi
            i[num_workspace:, :, [1,4,5]] = hi
            i = torch.LongTensor(i).cuda()
            tile_padding_mask = numpy.concatenate((w_mask, h_mask))
            tile_padding_mask = torch.BoolTensor(tile_padding_mask).cuda()
            t = torch.LongTensor(seq_lengths).cuda()
            terminal = torch.BoolTensor(terminal).cuda()
            
            action_logits = model(x, i, tile_padding_mask, t, terminal)
            
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
            observation, terminal, reward, info = env.step(actions)
            
            # store actions and rewards
            action_reward_storage.append_batch(
                #action=stack_gym_data(*actions),
                action=stack_numpy_hierarchies(*actions),
                reward=reward,
            )
    
    return observation_storage | action_reward_storage

def train_on_rollouts(train_config, epoch, model, rollout_data, log):
    print('- '*40)
    print('Training on episodes')
    
    batch_iterator = rollout_data.batch_sequence_iterator(
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
