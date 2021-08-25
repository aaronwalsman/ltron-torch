#!/usr/bin/env python
import random
import time
import os

import numpy

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from gym import Env
from gym.spaces import Discrete
from gym.vector.async_vector_env import AsyncVectorEnv

from ltron.dataset.paths import get_dataset_info
from ltron.gym.ltron_env import async_ltron
from ltron.gym.reassembly_env import reassembly_env
from ltron.gym.rollout_storage import RolloutStorage
from ltron.compression import batch_deduplicate_tiled_seqs

from ltron_torch.config import Config
from ltron_torch.gym_tensor import gym_space_to_tensors
from ltron_torch.train.optimizer import OptimizerConfig, adamw_optimizer
from ltron_torch.models.compressed_transformer import (
    CompressedTransformer, CompressedTransformerConfig)

class CountingEnvA(Env):
    def __init__(self, count_max=32):
        self.count_max = count_max
        self.observation_space = Discrete(count_max)
        self.action_space = Discrete(count_max)
    
    def reset(self):
        self.index = random.randint(0, self.count_max-1)
        return self.index
    
    def step(self, action):
        if action == self.index+1:
            reward = 1
        else:
            reward = 0
        self.index += 1
        return self.index, reward, self.index == self.count_max, {}

class CountingEnvB(Env):
    def __init__(self, episode_len=32):
        self.episode_len = episode_len
        self.observation_space = Discrete(2)
        self.action_space = Discrete(episode_len)
    
    def reset(self):
        self.count = 0
        self.steps = 0
        obs = random.random() < 0.05
        self.count += obs
        return obs
    
    def step(self, action):
        reward = action == self.count
        obs = random.random() < 0.05
        self.count += obs
        self.steps += 1
        return obs, reward, self.steps == self.episode_len, {}
    
class MaxEnv(Env):
    def __init__(self, num_values=32, episode_len=32):
        self.num_values = num_values
        self.episode_len = episode_len
        self.observation_space = Discrete(num_values)
        self.action_space = Discrete(num_values)
    
    def reset(self):
        obs = random.randint(0, self.num_values-1)
        self.max_value = obs
        self.steps = 0
        return obs
    
    def step(self, action):
        reward = action == self.max_value
        obs = random.randint(0, self.num_values-1)
        self.max_value = max(self.max_value, obs)
        self.steps += 1
        return obs, reward, self.steps == self.episode_len, {}

class TrainConfig(Config):
    epochs=10
    training_passes_per_epoch=8
    batch_size=16
    num_envs=16
    rollout_steps_per_epoch=2048*64
    
    env = 'Max'
    count_max=32
    
    test_frequency=None
    checkpoint_frequency=1
    
    def set_dependent_variables(self):
        self.batch_rollout_steps_per_epoch = (
            self.rollout_steps_per_epoch // self.num_envs
        )

def train_compressed_transformer(train_config):
    
    print('='*80)
    print('Setup')
    print('-'*80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    
    print('-'*80)
    print('Building Model')
    model_config = CompressedTransformerConfig(
        t=32,
        h=1,
        w=1,
        tile_h=1,
        tile_w=1,
        
        num_blocks=6, #TMP
        
        input_mode='token',
        input_token_vocab=train_config.count_max,
        
        decode_input=True,
        decoder_channels=train_config.count_max+1,
        
        content_dropout = 0.,
        embedding_dropout = 0.,
        attention_dropout = 0.,
        residual_dropout= 0.,
    )
    model = CompressedTransformer(model_config).cuda()
    
    optimizer_config = OptimizerConfig(learning_rate=3e-5)
    optimizer = adamw_optimizer(model, optimizer_config)
    
    print('-'*80)
    print('Building Train Env')
    if train_config.env == 'A':
        constructors = [CountingEnvA for i in range(train_config.num_envs)]
    elif train_config.env == 'B':
        constructors = [CountingEnvB for i in range(train_config.num_envs)]
    elif train_config.env == 'Max':
        constructors = [MaxEnv for i in range(train_config.num_envs)]
    envs = AsyncVectorEnv(constructors, context='spawn')
    
    for epoch in range(1, train_config.epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_epoch(train_config, epoch, envs, model, optimizer, log, clock)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        test_epoch(train_config, epoch, envs, model, log, clock)
        
        epoch_end = time.time()
        print('Elapsed: %.02f seconds'%(epoch_end-epoch_start))
        
def train_epoch(train_config, epoch, train_env, model, optimizer, log, clock):
    print('-'*80)
    print('Train')
    
    rollout_data_labels = rollout(
        train_config, epoch, train_env, model, log, clock)
    train_on_rollouts(
        train_config, epoch, model, optimizer, rollout_data_labels, log, clock)

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
            action_reward_storage.start_new_sequences(terminal)
            observation_storage.start_new_sequences(terminal)
            
            # add latest observation to storage
            observation_storage.append_batch(observation=observation)
            
            # use model to compute actions
            x = torch.LongTensor(observation).view(1,-1).cuda()
            padding_mask = torch.zeros(x.shape, dtype=torch.bool).cuda()
            n, b = x.shape
            i = torch.zeros((n, b, 3), dtype=torch.long).cuda()
            i[:,:,0] = torch.arange(n).view(n,1)
            t = torch.BoolTensor(terminal).cuda()
            a_logits = model(x, i, padding_mask, t)
            
            #a = torch.softmax(x[-1], dim=-1)
            action_distribution = torch.distributions.Categorical(
                logits=a_logits[-1])
            a = action_distribution.sample().cpu().numpy()
            
            # send actions to the environment
            observation, reward, terminal, info = env.step(a)
            
            log.add_scalar(
                'rollout/reward', numpy.sum(reward)/reward.shape[0], clock[0])
            clock[0] += 1
            
            # make labels
            if train_config.env == 'A':
                labels = (x[0] + 1).cpu().numpy()
            elif train_config.env == 'B':
                labels, padding_mask = observation_storage.get_current_seqs()
                labels = labels['observation']
                labels = numpy.sum(labels, axis=0)
            elif train_config.env == 'Max':
                labels, padding_mask = observation_storage.get_current_seqs()
                labels = labels['observation']
                labels = numpy.max(labels, axis=0)
            
            # store actions and rewards
            action_reward_storage.append_batch(
                action=a,
                reward=reward,
                labels=labels,
            )
    
    return observation_storage | action_reward_storage

def train_on_rollouts(
    train_config, epoch, model, optimizer, rollout_data, log, clock):
    print('- '*40)
    print('Training on episodes')
    
    model.train()
    
    for p in range(1, train_config.training_passes_per_epoch+1):
        print('Pass: %i'%p)
        batch_iterator = rollout_data.batch_sequence_iterator(
            train_config.batch_size, shuffle=True)
        for batch, padding_mask in tqdm.tqdm(batch_iterator):
            
            # train the model on the batch
            x = torch.LongTensor(batch['observation']).cuda()
            n, b = x.shape
            i = torch.zeros((n, b, 3), dtype=torch.long).cuda()
            i[:,:,0] = torch.arange(n).view(n,1)
            t = None
            padding_mask = torch.BoolTensor(padding_mask).cuda()
            a_logits = model(x, i, padding_mask)
            
            '''
            if p == 4:
                b_logits = []
                with torch.no_grad():
                    model.eval()
                    steps = x.shape[0]
                    for step in range(steps):
                        xx = x[step].unsqueeze(0)
                        ii = i[step].unsqueeze(0)
                        if step == 0:
                            tt = torch.ones(b, dtype=torch.bool).cuda()
                        else:
                            tt = torch.zeros(b, dtype=torch.bool).cuda()
                        pp = torch.zeros(1, b, dtype=torch.bool).cuda()
                        b_logits.append(model(xx, ii, tt, pp))
                
                b_logits = torch.cat(b_logits, dim=0)
                
                import pdb
                pdb.set_trace()
            '''
            
            s, b, c = a_logits.shape
            a_logits = a_logits.view(s*b, c)
            a_prediction = torch.argmax(a_logits, dim=-1)
            a_labels = torch.LongTensor(batch['labels']).cuda().view(s*b)
            
            loss = torch.nn.functional.cross_entropy(
                a_logits, a_labels, reduction='none')
            loss = loss.masked_fill(padding_mask.view(-1), 0.)
            loss = torch.sum(loss) / torch.sum(~padding_mask)
            loss.backward()
            
            train_correct = a_prediction == a_labels
            train_correct = train_correct.masked_fill(padding_mask.view(-1), 0)
            train_correct = (
                torch.sum(train_correct).float() / torch.sum(~padding_mask))
            log.add_scalar('train/correct', train_correct, clock[0])
            clock[0] += 1
            
            optimizer.step()

def test_epoch(train_config, epoch, test_env, model, log, clock):
    frequency = train_config.test_frequency
    if frequency is not None and epoch % frequency == 0:
        rollout_data = rollout(train_config, epoch, test_env, model, log, clock)

def save_checkpoint(train_config, epoch, model, optimizer, log, clock):
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
        torch.save(model.state_dict(), model_path)
        
        optimizer_path = os.path.join(
            checkpoint_directory, 'optimizer_%04i.pt'%epoch)
        print('Saving optimizer to: %s'%optimizer_path)
        torch.save(optimizer.state_dict(), optimizer_path)

if __name__ == '__main__':
    train_compressed_transformer()
