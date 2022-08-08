import random
import os
import io
import tarfile

import numpy

import torch
from torch.nn.functional import cross_entropy
from torch.nn.utils import clip_grad_norm_

import tqdm

from conspiracy.plot import plot_logs, plot_logs_grid

from ltron.rollout import rollout
from ltron.dataset.tar_dataset import generate_tar_dataset

from ltron_torch.models.padding import get_seq_batch_indices
#from ltron_torch.dataset.tar_dataset import TarDataset, build_episode_loader
from ltron_torch.dataset.webdataset import build_episode_loader_from_shards

def rollout_epoch(
    name,
    episodes,
    env,
    model=None,
    rollout_mode='sample',
    expert_probability=1.,
    batch_size=1,
    workers=0,
    shuffle=False,
    shuffle_buffer=1000,
    tar_path=None,
    additional_tar_paths=None,
    shards=1,
    start_shard=0,
    save_episode_frequency=256,
    **kwargs,
):
    
    print('-'*80)
    print('Rolling Out %i Episodes: %s'%(episodes, name))
    print('p(expert) = %.04f'%expert_probability)
    print('rollout_mode = "%s"'%rollout_mode)
    
    if hasattr(model, 'initialize_memory'):
        memory = model.initialize_memory(env.num_envs)
    else:
        memory = None
    
    def actor_fn(observation, terminal, memory):
        
        # reset memory
        if hasattr(model, 'reset_memory'):
            model.reset_memory(memory, terminal)
        
        # if the expert probability is 1, don't run the model forward pass
        b = observation['step'].shape[1]
        if expert_probability == 1.:
            distribution = numpy.zeros((b, env.metadata['action_space'].n))
        
        else:
            # move observations to torch and cuda
            pad = numpy.ones(b, dtype=numpy.long)
            x = model.observation_to_tensors(
                {'observation':observation}, pad)
            
            # model forward pass
            if hasattr(model, 'initialize_memory'):
                if hasattr(model, 'forward_rollout'):
                    x = model.forward_rollout(
                        terminal, **x, memory=memory)
                else:
                    x = model(**x, memory=memory)
                memory = x['memory']
            else:
                if hasattr(model, 'forward_rollout'):
                    x = model.forward_rollout(terminal, **x)
                else:
                    x = model(**x)
            
            # convert model output to actions
            distribution = (
                model.tensor_to_distribution(x).probs.detach().cpu().numpy()
            )
            
            s, b, c = distribution.shape
            distribution = distribution.reshape(b, c)
        
        # insert expert actions
        for i in range(b):
            r = random.random()
            if r < expert_probability:
                expert_actions = observation['expert'][0,i]
                expert_actions = list(set([
                    a for a in expert_actions
                    if a != env.metadata['no_op_action']
                ]))
                #expert_action = random.choice(expert_actions)
                #actions[i] = expert_action
                d = numpy.zeros(env.metadata['action_space'].n)
                d[expert_actions] = 1. / len(expert_actions)
                distribution[i] = d
        
        #for bb in range(b):
        #    max_i = numpy.argmax(distribution[bb])
        #    print(max_i, distribution[bb][max_i])
        
        return distribution, memory
    
    if tar_path:
        with torch.no_grad():
            dataset_shards = generate_tar_dataset(
                name,
                episodes,
                shards=shards,
                start_shard=start_shard,
                save_episode_frequency=save_episode_frequency,
                env=env,
                actor_fn=actor_fn,
                initial_memory=memory,
                path=tar_path,
                rollout_mode=rollout_mode,
                **kwargs,
            )
        
        if additional_tar_paths:
            dataset_shards = dataset_shards + additional_tar_paths
        #dataset = TarDataset(shards)
        #loader = build_episode_loader(dataset, batch_size, workers, shuffle)
        loader = build_episode_loader_from_shards(
            dataset_shards,
            batch_size,
            workers,
            shuffle=shuffle,
            shuffle_buffer=shuffle_buffer,
        )
    
    else:
        with torch.no_grad():
            rollout_episodes = rollout(
                episodes,
                env,
                actor_fn=actor_fn,
                initial_memory=memory,
                rollout_mode=rollout_mode,
                **kwargs
            )
            loader = rollout_episodes.batch_seq_iterator(
                batch_size, finished_only=True, shuffle=shuffle)
    
    return loader

def train_epoch(
    name,
    model,
    optimizer,
    scheduler,
    data_loader,
    loss_log=None,
    agreement_log=None,
    learning_rate_log=None,
    grad_norm_clip=None,
    supervision_mode='action',
    plot=False,
):
    print('-'*80)
    print('Training: %s'%name)
    print('Supervision Mode: %s'%supervision_mode)
    print('Optimizer Settings:')
    print(optimizer)
    model.train()
    
    running_loss = None
    iterate = tqdm.tqdm(data_loader)
    for batch, seq_pad in iterate:
        
        # convert observations to input tensors (x) and labels (y)
        x = model.observation_to_tensors(batch, seq_pad)
        y = model.observation_to_label(batch, seq_pad, supervision_mode)
        
        # forward
        x = model(**x)
        
        # compute loss
        loss = torch.sum(-torch.log_softmax(x, dim=-1) * y, dim=-1)
        
        # average loss over valid entries
        s_i, b_i = get_seq_batch_indices(torch.LongTensor(seq_pad))
        
        loss = loss[s_i, b_i].mean()
        
        if agreement_log is not None:
            s, b, c = x.shape
            max_x = torch.argmax(x, dim=-1).reshape(-1)
            ss = torch.arange(s).view(s,1).expand(s,b).reshape(-1)
            bb = torch.arange(b).view(1,b).expand(s,b).reshape(-1)
            max_y = y[ss, bb, max_x].reshape(s, b)[s_i, b_i]
            agreement = torch.sum(max_y > 0) / max_y.shape[0]
            a = float(agreement.detach().cpu())
            agreement_log.log(a)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        if grad_norm_clip:
            clip_grad_norm_(model.parameters(), grad_norm_clip)
        
        # step
        scheduler.step()
        optimizer.step()

        # update log and tqdm
        l = float(loss)
        if loss_log is not None:
            loss_log.log(l)
        if learning_rate_log is not None:
            learning_rate_log.log(scheduler.get_learning_rate())
        running_loss = running_loss or l
        running_loss = running_loss * 0.9 + l * 0.1
        iterate.set_description('loss: %.04f'%running_loss)
    
    if plot:
        if loss_log is not None:
            loss_chart = plot_logs(
                {'loss':loss_log},
                border='line',
                legend=True,
                min_max_y=True,
                colors={'loss':'RED'},
                x_range=(0.1,1.),
            )
            print(loss_chart)
        
        if agreement_log is not None:
            agreement_chart = plot_logs(
                {'expert agreement':agreement_log},
                border='line',
                legend=True,
                min_max_y=True,
                colors={'expert agreement':'YELLOW'},
                x_range=(0.1,1.),
            )
            print(agreement_chart)
        
        if learning_rate_log is not None:
            learning_rate_chart = plot_logs(
                {'learning rate':learning_rate_log},
                border='line',
                legend=True,
                min_max_y=True,
                colors={'learning rate':'MAGENTA'},
                x_range=(0.1,1.),
            )
            print(learning_rate_chart)

def evaluate_epoch(
    name,
    episodes,
    model,
    success_value,
    reward_log,
    success_log,
):
    print('-'*80)
    print('Evaluating: %s'%name)
    
    avg_terminal_reward = 0.
    total_success = 0.
    total_episodes = 0
    for seq, pad in episodes:
        
        s, b = seq['reward'].shape
        reward = seq['reward'][pad-1, range(b)]
        avg_terminal_reward += numpy.sum(reward)
        total_success += numpy.sum(reward >= success_value)
        total_episodes += len(pad)
    
    if total_episodes:
        avg_terminal_reward /= total_episodes
        avg_success = total_success/total_episodes

    print('Average Terminal Reward: %f'%avg_terminal_reward)
    reward_log.log(avg_terminal_reward)

    print('Average Success: %f (%i/%i)'%(
        avg_success, total_success, total_episodes))
    success_log.log(avg_success)

    chart = plot_logs_grid(
        [[{'%s_reward'%name:reward_log}, {'%s_success'%name:success_log}]],
        border='line',
        legend=True,
        colors={'%s_reward'%name:'BLUE', '%s_success'%name:'GREEN'},
        min_max_y=True,
        x_range=(0.,1.),
    )
    print(chart)

def visualize_epoch(
    name,
    epoch,
    episodes,
    num_episodes,
    model,
):
    print('-'*80)
    print('Generating Visualizations: %s'%name)
    
    visualization_directory = './visualization/epoch_%04i_%s'%(epoch, name)
    if not os.path.exists(visualization_directory):
        os.makedirs(visualization_directory)
    
    model.visualize_episodes(
        epoch, episodes, num_episodes, visualization_directory)
