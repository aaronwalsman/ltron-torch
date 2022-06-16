import random
import os
import io
import tarfile

import numpy

import torch
from torch.nn.utils import clip_grad_norm_

import tqdm

from conspiracy.plot import plot_logs, plot_logs_grid

#from ltron.hierarchy import map_hierarchies, stack_numpy_hierarchies
from ltron.gym.rollout_storage import RolloutStorage
from ltron.rollout import rollout

from ltron_torch.models.padding import get_seq_batch_indices

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
    tar_path=None,
    additional_tar_paths=None,
    shards=1,
    start_start=0,
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
        
        # if the expert probability is 1 and we don't need model
        # activations, don't bother running the model forward pass
        #if expert_probability == 1. and not store_activations:
        b = observation['step'].shape[1]
        if expert_probability == 1.:
            distribution = numpy.zeros(b, env.metadata['action_space'].n)
        
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
                model.tensor_to_distribution(x).detach().cpu().numpy()
            )
        
        # insert expert actions
        for i in range(b):
            r = random.random()
            if r < expert_probability:
                expert_actions = observation['expert'][0,i]
                expert_actions = [
                    a for a in expert_actions
                    if a != env.metadata['no_op_action']
                ]
                #expert_action = random.choice(expert_actions)
                #actions[i] = expert_action
                d = numpy.zeros(env.metadata['action_space'].n)
                d[expert_actions] = 1. / len(expert_actions)
                distribution[i] = d
        
        return distribution, memory
    
    if tar_path:
        shards = generate_tar_dataset(
            name,
            episodes,
            shards=shards,
            shard_start=shard_start,
            save_episode_frequency=save_episode_frequency,
            env=env,
            actor_fn=actor_fn,
            initial_memory=memory,
            path=tar_path,
            **kwargs,
        )
        
        if additional_tar_paths:
            shards = shards + additional_tar_paths
        dataset = TarDataset(shards)
        loader = build_episode_loader(dataset, batch_size, workers, shuffle)
    
    else:
        rollout_episodes = rollout(
            episodes, env, actor_fn=actor_fn, initial_memory=memory, **kwargs)
        loader = rollout_episodes.batch_seq_iterator(
            batch_size, finished_only=True, shuffle=shuffle)
    
    return loader

def rollout_epoch_old(
    name,
    episodes,
    env,
    model=None,
    store_observations=True,
    store_actions=True,
    store_activations=True,
    store_rewards=True,
    rollout_mode='sample',
    expert_probability=1.,
):
    print('-'*80)
    print('Rolling Out %i Episodes: %s'%(episodes, name))
    print('p(expert) = %.04f'%expert_probability)
    print('rollout_mode = "%s"'%rollout_mode)
    
    # initialize storage for observations, actions, rewards and activations
    storage = {}
    if store_observations:
        storage['observation'] = RolloutStorage(env.num_envs)
    if store_actions:
        storage['action'] = RolloutStorage(env.num_envs)
    if store_activations:
        storage['activation'] = RolloutStorage(env.num_envs)
    if store_rewards:
        storage['reward'] = RolloutStorage(env.num_envs)
    
    assert len(storage)
    first_key, first_storage = next(iter(storage.items()))
    
    # put the model in eval mode
    if expert_probability < 1.:
        model.eval()
        device = next(model.parameters()).device
    
    assert rollout_mode in ('sample', 'max')
    b = env.num_envs

    # reset
    observation = env.reset()
    terminal = numpy.ones(env.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(env.num_envs)
    
    with torch.no_grad():
        if hasattr(model, 'initialize_memory') and expert_probability < 1.:
            memory = model.initialize_memory(b)
        else:
            memory = None
    
        #for step in tqdm.tqdm(range(steps)):
        progress = tqdm.tqdm(total=episodes)
        with progress:
            while first_storage.num_finished_seqs() < episodes:
                # prep ---------------------------------------------------------
                # start new sequences if necessary
                if store_observations:
                    storage['observation'].start_new_seqs(terminal)
                if store_actions:
                    storage['action'].start_new_seqs(terminal)
                if store_activations:
                    storage['activation'].start_new_seqs(terminal)
                if store_rewards:
                    storage['reward'].start_new_seqs(terminal)

                # add latest observation to storage
                if store_observations:
                    storage['observation'].append_batch(observation=observation)
                
                # compute actions ----------------------------------------------
                pad = numpy.ones(b, dtype=numpy.long)
                observation = stack_numpy_hierarchies(observation)
                
                # if the expert probability is 1 and we don't need model
                # activations, don't bother running the model forward pass
                if expert_probability == 1. and not store_activations:
                    actions = [0] * b
                
                else:
                    # move observations to torch and cuda
                    x = model.observation_to_tensors(
                        {'observation':observation}, pad)
                    
                    # model forward
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
                    actions = model.tensor_to_actions(x, mode=rollout_mode)
                    #for a in actions:
                    #    print(model.action_space.unravel(a))
                
                # insert expert actions
                for i in range(b):
                    r = random.random()
                    if r < expert_probability:
                        expert_actions = observation['expert'][0,i]
                        expert_actions = [
                            a for a in expert_actions
                            if a != env.metadata['no_op_action']
                        ]
                        expert_action = random.choice(expert_actions)
                        actions[i] = expert_action
                        
                # step ---------------------------------------------------------
                observation, reward, terminal, info = env.step(actions)
                
                # reset memory -------------------------------------------------
                if hasattr(model, 'reset_memory'):
                    model.reset_memory(memory, terminal)

                # storage ------------------------------------------------------
                if store_actions:
                    a = stack_numpy_hierarchies(*actions)
                    storage['action'].append_batch(action=a)

                if store_activations:
                    def to_numpy(xx):
                        return xx[0].detach().cpu().numpy()
                    a = map_hierarchies(to_numpy, x)
                    storage['activation'].append_batch(activations=a)
                
                if store_rewards:
                    storage['reward'].append_batch(reward=reward)
                
                update = first_storage.num_finished_seqs() - progress.n
                progress.update(update)
                
            #progress.total = first_storage.num_finished_seqs()
            #progress.n = first_storage.num_finished_seqs()
            progress.n = episodes
            progress.refresh()
    
    combined_storage = first_storage
    for key, s in storage.items():
        if key == first_key:
            continue
        
        try:
            combined_storage = combined_storage | s
        except:
            import pdb
            pdb.set_trace()
    
    return combined_storage

def train_epoch(
    name,
    model,
    optimizer,
    scheduler,
    data_loader,
    loss_log,
    grad_norm_clip=None,
    supervision_mode='action',
    plot=False,
):
    print('-'*80)
    print('Training: %s'%name)
    model.train()
    
    running_loss = None
    iterate = tqdm.tqdm(data_loader)
    for batch, pad in iterate:
        
        # convert observations to input tensors (x) and labels (y)
        x = model.observation_to_tensors(batch, pad)
        y = model.observation_to_label(batch, pad, supervision_mode)
        
        # forward
        x = model(**x)

        # compute loss
        loss = torch.sum(-torch.log_softmax(x, dim=-1) * y, dim=-1)
        s_i, b_i = get_seq_batch_indices(torch.LongTensor(pad))
        loss = loss[s_i, b_i].mean()

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
        loss_log.log(l)
        running_loss = running_loss or l
        running_loss = running_loss * 0.9 + l * 0.1
        iterate.set_description('loss: %.04f'%running_loss)
    
    if plot:
        chart = plot_logs(
            {'loss':loss_log},
            border='line',
            legend=True,
            min_max_y=True,
            colors={'loss':'RED'},
            x_range=(0.2,1.),
        )
        print(chart)


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
        #seq = episodes.get_seq(seq_id)
        #reward = seq['reward'][-1]
        #avg_terminal_reward += reward
        #if reward >= success_value:
        #    total_success += 1
        print('MAKE SURE THE BELOW IS RIGHT')
        import pdb
        pdb.set_trace()
        
        reward = seq['reward'][pad-1]
        avg_terminal_reward += numpy.sum(reward)
        total_success += numpy.sum(reward >= success_value)
        total_episodes += len(pad)

    #n = len(episodes)
    #if n:
    #    avg_terminal_reward /= n
    #    avg_success = total_success/n
    if total_episodes:
        avg_terminal_reward /= total_episodes
        avg_success /= total_episodes

    print('Average Terminal Reward: %f'%avg_terminal_reward)
    reward_log.log(avg_terminal_reward)

    print('Average Success: %f (%i/%i)'%(avg_success, total_success, n))
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
