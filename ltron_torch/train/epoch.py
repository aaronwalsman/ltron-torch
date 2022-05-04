import random
import os
import io
import tarfile

import numpy

import torch
from torch.nn.utils import clip_grad_norm_

import tqdm

from ltron.hierarchy import map_hierarchies, stack_numpy_hierarchies
from ltron.gym.rollout_storage import RolloutStorage

from ltron_torch.train.optimizer import clip_grad

def rollout_epoch(
    name,
    episodes,
    env,
    model,
    store_observations,
    store_activations,
    rollout_mode='sample',
    expert_probability=0.,
    save_episodes=None,
):
    print('-'*80)
    print('Rolling Out Episodes: %s'%name)
    print('p(expert) = %.04f'%expert_probability)
    print('rollout_mode = "%s"'%rollout_mode)

    # initialize storage for observations, actions, rewards and activations
    if store_observations:
        observation_storage = RolloutStorage(env.num_envs)
    action_reward_storage = RolloutStorage(env.num_envs)

    if store_activations:
        activation_storage = RolloutStorage(env.num_envs)

    # put the model in eval mode
    model.eval()
    device = next(model.parameters()).device
    
    assert rollout_mode in ('sample', 'max')
    b = env.num_envs

    # reset
    observation = env.reset()
    terminal = numpy.ones(env.num_envs, dtype=numpy.bool)
    reward = numpy.zeros(env.num_envs)

    with torch.no_grad():
        if hasattr(model, 'initialize_memory'):
            memory = model.initialize_memory(b)
        else:
            memory = None

        #for step in tqdm.tqdm(range(steps)):
        progress = tqdm.tqdm(total=episodes)
        with progress:
            while action_reward_storage.num_finished_seqs() < episodes:
                # prep ---------------------------------------------------------
                # start new sequences if necessary
                action_reward_storage.start_new_seqs(terminal)
                if store_observations:
                    observation_storage.start_new_seqs(terminal)
                if store_activations:
                    activation_storage.start_new_seqs(terminal)

                # add latest observation to storage
                if store_observations:
                    observation_storage.append_batch(observation=observation)

                # move observations to torch and cuda
                pad = numpy.ones(b, dtype=numpy.long)
                observation = stack_numpy_hierarchies(observation)
                x = model.observation_to_tensors(
                    {'observation':observation, 'action':None}, pad, device)
                
                # compute actions ----------------------------------------------
                if hasattr(model, 'initialize_memory'):
                    if hasattr(model, 'forward_rollout'):
                        x = model.forward_rollout(terminal, **x, memory=memory)
                    else:
                        x = model(**x, memory=memory)
                    memory = x['memory']
                else:
                    if hasattr(model, 'forward_rollout'):
                        x = model.forward_rollout(terminal, **x)
                    else:
                        x = model(**x)
                
                actions = model.tensor_to_actions(x, mode=rollout_mode)
                for i in range(len(actions)):
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
                action_reward_storage.append_batch(
                    action=stack_numpy_hierarchies(*actions),
                    reward=reward,
                )

                if store_activations:
                    def to_numpy(xx):
                        return xx[0].detach().cpu().numpy()
                    a = map_hierarchies(to_numpy, x)
                    activation_storage.append_batch(activations=a)
                
                update = action_reward_storage.num_finished_seqs() - progress.n
                progress.update(update)
    
    if store_observations:
        episodes = observation_storage | action_reward_storage
    else:
        episodes = action_reward_storage
    
    if store_activations:
        episodes = episodes | activation_storage
    
    # save the trajectories if necessary
    if save_episodes is not None:
        
        print('-'*80)
        print('Saving episodes to: %s'%save_episodes)
        
        # make the folder if necessary
        directory, file_name = os.path.split(save_episodes)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        
        # write the tar file
        out = tarfile.open(save_episodes, 'w')
        for seq_id in tqdm.tqdm(episodes.finished_seqs):
            seq = episodes.get_seq(seq_id)
            buf = io.BytesIO()
            numpy.savez_compressed(buf, episode=seq)
            buf.seek(0)
            info = tarfile.TarInfo(name='episode_%06i.npz'%seq_id)
            info.size=len(buf.getbuffer())
            out.addfile(tarinfo=info, fileobj=buf)
        
        out.close()
    
    return episodes

def train_epoch(
    name,
    model,
    optimizer,
    schedulder,
    data,
    loss_log,
    grad_norm_clip=None,
    supervision_mode='action',
    plot=None,
):
    print('-'*80)
    print('Training: %s'%name)
    model.train()
    
    running_loss = None
    iterate = tqdm.tqdm(data):
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
        train_loss_log.log(l)
        running_loss = running_loss or l
        running_loss = running_loss * 0.9 + l * 0.1
        iterate.set_description('loss: %.04f'%running_loss)
    
    if plot is not None:
        chart = plot_logs(
            {'%s_loss'%name:loss_log},
            border='line',
            legend=True,
            min_max_y=True,
            colors={'%s_loss'%name:2},
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
        colors='auto',
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
    
