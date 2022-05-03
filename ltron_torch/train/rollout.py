import random
import os
import io
import tarfile

import numpy

import torch

import tqdm

from ltron.hierarchy import map_hierarchies, stack_numpy_hierarchies
from ltron.gym.rollout_storage import RolloutStorage

def rollout_epoch(
    epoch,
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
    print('Rolling out episodes with p(expert) = %.04f'%expert_probability)

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
                #x = interface.observation_to_tensors(
                x = model.observation_to_tensors(
                    {'observation':observation, 'action':None}, pad, device)
                
                # compute actions ----------------------------------------------
                if hasattr(model, 'initialize_memory'):
                    #if hasattr(interface, 'forward_rollout'):
                    if hasattr(model, 'forward_rollout'):
                        #x = interface.forward_rollout(
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
                #nas = []
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
                        
                        #action_space = env.metadata['action_space']
                        #n, a = action_space.unravel(actions[i])
                
                #for action in actions:
                #    n, a = env.metadata['action_space'].unravel_index(action)
                
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
                    #a = interface.numpy_activations(x)
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

