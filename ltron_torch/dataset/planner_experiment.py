import time

import numpy

from ltron.dataset.paths import get_dataset_info, get_dataset_paths
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnv, BreakAndMakeEnvConfig)
#from ltron.plan.roadmap import Roadmap, RoadmapPlanner
from ltron.plan.break_and_make import plan_break_and_make

def planner_experiment():
    dataset = 'carbon_star'
    split = 'no_wheels'
    dataset_info = get_dataset_info(dataset)
    shape_ids = dataset_info['shape_ids']
    color_ids = dataset_info['color_ids']
    dataset_paths = get_dataset_paths(dataset, split, subset=None)
    
    for i in range(10):
        env_config = BreakAndMakeEnvConfig(dataset=dataset)
        env_config.dataset_reset_mode = 'single_pass'
        env_config.split = split
        env_config.include_score = True
        env = BreakAndMakeEnv(
            env_config, rank=i, size=10, print_traceback=True)
        #env.time = True
        
        start_observation = env.reset()
        
        o, a, r = plan_break_and_make(
            env, start_observation, shape_ids, color_ids)
        
        t_first = numpy.array([oo['step'] for oo in o[:-1]])
        t_second = numpy.array([oo['step'] for oo in o[1:]])
        
        diff = t_first + 1 - t_second
        print(diff)
        print(sum(diff))
        
        import pdb
        pdb.set_trace()
        
        '''
        full_assembly = start_observation['table_assembly']
        full_state = env.get_state()
        
        empty_assembly = {
            'shape' : numpy.zeros_like(full_assembly['shape']),
            'color' : numpy.zeros_like(full_assembly['color']),
            'pose' : numpy.zeros_like(full_assembly['pose']),
            'edges' : numpy.zeros_like(full_assembly['edges']),
        }
        
        break_roadmap = Roadmap(
            env, full_state, empty_assembly, shape_ids, color_ids,
            target_steps_per_view_change=2)
        
        t0 = time.time()
        break_path = break_roadmap.plan(timeout=60)
        print('break found:')
        for a, b in zip(break_path[:-1], break_path[1:]):
            print(next(iter(a-b)))
        t1 = time.time()
        print('elapsed: %f'%(t1-t0))
        
        action = env.no_op_action()
        action['phase'] = 1
        empty_state = env.get_state()
        
        make_roadmap = Roadmap(
            env, empty_state, full_assembly, shape_ids, color_ids,
            target_steps_per_view_change=2)
        
        t0 = time.time()
        make_path = make_roadmap.plan(timeout=60)
        print('make found:')
        for a, b in zip(make_path[:-1], make_path[1:]):
            print(next(iter(b-a)))
        t1 = time.time()
        print('elapsed: %f'%(t1-t0))
        
        ob, ab, rb = break_roadmap.get_observation_action_reward_seq(break_path)
        om, am, rm = make_roadmap.get_observation_action_reward_seq(make_path)
        
        t_first = numpy.array([o['step'] for o in om[:-1]])
        t_second = numpy.array([o['step'] for o in om[1:]])
        
        diff = t_first + 1 - t_second
        print(diff)
        print(sum(diff))
        
        import pdb
        pdb.set_trace()
        '''

planner_experiment()
