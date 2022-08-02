import random
import argparse

import numpy

import torch

from conspiracy.log import Log

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnvConfig,
    BreakAndMakeEnv,
)

from ltron_torch.models.auto_transformer import (
    AutoTransformerConfig,
    AutoTransformer,
)
from ltron_torch.train.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from ltron_torch.train.behavior_cloning import (
    BehaviorCloningConfig, behavior_cloning,
)
from ltron_torch.dataset.tar_dataset import make_tar_dataset_and_loader

# TODO: This file is very similar to blocks_bc.py but with different
# defaults and a different interface.  These could probably be consolidated,
# but I'm sick of tearing up everything every five minutes.  Also, let's wait
# and see what the different interface configs look like and how easy it would
# be to reconcile them.  I also need a better way for owner configs to
# conditionally overwrite stuff they inherit.  For example table_channels below
# is 2 for BreakAndMake but 1 for Blocks, and I don't want / can't have that
# specified in the config file.  It should be overwritten based on the task,
# using set_dependents, but this is error-prone because if earlier
# set_dependents calls use the old value, then things might get really messy.
# One option is just to call set_dependents again after the override?  It would
# be nice to have a simple mechanism for this kind of thing though.  Maybe a new
# overrides method or something?  Whatever, avoid the issue for now.

class BreakAndMakeBCConfig(
    BreakAndMakeEnvConfig,
    AutoTransformerConfig,
    OptimizerConfig,
    BehaviorCloningConfig,
):
    device = 'cuda'
    model = 'transformer'
    
    load_checkpoint = None
    use_checkpoint_config = False
    
    dataset = 'rca'
    train_split = '2_2_sss_train_episodes'
    test_split = '2_2_test'
    train_subset = None
    test_subset = None
    
    parallel_envs = 4
    
    async_ltron = True
    
    seed = 1234567890

def train_break_and_make_bc(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeBCConfig.from_commandline()

    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    if config.load_checkpoint is not None:
        print('-'*80)
        print('Loading Checkpoint')
        checkpoint = torch.load(config.load_checkpoint)
        if config.use_checkpoint_config:
            assert 'config' in checkpoint, (
                '"config" not found in checkpoint: %s'%config.load_checkpoint)
            config = BreakAndMakeBCConfig(**checkpoint['config'])
        
        # model
        model_checkpoint = checkpoint['model']
        
        # optimizer
        if config.train_frequency:
            optimizer_checkpoint = checkpoint['optimizer']
        else:
            optimizer_checkpoint = None
        
        # scheduler
        scheduler_checkpoint = checkpoint['scheduler']
        
        # logs
        train_loss_log_checkpoint = checkpoint.get('train_loss_log', None)
        test_reward_log_checkpoint = checkpoint.get('test_reward_log', None)
        test_success_log_checkpoint = checkpoint.get('test_success_log', None)
        
        # epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        model_checkpoint = None
        optimizer_checkpoint = None
        scheduler_checkpoint = None
        
        train_loss_log_checkpoint = None
        test_reward_log_checkpoint = None
        test_success_log_checkpoint = None
        
        start_epoch = 1
    
    device = torch.device(config.device)
    
    print('-'*80)
    print('Building Train Data Loader')
    train_dataset, train_loader = make_tar_dataset_and_loader(
        config.dataset,
        config.train_split,
        config.batch_size,
        config.workers,
        subset=config.train_subset,
        shuffle=True,
    )
    
    print('-'*80)
    print('Building Test Env')
    test_config = BreakAndMakeBCConfig.translate(
        config,
        split='test_split',
        subset='test_subset',
    )
    if config.async_ltron:
        vector_ltron = async_ltron
    else:
        vector_ltron = sync_ltron
    test_env = vector_ltron(
        config.parallel_envs,
        BreakAndMakeEnv,
        test_config,
        print_traceback=True,
    )
    
    print('-'*80)
    print('Building Model (%s)'%config.model)
    if config.model == 'transformer':
        observation_space = test_env.metadata['observation_space']
        action_space = test_env.metadata['action_space']
        model = AutoTransformer(
            config,
            observation_space,
            action_space,
            model_checkpoint,
        ).to(device)
    else:
        raise ValueError(
            'config "model" parameter ("%s") must be '
            '"transformer"'%config.model
        )
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(config, model, optimizer_checkpoint)
    
    print('-'*80)
    print('Building Scheduler')
    scheduler = build_scheduler(config, optimizer, scheduler_checkpoint)
    
    print('-'*80)
    print('Building Logs')
    train_loss_log = Log(state=train_loss_log_checkpoint)
    test_reward_log = Log(state=test_reward_log_checkpoint)
    test_success_log = Log(state=test_success_log_checkpoint)

    behavior_cloning(
        config,
        train_loader,
        test_env,
        model,
        optimizer,
        scheduler,
        start_epoch=start_epoch,
        train_loss_log=train_loss_log,
        test_reward_log=test_reward_log,
        test_success_log=test_success_log,
    )

def plot_break_and_make_bc(checkpoint=None):
    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('checkpoint', type=str)
        args = parser.parse_args()
        checkpoint = args.checkpoint
    
    data = torch.load(open(checkpoint, 'rb'), map_location='cpu')
    train_log = BreakAndMakeInterface.make_train_log()
    train_log.set_state(data['train_log'])
    test_log = BreakAndMakeInterface.make_test_log()
    test_log.set_state(data['test_log'])
    
    train_chart = train_log.plot_grid(
        topline=True, legend=True, minmax_y=True, height=40, width=72)
    print('='*80)
    print('Train Plots')
    print(train_chart)
    
    test_chart = test_log.plot_sequential(
        legend=True, minmax_y=True, height=60, width=160)
    print('='*80)
    print('Test Plots')
    print('-'*80)
    print(test_chart)

def eval_break_and_make_bc(checkpoint=None):
    if checkpoint is None:
        parser = argparse.ArgumentParser()
        parser.add_argument
