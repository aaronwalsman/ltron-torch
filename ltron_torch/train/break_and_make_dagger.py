import random

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
from ltron_torch.train.dagger import (
    DAggerConfig, dagger,
)

class BreakAndMakeDAggerConfig(
    BreakAndMakeEnvConfig,
    AutoTransformerConfig,
    OptimizerConfig,
    DAggerConfig,
):
    device = 'cuda'
    model = 'transformer'
    
    load_checkpoint = None
    use_checkpoint_config = False
    
    dataset = 'rca'
    train_split = '2_2_train'
    test_split = '2_2_test'
    train_subset = None
    test_subset = None

    parallel_envs = 4

    async_ltron = True

    seed = 1234567890

def train_break_and_make_dagger(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeDAggerConfig.from_commandline()
    
    print('-'*80)
    print('Dataset: %s'%config.dataset)
    print('Train Split: %s'%config.train_split)
    if config.train_subset:
        print('  Subset: %s'%config.train_subset)
    print('Test Split: %s'%config.test_split)
    if config.test_subset:
        print('  Subset: %s'%config.test_subset)
    print('Parallel Envs: %i'%config.parallel_envs)
    
    print('-'*80)
    print('Setting Random Seed: %i'%config.seed)
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
            config = BreakAndMakeDAggerConfig(**checkpoint['config'])

        # model
        model_checkpoint = checkpoint['model']

        # optimizer
        if config.train_frequency:
            if 'optimizer' in checkpoint:
                optimizer_checkpoint = checkpoint['optimizer']
            else:
                print('WARNING: NO OPTIMIZER FOUND IN CHECKPOINT')
                optimizer_checkpoint = None
        else:
            optimizer_checkpoint = None

        # scheduler
        scheduler_checkpoint = checkpoint['scheduler']

        # logs
        train_loss_log_checkpoint = checkpoint.get('train_loss_log', None)
        train_agreement_log_checkpoint = checkpoint.get(
            'train_agreement_log', None)
        learning_rate_log_checkpoint = checkpoint.get('learning_rate_log', None)
        train_reward_log_checkpoint = checkpoint.get('train_reward_log', None)
        train_success_log_checkpoint = checkpoint.get('train_success_log', None)
        test_reward_log_checkpoint = checkpoint.get('test_reward_log', None)
        test_success_log_checkpoint = checkpoint.get('test_success_log', None)
        
        # epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        model_checkpoint = None
        optimizer_checkpoint = None
        scheduler_checkpoint = None

        train_loss_log_checkpoint = None
        train_agreement_log_checkpoint = None
        learning_rate_log_checkpoint = None
        train_reward_log_checkpoint = None
        train_success_log_checkpoint = None
        test_reward_log_checkpoint = None
        test_success_log_checkpoint = None

        start_epoch = 1
    
    device = torch.device(config.device)
    
    print('-'*80)
    print('Building Train Env')
    if config.async_ltron:
        vector_ltron = async_ltron
    else:
        vector_ltron = sync_ltron
    train_config = BreakAndMakeDAggerConfig.translate(
        config,
        split='train_split',
        subset='train_subset',
    )
    train_config.tile_color_render = ('transformer' in config.model)
    train_env = vector_ltron(
        config.parallel_envs,
        BreakAndMakeEnv,
        train_config,
        include_expert=True,
        print_traceback=True,
    )
    
    print('Building Test Env')
    test_config = BreakAndMakeDAggerConfig.translate(
        config,
        split='test_split',
        subset='test_subset',
    )
    test_config.tile_color_render = ('transformer' in config.model)
    test_env = vector_ltron(
        config.parallel_envs,
        BreakAndMakeEnv,
        test_config,
        print_traceback=True,
    )
    
    assert (
        train_env.metadata['action_space'] == test_env.metadata['action_space'])
    
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
    train_agreement_log = Log(state=train_agreement_log_checkpoint)
    learning_rate_log = Log(state=learning_rate_log_checkpoint)
    train_reward_log = Log(state=train_reward_log_checkpoint)
    train_success_log = Log(state=train_success_log_checkpoint)
    test_reward_log = Log(state=test_reward_log_checkpoint)
    test_success_log = Log(state=test_success_log_checkpoint)
    
    dagger(
        config,
        train_env,
        test_env,
        model,
        optimizer,
        scheduler,
        start_epoch=start_epoch,
        train_loss_log=train_loss_log,
        train_agreement_log=train_agreement_log,
        learning_rate_log=learning_rate_log,
        train_reward_log=train_reward_log,
        train_success_log=train_success_log,
        test_reward_log=test_reward_log,
        test_success_log=test_success_log,
    )
