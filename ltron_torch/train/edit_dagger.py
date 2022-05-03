import random
import argparse

import numpy

import torch

from conspiracy.log import Log

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.multiscreen_edit_env import (
    MultiScreenEditEnvConfig, MultiScreenEditEnv
)

from ltron_torch.models.auto_transformer import (
    AutoTransformerConfig,
    AutoTransformer,
)
#from ltron_torch.interface.auto_transformer import (
#    AutoTransformerInterfaceConfig,
#    AutoTransformerInterface,
#)
from ltron_torch.train.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from ltron_torch.train.dagger import (
    DAggerConfig, dagger,
)

class EditDAggerConfig(
    #AutoTransformerInterfaceConfig,
    MultiScreenEditEnvConfig,
    AutoTransformerConfig,
    OptimizerConfig,
    DAggerConfig,
):
    device = 'cuda'
    model = 'transformer'
    
    load_checkpoint = None
    use_checkpoint_config = False
    
    dataset = 'random_construction'
    train_split = '6b_6c_2i_ldraw_train'
    test_split = '6b_6c_2i_ldraw_test'
    train_subset = None
    test_subset = None
    
    parallel_envs = 4
    
    #num_modes = 23 # 7 + 7 + 3 + 2 + 1 + 2 + 1 (+2 for factored cursor)
    factor_cursor_distribution = False
    num_shapes = 6
    num_colors = 6
    
    table_channels = 2
    hand_channels = 2
    
    async_ltron = True
    
    seed = 1234567890
    
    allow_snap_flip = False

def train_edit_dagger(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = EditDAggerConfig.from_commandline()

    random.seed(config.seed)
    numpy.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    
    if config.load_checkpoint is not None:
        print('-'*80)
        print('Loading Checkpoint')
        checkpoint = torch.load(config.load_checkpoint)
        # this is bad because it overwrites things specified on the command line
        # if you want to do this, find a better way, and put it in the Config
        # class itself (which is hard because the Config class is in ltron and 
        # doesn't know about pytorch
        # ok here's the compromise: I just added "use_checkpoint_config" which
        # turns on this behavior
        if config.use_checkpoint_config:
            assert 'config' in checkpoint, (
                '"config" not found in checkpoint: %s'%config.load_checkpoint)
            config = BreakAndMakeBCConfig(**checkpoint['config'])
        model_checkpoint = checkpoint['model']
        if config.train_frequency:
            optimizer_checkpoint = checkpoint['optimizer']
        else:
            optimizer_checkpoint = None
        
        # scheduler
        scheduler_checkpoint = checkpoint['scheduler']
        
        # logs
        train_loss_log_checkpoint = checkpoint.get('train_loss_log', None)
        train_reward_log_checkpoint = checkpoint.get('train_reward_log', None)
        train_success_log_checkpoint = checkpoint.get('train_success_log', None)
        test_reward_log_checkpoint = checkpoint.get('test_reward_log', None)
        test_success_checkpoint = checkpoint.get('test_success_log', None)
        
        # epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
    else:
        model_checkpoint = None
        optimizer_checkpoint = None
        scheduler_checkpoint = None
        train_loss_log_checkpoint = None
        train_reward_log_checkpoint = None
        test_reward_log_checkpoint = None
        start_epoch = 1
    
    if config.factor_cursor_distribution:
        config.num_modes = 25
    else:
        config.num_modes = 23
    
    if config.allow_snap_flip:
        config.num_modes += 4
    
    config.multiscreen = True
    
    device = torch.device(config.device)
    
    print('-'*80)
    print('Building Train and Test Env')
    train_config = EditDAggerConfig.translate(
        config,
        split='train_split',
        subset='train_subset',
    )
    test_config = EditDAggerConfig.translate(
        config,
        split='test_split',
        subset='test_subset',
    )
    if config.async_ltron:
        vector_ltron = async_ltron
    else:
        vector_ltron = sync_ltron
    train_env = vector_ltron(
        config.parallel_envs,
        MultiScreenEditEnv,
        train_config,
        print_traceback=True,
    )
    test_env = vector_ltron(
        config.parallel_envs,
        MultiScreenEditEnv,
        test_config,
        print_traceback=True,
    )
    
    print('-'*80)
    print('Building Interface (%s)'%config.model)
    #interface = AutoTransformerInterface(config, train_env)
    
    print('-'*80)
    print('Building Model (%s)'%config.model)
    if config.model == 'transformer':
        observation_space = train_env.metadata['observation_space']
        action_space = train_env.metadata['action_space']
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
    print('Loading Logs')
    train_loss_log = Log()
    train_reward_log = Log()
    test_reward_log = Log()
    if train_loss_log_checkpoint is not None:
        train_loss_log.set_state(train_loss_log_checkpoint)
    if train_reward_log_checkpoint is not None:
        train_reward_log.set_state(train_reward_log_checkpoint)
    if test_reward_log_checkpoint is not None:
        test_reward_log.set_state(test_reward_log_checkpoint)
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(config, model, optimizer_checkpoint)
    
    print('-'*80)
    print('Building Scheduler')
    scheduler = build_scheduler(config, optimizer, scheduler_checkpoint)
    
    dagger(
        config,
        train_env,
        test_env,
        #interface,
        model,
        optimizer,
        scheduler,
        start_epoch=start_epoch,
        train_loss_log=train_loss_log,
        train_reward_log=train_reward_log,
        test_reward_log=test_reward_log,
    )
