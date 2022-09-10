import random

import numpy

import torch

from conspiracy.log import Log

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.classify_env import ClassifyEnvConfig, ClassifyEnv

from ltron_torch.models.auto_transformer import (
    AutoTransformerConfig,
    AutoTransformer,
)
from ltron_torch.train.runs import redirect_output_to_new_run
from ltron_torch.train.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from ltron_torch.train.dagger import (
    DAggerConfig, dagger,
)

class ClassifyDAggerConfig(
    ClassifyEnvConfig,
    AutoTransformerConfig,
    OptimizerConfig,
    DAggerConfig,
):
    device = 'cuda'
    model = 'transformer'

    run_directory = '.'

    load_checkpoint = None
    use_checkpoint_config = False

    dataset = 'rca'
    train_split = '2_2_train'
    test_split = '2_2_test'

    parallel_envs = 4

    async_ltron = True

    seed = 1234567890

def train_classify_dagger(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = ClassifyDAggerConfig.from_commandline()

    if config.run_directory:
        redirect_output_to_new_run(run_directory=config.run_directory)
    
    print('-'*80)
    print('Dataset: %s'%config.dataset)
    print('Train Split: %s'%config.train_split)
    print('Test Split: %s'%config.test_split)
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
            config = BreakAndEstimateDAggerConfig(**checkpoint['config'])

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
        '''
        train_loss_log_checkpoint = checkpoint.get('train_loss_log', None)
        train_agreement_log_checkpoint = checkpoint.get(
            'train_agreement_log', None)
        learning_rate_log_checkpoint = checkpoint.get('learning_rate_log', None)
        train_reward_log_checkpoint = checkpoint.get('train_reward_log', None)
        train_success_log_checkpoint = checkpoint.get('train_success_log', None)
        test_reward_log_checkpoint = checkpoint.get('test_reward_log', None)
        test_success_log_checkpoint = checkpoint.get('test_success_log', None)
        '''
        logs_checkpoint = checkpoint.get('logs', None)
        logs = {name : Log(state=log) for name,log in logs_checkpoint.items()}
        
        # epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
    
    else:
        model_checkpoint = None
        optimizer_checkpoint = None
        scheduler_checkpoint = None
        
        '''
        train_loss_log_checkpoint = None
        train_agreement_log_checkpoint = None
        learning_rate_log_checkpoint = None
        train_reward_log_checkpoint = None
        train_success_log_checkpoint = None
        test_reward_log_checkpoint = None
        test_success_log_checkpoint = None
        '''
        logs = None
        
        start_epoch = 1
    
    device = torch.device(config.device)
    
    print('-'*80)
    print('Building Train Env')
    if config.async_ltron:
        vector_ltron = async_ltron
    else:
        vector_ltron = sync_ltron
    config.tile_color_render = ('transformer' in config.model)
    train_env = vector_ltron(
        config.parallel_envs,
        ClassifyEnv,
        config,
        include_expert=True,
        print_traceback=True,
    )
    test_env = vector_ltron(
        config.parallel_envs,
        ClassifyEnv,
        config,
        print_traceback=True,
    )
    
    assert (
        train_env.metadata['action_space'] == test_env.metadata['action_space'])
    
    print('-'*80)
    print('Building Model (%s)'%config.model)
    if config.model == 'transformer':
        observation_space = test_env.metadata['observation_space']
        action_space = test_env.metadata['action_space']
        no_op_action = test_env.metadata['no_op_action']
        model = AutoTransformer(
            config,
            observation_space,
            action_space,
            no_op_action,
            checkpoint=model_checkpoint,
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
    
    '''
    print('-'*80)
    print('Building Logs')
    train_loss_log = Log(state=train_loss_log_checkpoint)
    train_agreement_log = Log(state=train_agreement_log_checkpoint)
    learning_rate_log = Log(state=learning_rate_log_checkpoint)
    train_reward_log = Log(state=train_reward_log_checkpoint)
    train_success_log = Log(state=train_success_log_checkpoint)
    test_reward_log = Log(state=test_reward_log_checkpoint)
    test_success_log = Log(state=test_success_log_checkpoint)
    '''
    
    dagger(
        config,
        train_env,
        test_env,
        model,
        optimizer,
        scheduler,
        start_epoch=start_epoch,
        logs=logs,
        #train_loss_log=train_loss_log,
        #train_agreement_log=train_agreement_log,
        #learning_rate_log=learning_rate_log,
        #train_reward_log=train_reward_log,
        #train_success_log=train_success_log,
        #test_reward_log=test_reward_log,
        #test_success_log=test_success_log,
    )
