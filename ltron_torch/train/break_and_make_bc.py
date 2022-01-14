import torch

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnvConfig, BreakAndMakeEnv)

from ltron_torch.dataset.break_and_make_dataset import (
    BreakAndMakeDatasetConfig, BreakAndMakeDataset
)
from ltron_torch.dataset.episode_dataset import build_episode_loader
from ltron_torch.models.hand_table_transformer import (
    HandTableTransformerConfig,
    HandTableTransformer,
)
from ltron_torch.models.hand_table_lstm import (
    HandTableLSTMConfig,
    HandTableLSTM,
)
from ltron_torch.interface.break_and_make_hand_table_transformer import (
    BreakAndMakeHandTableTransformerInterfaceConfig,
    BreakAndMakeHandTableTransformerInterface,
)
#from ltron_torch.interface.break_and_make_hand_table_lstm import (
#    BreakAndMakeHandTableLSTMInterface,
#)
from ltron_torch.train.optimizer import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from ltron_torch.train.behavior_cloning import (
    BehaviorCloningConfig, behavior_cloning,
)

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
    BreakAndMakeDatasetConfig,
    BreakAndMakeEnvConfig,
    BreakAndMakeHandTableTransformerInterfaceConfig,
    HandTableTransformerConfig,
    HandTableLSTMConfig,
    OptimizerConfig,
    BehaviorCloningConfig,
):
    device = 'cuda'
    model = 'transformer'
    
    load_checkpoint = None
    use_checkpoint_config = False
    
    dataset = 'random_construction_6_6'
    train_split = 'train_episodes'
    test_split = 'test'
    
    num_test_envs = 4
    
    num_modes = 23 # 7 + 7 + 3 + 2 + 1 + 2 + 1
    num_shapes = 6
    num_colors = 6
    
    table_channels = 2
    hand_channels = 2

def train_break_and_make_bc(config=None):
    if config is None:
        print('='*80)
        print('Loading Config')
        config = BreakAndMakeBCConfig.from_commandline()
    
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
        optimizer_checkpoint = checkpoint['optimizer']
        scheduler_checkpoint = checkpoint['scheduler']
    else:
        model_checkpoint = None
        optimizer_checkpoint = None
        scheduler_checkpoint = None
    
    device = torch.device(config.device)
    
    print('-'*80)
    print('Building Model')
    if config.model == 'transformer':
        model = HandTableTransformer(config, model_checkpoint).to(device)
        interface = BreakAndMakeHandTableTransformerInterface(model, config)
    elif config.model == 'lstm':
        model = HandTableLSTM(config, model_checkpoint).to(device)
        interface = BreakAndMakeHandTableLSTMInterface(model, config)
    else:
        raise ValueError(
            'config "model" parameter must be either "transformer" or "lstm"')
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(config, model, optimizer_checkpoint)
    
    print('-'*80)
    print('Building Scheduler')
    scheduler = build_scheduler(config, optimizer, scheduler_checkpoint)
    
    print('-'*80)
    print('Building Data Loader')
    train_config = BreakAndMakeBCConfig.translate(config, split='train_split')
    train_dataset = BreakAndMakeDataset(train_config)
    train_loader = build_episode_loader(train_config, train_dataset)

    print('-'*80)
    print('Building Test Env')
    test_config = BreakAndMakeBCConfig.translate(config, split='test_split')
    test_env = sync_ltron(
        config.num_test_envs,
        BreakAndMakeEnv,
        test_config,
        print_traceback=True,
    )

    behavior_cloning(
        config, model, optimizer, scheduler, train_loader, test_env, interface)
