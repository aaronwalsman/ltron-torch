import torch

from ltron.gym.envs.ltron_env import async_ltron, sync_ltron
from ltron.gym.envs.break_and_make_env import (
    BreakAndMakeEnvConfig, BreakAndMakeEnv)

from ltron_torch.dataset.episode_dataset import (
    EpisodeDatasetConfig, build_episode_loader,
)
from ltron_torch.models.hand_table_transformer import (
    HandTableTransformerConfig,
    HandTableTransformer,
)
from ltron_torch.models.hand_table_lstm import (
    HandTableLSTMConfig,
    HandTableLSTM,
)
from ltron_torch.interface.break_and_make import BreakAndMakeInterfaceConfig
from ltron_torch.interface.break_and_make_hand_table_transformer import (
    BreakAndMakeHandTableTransformerInterface,
)
#from ltron_torch.interface.break_and_make_hand_table_lstm import (
#    BreakAndMakeHandTableLSTMInterface,
#)
from ltron_torch.train.optimizer import OptimizerConfig, build_optimizer
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
    EpisodeDatasetConfig,
    BreakAndMakeEnvConfig,
    BreakAndMakeInterfaceConfig,
    HandTableTransformerConfig,
    HandTableLSTMConfig,
    OptimizerConfig,
    BehaviorCloningConfig,
):
    device = 'cuda'
    model = 'transformer'
    
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
    
    print('-'*80)
    print('Building Model')
    if config.model == 'transformer':
        model = HandTableTransformer(config).to(torch.device(config.device))
        interface = BreakAndMakeHandTableTransformerInterface(model, config)
    elif config.model == 'lstm':
        model = HandTableLSTM(config).to(torch.device(config.device))
        interface = BreakAndMakeHandTableLSTMInterface(model, config)
    else:
        raise ValueError(
            'config "model" parameter must be either "transformer" or "lstm"')
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(model, config)

    print('-'*80)
    print('Building Data Loader')
    train_config = BreakAndMakeBCConfig.translate(config, split='train_split')
    train_loader = build_episode_loader(train_config)

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
        config, model, optimizer, train_loader, test_env, interface)
