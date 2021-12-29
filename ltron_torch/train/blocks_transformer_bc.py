import torch

from ltron_torch.dataset.blocks import (
    BlocksBehaviorCloningConfig, build_sequence_train_loader, build_test_env,
)
from ltron_torch.models.break_and_make_transformer import (
    BreakAndMakeTransformerConfig,
    BreakAndMakeTransformer,
    BlocksTransformerInterface,
)
from ltron_torch.train.optimizer import OptimizerConfig, build_optimizer
from ltron_torch.train.behavior_cloning import (
    BehaviorCloningConfig, behavior_cloning,
)

class BlocksTransformerBCConfig(
    BlocksBehaviorCloningConfig,
    BehaviorCloningConfig,
    OptimizerConfig,
    BreakAndMakeTransformerConfig,
):
    device = 'cuda'
    load_checkpoint = None
    
    mode_loss_weight = 1.
    table_loss_weight = 1.
    hand_loss_weight = 1.
    shape_loss_weight = 1.
    color_loss_weight = 1.

def train_blocks_transformer_bc(config):
    print('='*80)
    print('Blocks Training Setup')
    config.spatial_channels = 1
    config.mode_channels = (
        6 + len(config.block_shapes) + len(config.block_colors))
    config.set_all_dependents()
    
    print('-'*80)
    print('Building Model')
    model = BreakAndMakeTransformer(config).to(torch.device(config.device))
    
    print('-'*80)
    print('Building Optimizer')
    optimizer = build_optimizer(model, config)
    
    print('-'*80)
    print('Building Data Loader')
    train_loader = build_sequence_train_loader(config)
    
    print('-'*80)
    print('Building Test Env')
    test_env = build_test_env(config)
    
    print('-'*80)
    print('Building Interface')
    interface = BlocksTransformerInterface(model, config)
    
    behavior_cloning(
        config, model, optimizer, train_loader, test_env, interface)

def sanity_check_transformer():
    
    # some imports that I don't want to polute the main module
    from ltron.hierarchy import index_hierarchy
    import numpy
    
    # load the config
    config = BlocksTransformerBCConfig.from_commandline()
    config.shuffle=False
    config.batch_size=1
    device = torch.device(config.device)
    
    # build the model and interface
    model = BreakAndMakeTransformer(config).to(device)
    model.eval()
    interface = BlocksTransformerInterface(model, config)
    
    # build the loader
    train_loader = build_sequence_train_loader(config)
    
    # peel off the first batch
    batch, pad = next(iter(train_loader))
    
    # do a bunch of tests
    with torch.no_grad():
        
        # make sure multiple passes with a single frame produce the same results
        pad0 = numpy.ones(config.batch_size, numpy.long)
        tensors0 = interface.observation_to_tensors(
            index_hierarchy(batch['observations'], [0]), pad0)
        
        x_table0a, x_hand0a, x_token0a = model(*tensors0)
        x_table0b, x_hand0b, x_token0b = model(*tensors0)
        assert torch.allclose(x_table0a, x_table0b)
        assert torch.allclose(x_hand0a, x_hand0b)
        assert torch.allclose(x_token0a, x_token0b)
        
        # make sure a pass with a single frame matches the first frame of a
        # pass with two frames
        pad01 = numpy.ones(config.batch_size, numpy.long)*2
        tensors01 = interface.observation_to_tensors(
            index_hierarchy(batch['observations'], [0,1]), pad01)
        x_table01, x_hand01, x_token01 = model(*tensors01)
        assert torch.allclose(x_table0a, x_table01[[0]], atol=1e-6)
        assert torch.allclose(x_hand0a, x_hand01[[0]], atol=1e-6)
        assert torch.allclose(x_token0a, x_token01[[0]], atol=1e-6)
        
        # make sure a pass with two frames matches the first two frames of a
        # pass with three frames
        pad012 = numpy.ones(config.batch_size, numpy.long)*3
        tensors012 = interface.observation_to_tensors(
            index_hierarchy(batch['observations'], [0,1,2]), pad012)
        x_table012, x_hand012, x_token012 = model(*tensors012)
        assert torch.allclose(x_table01, x_table012[[0,1]], atol=1e-6)
        assert torch.allclose(x_hand01, x_hand012[[0,1]], atol=1e-6)
        assert torch.allclose(x_token01, x_token012[[0,1]], atol=1e-6)
        
        # make sure the results computed from an entire sequence match the
        # results computed frame-by-frame with use-memory
        tensors = interface.observation_to_tensors(batch['observations'], pad)
        x_table_seq, x_hand_seq, x_token_seq = model(*tensors)
        
        x_tables = []
        x_hands = []
        x_tokens = []
        seq_len = numpy.max(pad)
        total_tiles = 0
        total_tokens = 0
        for i in range(seq_len):
            print('-'*80)
            print(i)
            seq_obs = index_hierarchy(batch['observations'], [i])
            
            i_pad = numpy.ones(pad.shape, dtype=numpy.long)
            i_tensors = interface.observation_to_tensors(seq_obs, i_pad)
            if i == 0:
                use_memory = torch.zeros(pad.shape, dtype=torch.long).to(device)
            else:
                use_memory = torch.ones(pad.shape, dtype=torch.long).to(device)
            
            total_tiles += i_tensors[0].shape[0]
            total_tokens += i_tensors[4].shape[0]
            
            xi_table, xi_hand, xi_token = model(
                *i_tensors, use_memory=use_memory)
            x_tables.append(xi_table)
            x_hands.append(xi_hand)
            x_tokens.append(xi_token)
            
        x_table_cat = torch.cat(x_tables, dim=0)
        x_hand_cat = torch.cat(x_hands, dim=0)
        x_token_cat = torch.cat(x_tokens, dim=0)
        
    try:
        assert torch.allclose(x_table_seq, x_table_cat, atol=1e-6)
        assert torch.allclose(x_hand_seq, x_hand_cat, atol=1e-6)
        assert torch.allclose(x_token_seq, x_token_cat, atol=1e-6)
    except AssertionError:
        print('BORK')
        import pdb
        pdb.set_trace()
        
    print('we did it!')
