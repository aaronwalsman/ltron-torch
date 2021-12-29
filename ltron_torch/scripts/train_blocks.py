from ltron_torch.train.blocks_transformer_bc import (
    BlocksTransformerBCConfig, train_blocks_transformer_bc)

def main():
    config = BlocksTransformerBCConfig.from_commandline()
    train_blocks_transformer_bc(config)
