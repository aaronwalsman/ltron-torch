#!/usr/bin/env python
from ltron_torch.train.compressed_transformer import (
    TrainConfig, train_compressed_transformer)

if __name__ == '__main__':
    config = TrainConfig.load_config('./settings.cfg')
    train_compressed_transformer(config)
