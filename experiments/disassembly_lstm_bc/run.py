#!/usr/bin/env python
from ltron_torch.train.reassembly_lstm import (
    BehaviorCloningReassemblyConfig, train_reassembly_behavior_cloning)

if __name__ == '__main__':
    config = BehaviorCloningReassemblyConfig.load_config('./settings.cfg')
    train_reassembly_behavior_cloning(config)
