#!/usr/bin/env python
from ltron_torch.train.partial_reassembly_lstm import (
    BehaviorCloningReassemblyConfig, train_full_reassembly_behavior_cloning)

if __name__ == '__main__':
    config = BehaviorCloningReassemblyConfig.load_config('./settings.cfg')
    train_partial_reassembly_behavior_cloning(config)
