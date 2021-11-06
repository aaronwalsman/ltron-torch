#!/usr/bin/env python
from ltron_torch.train.reassembly_resnet import (
    BehaviorCloningReassemblyConfig, train_disassembly_behavior_cloning)

if __name__ == '__main__':
    config = BehaviorCloningReassemblyConfig.load_config('./settings.cfg')
    train_disassembly_behavior_cloning(config)
