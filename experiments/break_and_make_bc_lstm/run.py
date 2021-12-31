#!/usr/bin/env python
from ltron_torch.train.break_and_make_bc_lstm import (
    BehaviorCloningReassemblyConfig, train_break_and_make_bc_lstm)

if __name__ == '__main__':
    config = BehaviorCloningReassemblyConfig.load_config('./settings.cfg')
    train_reassembly_behavior_cloning(config)
