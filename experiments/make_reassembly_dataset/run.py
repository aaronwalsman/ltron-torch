#!/usr/bin/env python
import sys
from ltron.gym.envs.break_and_make_env import BreakAndMakeEnvConfig
from ltron_torch.dataset.break_and_make import (
    BreakAndMakeDatasetConfig, generate_offline_dataset)

if __name__ == '__main__':
    #config = BreakAndMakeDatasetConfig.load_config('./settings.cfg')
    config = BreakAndMakeEnvConfig.load_config('./settings.cfg')
    if len(sys.argv) > 1:
        start, end = sys.argv[1:]
        config.start = int(start)
        config.end = int(end)
    generate_offline_dataset(config)
