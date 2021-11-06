#!/usr/bin/env python
import sys
from ltron_torch.dataset.reassembly import (
    ReassemblyDatasetConfig, generate_offline_dataset)

if __name__ == '__main__':
    config = ReassemblyDatasetConfig.load_config('./settings.cfg')
    if len(sys.argv) > 1:
        start, end = sys.argv[1:]
        config.start = int(start)
        config.end = int(end)
    generate_offline_dataset(config)
