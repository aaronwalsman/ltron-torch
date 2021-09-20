#!/usr/bin/env python
from ltron_torch.dataset.reassembly import (
    ReassemblyDatasetConfig, generate_offline_dataset)

if __name__ == '__main__':
    config = ReassemblyDatasetConfig.load_config('./settings.cfg')
    generate_offline_dataset(config)
