#!/usr/bin/env python
from ltron_torch.train.reassembly import (
    TeacherForcingReassemblyConfig, train_teacher_forcing_reassembly)

if __name__ == '__main__':
    config = TeacherForcingReassemblyConfig.load_config('./settings.cfg')
    train_teacher_forcing_reassembly(config)
