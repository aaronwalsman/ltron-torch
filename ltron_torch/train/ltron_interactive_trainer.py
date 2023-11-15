import os

import numpy

import torch
import torch.multiprocessing as mp

from splendor.masks import color_index_to_byte

#from avarice.trainers import PPOTrainerConfig, PPOTrainer, env_fn_wrapper
from avarice.simple_trainers.teacher_distill import (
    TeacherDistillConfig,
    TeacherDistillTrainer,
)

from ltron.gym.envs.break_env import BreakEnvConfig
from ltron.gym.envs.make_env import MakeEnvConfig
from ltron.gym.components import ViewpointActions

from ltron_torch.models.ltron_visual_transformer import (
    LtronVisualTransformerConfig,
    LtronVisualTransformer,
    cursor_islands,
)
from ltron_torch.models.equivalence import equivalent_outcome_categorical

class LtronInteractiveTrainerConfig(
    BreakEnvConfig,
    MakeEnvConfig,
    LtronVisualTransformerConfig,
    TeacherDistillConfig,
):
    algorithm = 'teacher_distill'
    
    # change default
    optimizer = 'adamw'
    grad_norm_clipping = 1.
    distributed_world_size = None

def distributed_train(rank, world_size, config):
    if config.algorithm == 'teacher_distill':
        trainer = TeacherDistillTrainer(
            config=config,
            ModelClass=LtronVisualTransformer,
            train_env_kwargs={'config':config, 'train':True},
            eval_env_kwargs={'config':config, 'train':False},
            distributed_rank=rank,
            distributed_world_size=world_size,
        )
    try:
        trainer.train()
    finally:
        trainer.cleanup()
    

def train_ltron_teacher_distill(config=None):
    if config is None:
        config = LtronInteractiveTrainerConfig.from_commandline()
    if config.distributed_world_size is None:
        distributed_train(None, None, config)
    else:
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        mp.spawn(
            distributed_train,
            args=(config.distributed_world_size, config,),
            nprocs=config.distributed_world_size,
            join=True,
        )

def eval_ltron_teacher_distill(config=None):
    if config is None:
        config = LtronInteractiveTrainerConfig.from_commandline()
    if config.algorithm == 'teacher_distill':
        if config.distributed_world_size is None:
            trainer = TeacherDistillTrainer(
                config=config,
                ModelClass=LtronVisualTransformer,
                train_env_kwargs={'config':config, 'train':True},
                eval_env_kwargs={'config':config, 'train':False},
                distributed_rank=None,
                distributed_world_size=None,
            )
        else:
            trainer = TeacherDistillTrainer(
                config=config,
                ModelClass=LtronVisualTransformer,
                train_env_kwargs={'config':config, 'train':True},
                eval_env_kwargs={'config':config, 'train':False},
                distributed_rank=0,
                distributed_world_size=1,
            )
    trainer.evaluate(0,0)

def eval_ltron_teacher_distill_full(config=None):
    if config is None:
        config = LtronInteractiveTrainerConfig.from_commandline()
    if config.algorithm == 'teacher_distill':
        trainer = TeacherDistillTrainer(
            config=config,
            ModelClass=LtronVisualTransformer,
            train_env_kwargs={'config':config, 'train':True},
            eval_env_kwargs={'config':config, 'train':False},
        )
    trainer.evaluate(0,0)
