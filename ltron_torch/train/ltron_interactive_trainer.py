import numpy

import torch
import torch.nn.functional as F

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

def train_ltron_teacher_distill(config=None):
    if config is None:
        config = LtronInteractiveTrainerConfig.from_commandline()
    if config.algorithm == 'teacher_distill':
        trainer = TeacherDistillTrainer(
            config=config,
            ModelClass=LtronVisualTransformer,
            train_env_kwargs={'config':config, 'train':True},
            eval_env_kwargs={'config':config, 'train':False},
        )
    trainer.train()

def eval_ltron_teacher_distill(config=None):
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
