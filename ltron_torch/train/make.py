from avarice.trainers import PPOTrainerConfig, PPOTrainer, env_fn_wrapper

from ltron_torch.models.ltron_visual_transformer import LtronVisualTransformer
from ltron_torch.models.ltron_resnet import LtronResNet
from ltron_torch.train.ltron_ppo_trainer import (
    LtronPPOTrainerConfig,
    LtronPPOTrainer,
)
from ltron_torch.train.ltron_interactive_trainer import (
    LtronInteractiveTrainerConfig,
    train_ltron_teacher_distill,
    eval_ltron_teacher_distill,
)

class MakeTrainerConfig(LtronInteractiveTrainerConfig):
    pass

class MakeTrainer(LtronPPOTrainer):
    pass

def train_make():
    print('Loading Config')
    config = MakeTrainerConfig.from_commandline()
    if config.algorithm == 'ppo':
        trainer = MakePPOTrainer(config, ModelClass=LtronVisualTransformer)
        trainer.train()
    elif config.algorithm == 'teacher_distill':
        train_ltron_teacher_distill(config)

def eval_make():
    print('Loading Config')
    config = MakeTrainerConfig.from_commandline()
    if config.algorithm == 'ppo':
        raise NotImplementedError
    elif config.algorithm == 'teacher_distill':
        eval_ltron_teacher_distill(config)
