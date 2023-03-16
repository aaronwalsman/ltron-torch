from avarice.trainers import PPOTrainerConfig, PPOTrainer, env_fn_wrapper

from ltron.gym.wrappers.break_vector_reward import (
    BreakVectorEnvAssemblyRewardWrapper,
)

from ltron_torch.models.ltron_visual_transformer import LtronVisualTransformer
from ltron_torch.models.ltron_resnet import LtronResNet
from ltron_torch.train.ltron_ppo_trainer import (
    LtronPPOTrainerConfig,
    LtronPPOTrainer,
)
from ltron_torch.train.ltron_interactive_trainer import (
    LtronInteractiveTrainerConfig,
    train_ltron_teacher_distill,
)

class BreakTrainerConfig(LtronInteractiveTrainerConfig):
    #, LtronPPOTrainerConfig):
    #algorithm = 'ppo'
    brick_identification_mode = 'assembly'

class BreakPPOTrainer(LtronPPOTrainer):
    def initialize_new_vector_env(self, env_fns):
        vector_env = super().initialize_new_vector_env(env_fns)
        if self.config.brick_identification_mode == 'assembly':
            vector_env = BreakVectorEnvAssemblyRewardWrapper(vector_env)
        else:
            raise NotImplementedError
        
        return vector_env

def train_break():
    print('Loading Config')
    config = BreakTrainerConfig.from_commandline()
    if config.algorithm == 'ppo':
        trainer = BreakPPOTrainer(config, ModelClass=LtronVisualTransformer)
        trainer.train()
    elif config.algorithm == 'teacher_distill':
        train_ltron_teacher_distill(config)
