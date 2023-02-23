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

class BreakPPOTrainerConfig(LtronPPOTrainerConfig):
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
    config = BreakPPOTrainerConfig.from_commandline()
    trainer = BreakPPOTrainer(config, ModelClass=LtronVisualTransformer)
    #trainer = BreakPPOTrainer(config, ModelClass=LtronResNet)
    
    trainer.train()
