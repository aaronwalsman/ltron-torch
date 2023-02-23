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

class MakePPOTrainerConfig(LtronPPOTrainerConfig):
    pass

class MakePPOTrainer(LtronPPOTrainer):
    pass

def train_make():
    print('Loading Config')
    config = MakePPOTrainerConfig.from_commandline()
    trainer = MakePPOTrainer(config, ModelClass=LtronVisualTransformer)
    
    trainer.train()
