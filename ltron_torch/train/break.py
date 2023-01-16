import copy

from ltron.gym.envs.break_env import BreakEnvConfig
from ltron.gym.wrappers.break_vector_reward import (
    BreakVectorEnvAssemblyRewardWrapper,
)

from ltron_torch.models.ltron_visual_transformer import (
    LtronVisualTransformerConfig,
    LtronVisualTransformer,
)

#from ltron_torch.models.auto_transformer import (
#    AutoTransformer,
#    AutoTransformerConfig,
#)

from avarice.trainers import PPOTrainerConfig, PPOTrainer, env_fn_wrapper

class BreakTrainerConfig(
    BreakEnvConfig, PPOTrainerConfig, LtronVisualTransformerConfig
):
    # override defaults
    channels = 256
    heads = 4
    
    batch_size = 32
    parallel_train_envs = 16
    parallel_eval_envs = 16
    
    recurrent = 1
    
    train_env = 'LTRON/Break-v0'
    train_dataset = 'rca'
    train_split = '2_2_train'
    train_subset = None
    train_repeat = 1
    
    eval_env = 'LTRON/Break-v0'
    eval_dataset = 'rca'
    eval_split = '2_2_test'
    eval_subset = None
    eval_repeat = 1
    
    brick_identification_mode = 'assembly'

class BreakTrainer(PPOTrainer):
    def make_single_train_env_fn(self, parallel_index):
        env_fn = env_fn_wrapper(
            self.config.train_env,
            config=self.config,
            dataset_name=self.config.train_dataset,
            dataset_split=self.config.train_split,
            dataset_subset=self.config.train_subset,
            dataset_repeat=self.config.train_repeat,
            dataset_shuffle=True,
            train=True,
            parallel_index=parallel_index,
        )
        return env_fn
    
    def make_single_eval_env_fn(self, parallel_index):
        env_fn = env_fn_wrapper(
            self.config.eval_env,
            config=self.config,
            dataset_name=self.config.eval_dataset,
            dataset_split=self.config.eval_split,
            dataset_subset=self.config.eval_subset,
            dataset_repeat=self.config.eval_repeat,
            dataset_shuffle=False,
            train=False,
            parallel_index=parallel_index,
        )
        return env_fn
    
    def initialize_new_vector_env(self, env_fns):
        vector_env = super().initialize_new_vector_env(env_fns)
        if self.config.brick_identification_mode == 'assembly':
            vector_env = BreakVectorEnvAssemblyRewardWrapper(vector_env)
        else:
            raise NotImplementedError
        
        return vector_env
    
    def initialize_new_model(self, *args, **kwargs):
        return super().initialize_new_model(
            *args, decode_mode='actor_critic', **kwargs)

def train_break():
    config = BreakTrainerConfig.from_commandline()
    trainer = BreakTrainer(config, ModelClass=LtronVisualTransformer)
    
    trainer.train()
