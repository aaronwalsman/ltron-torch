import copy

import gymnasium as gym

from ltron.gym.envs.select_connection_point import SelectConnectionPointConfig

from ltron_torch.models.auto_transformer import (
    AutoTransformer,
    AutoTransformerConfig,
)

from avarice.trainers import PPOTrainerConfig, PPOTrainer, env_fn_wrapper

class SelectConnectionPointTrainerConfig(
    SelectConnectionPointConfig, PPOTrainerConfig, AutoTransformerConfig
):
    train_env = 'LTRON/SelectConnectionPoint-v0'
    train_dataset = 'rca'
    train_split = '2_2_train'
    train_subset = None
    train_repeat = 1
    
    eval_env = 'LTRON/SelectConnectionPoint-v0'
    eval_dataset = 'rca'
    eval_split = '2_2_test'
    eval_subset = None
    eval_repeat = 1

class SelectConnectionPointTrainer(PPOTrainer):
    def make_single_train_env_fn(self, parallel_index):
        env_fn = env_fn_wrapper(
            self.config.train_env,
            config=self.config,
            dataset_name=self.config.train_dataset,
            dataset_split=self.config.train_split,
            dataset_subset=self.config.train_subset,
            dataset_repeat=self.config.train_repeat,
            dataset_shuffle=True,
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
            parallel_index=parallel_index,
        )
        return env_fn
    
    def initialize_new_model(self, *args, **kwargs):
        no_op_action = self.train_env.call('no_op_action')[0]
        super().initialize_new_model(no_op_action, *args, **kwargs)

def train_select_connection_point():
    config = SelectConnectionPointTrainerConfig.from_commandline()
    trainer = SelectConnectionPointTrainer(config, ModelClass=AutoTransformer)
    
    breakpoint()
