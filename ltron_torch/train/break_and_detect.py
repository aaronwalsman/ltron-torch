import random
import os

import numpy

import torch

import tqdm

from splendor.image import save_image

from conspiracy import Log, plot_logs

from steadfast.hierarchy import hierarchy_getitem

from avarice.optimizer import (
    OptimizerConfig, build_optimizer, linear_warmup_cosine_decay_scheduler)
from avarice.rng import seed_rngs
from avarice.checkpointer import Checkpointer
from avarice.scheduler import DynamicScheduler
from avarice.vector_env import make_vector_env
from avarice.data import (
    numpy_to_torch,
    hierarchy_to_device,
    ParallelRolloutDataset,
)
from avarice.evaluation import evaluate_policy

class BreakAndDetectConfig(OptimizerConfig):
    checkpoint_directory = './checkpoint'
    checkpoint_frequency = 1
    max_checkpoints = None
    save_checkpoint_on_exit = True
    save_checkpoint_on_error = True
    load_checkpoint = None
    
    train_env = 'NONE'
    eval_env = 'NONE'
    parallel_envs = 8
    async_envs = True
    recurrence = 1
    
    eval_episodes = 128
    visualization_directory = './visualization'
    evaluation_frequency = 1
    eval_mode = 'max'
    visualize_train = False
    visualize_evaluation = False
    
    status_frequency = 1
    
    device = 'cuda'
    
    steps = 80000
    
    seed = 0
    
    detect_anomaly = False
    
    batch_size = 256
    epochs = 4
    steps_per_epoch = 1024
    
    on_policy = True
    off_policy_sprinkle = 0.

class BreakAndDetectTrainer:
    def __init__(self,
        config=None,
        BreakModelClass=None,
        DetectModelClass=None,
        train_env_kwargs=None,
        eval_env_kwargs=None,
    ):
        
        # config
        if config is None:
            config = BreakAndDetectConfig.from_commandline()
        
        print('Config:')
        print(config)
        
        # store and initialize variables
        self.config = config
        self.device = torch.device(config.device)
        self.steps = 0
        self.batches = 0
        assert self.config.steps_per_epoch % self.config.parallel_envs == 0
        
        # turn on anomaly detection
        if config.detect_anomaly:
            torch.autograd.set_detect_anomaly(True)
        
        # seed
        seed_rngs(config.seed)
        
        # make the visualization directory
        if config.visualize_evaluation or config.visualize_train:
            if not os.path.exists(config.visualization_directory):
                os.makedirs(config.visualization_directory)
        
        # make the checkpointer, evaluator and status schedulers
        self.checkpointer = Checkpointer(
            self,
            directory=config.checkpoint_directory,
            frequency=config.checkpoint_frequency,
            max_checkpoints=config.max_checkpoints,
            save_on_exit=config.save_checkpoint_on_exit,
            save_on_error=config.save_checkpoint_on_error,
        )
        self.evaluator = DynamicScheduler(
            config.evaluation_frequency,
            self.evaluate,
        )
        self.status_scheduler = DynamicScheduler(
            config.status_frequency,
            self.status,
        )
        
        # initialize the parallel training and evaluation environments
        self.train_env = make_vector_env(
            config.train_env,
            config.parallel_envs,
            kwargs=train_env_kwargs,
            seed=config.seed,
            async_envs=config.async_envs,
        )
        
        self.eval_env = make_vector_env(
            config.eval_env,
            config.parallel_envs,
            kwargs=eval_env_kwargs,
            seed=config.seed+1,
            async_envs=config.async_envs,
        )
        
        # build the model
        if config.load_checkpoint is not None:
            checkpoint = torch.load(config.load_checkpoint)
            model_checkpoint = checkpoint['model']
        else:
            model_checkpoint = None
        self.model = ModelClass(
            config,
            self.train_env.single_observation_space,
            self.train_env.single_action_space,
            checkpoint=model_checkpoint,
        ).to(self.device)
        
        # tmp on policy only
        self.rollout_model = self.model
        
        # build the optimizer
        self.optimizer = build_optimizer(config, self.model)
        self.lr_scheduler = linear_warmup_cosine_decay_scheduler(
            config, self.optimizer)
        
        # reset the training environments and generate the first model inputs
        observation, info = self.train_env.reset()
        
        # initialize the rollout variables
        self.rollout_model_output = None
        self.rollout_observation = observation
        self.rollout_terminal = numpy.array([True] * config.parallel_envs)
        self.rollout_truncated = numpy.array([True] * config.parallel_envs)
        self.rollout_returns = numpy.zeros(config.parallel_envs)
        self.rollout_info = info
        
        # build the dataset
        self.initialize_dataset()
        
        # build the logs
        self.logs = {}
        self.logs['total_loss'] = Log()
        self.logs['train_return'] = Log()
        if self.config.eval_mode in ('sample', 'sample_and_max'):
            self.logs['eval_return_sample'] = Log(capacity='adaptive')
        if self.config.eval_mode in ('max', 'sample_and_max'):
            self.logs['eval_return_max'] = Log(capacity='adaptive')
        for i in range(len(self.optimizer.param_groups)):
            self.logs['lr_%i'%i] = Log()
        
        self.progress = tqdm.tqdm(total=self.config.steps)
        self.progress.set_description('Total Progress')
    
    def initialize_dataset(self):
        model_kwargs = self.rollout_model.observation_to_kwargs(
            self.rollout_observation,
            self.rollout_info,
            self.rollout_terminal | self.rollout_truncated,
            self.rollout_model_output,
        )
        
        # compute an example experience batch
        expert = self.rollout_model.expert(
            self.rollout_observation, self.rollout_info)
        example_experience = (
            model_kwargs,
            numpy_to_torch(expert),
        )
        example_experience = hierarchy_to_device(example_experience, 'cpu')
        
        self.dataset = ParallelRolloutDataset(
            self.config.parallel_envs,
            self.config.steps_per_epoch//self.config.parallel_envs,
            example_experience,
            batch_size=self.config.batch_size,
            recurrence=self.config.recurrence,
            shuffle=True,
            auto_shift=True,
        )
    
    def train(self):
        with self.checkpointer.save_on_exit_context():
            with self.progress:
                while self.steps < self.config.steps:
                    self.progress.write('='*80)
                    new_steps = self.collect_data()
                    for epoch in range(1, self.config.epochs+1):
                        self.progress.write('-'*80)
                        self.progress.write('Epoch: %i'%epoch)
                        self.train_epoch()
                    
                    self.steps += new_steps
                    self.evaluator.step(self.steps)
                    self.status_scheduler.step(self.steps)
                    self.checkpointer.step(self.steps)
                    self.progress.update(new_steps)
                    if self.progress.n > self.progress.total:
                        self.progress.total = self.progress.n
    
    def train_epoch(self):
        self.model.train()
        epoch_progress = tqdm.tqdm(iter(self.dataset))
        epoch_progress.set_description('Training')
        for seq_model_kwargs, seq_expert in epoch_progress:
            seq_model_kwargs = hierarchy_to_device(
                seq_model_kwargs, self.device)
            seq_expert = hierarchy_to_device(seq_expert, self.device)
            for i in range(self.config.recurrence):
                
                # get the step expert
                expert, num_expert_actions = hierarchy_getitem(seq_expert, i)
                loss_mask = num_expert_actions != 0
                b = num_expert_actions.shape[0]
                
                # pick a random expert action
                expert_index = torch.cat(
                    [torch.randint((n or 1), (1,)) for n in num_expert_actions],
                    dim=0,
                )
                expert_sample = hierarchy_getitem(
                    expert, (range(b), expert_index))
                
                # forward pass
                model_kwargs = hierarchy_getitem(seq_model_kwargs, i)
                model_output = self.model(**model_kwargs, sample=expert_sample)
                
                # supervision
                log_prob = -self.model.log_prob(model_output) * loss_mask
                loss = log_prob.mean()
                self.logs['total_loss'].log(loss)
                
                # backward
                loss.backward()
                if self.config.grad_norm_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_norm_clipping)
                self.batches += 1
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step(self.batches)
                self.optimizer.step()
                self.optimizer.zero_grad()
                for i, group in enumerate(self.optimizer.param_groups):
                    self.logs['lr_%i'%i].log(group['lr'])
    
    def evaluate(self, index, steps):
        self.progress.write('='*80)
        self.model.eval()
        if self.config.visualize_evaluation:
            visualization_subdirectory = os.path.join(
                self.config.visualization_directory, 'eval_%04i'%index)
            if not os.path.exists(visualization_subdirectory):
                os.makedirs(visualization_subdirectory)
        else:
            visualization_subdirectory = None
        
        if self.config.eval_mode in ('sample', 'sample_and_max'):
            returns = evaluate_policy(
                self.model,
                self.eval_env,
                self.config.eval_episodes,
                visualize=self.config.visualize_evaluation,
                visualization_directory=visualization_subdirectory,
                sample_max=False,
            )
            
            average_return = numpy.mean(returns)
            self.logs['eval_return_sample'].log(average_return)
            self.progress.write(
                'Evaluation Return Sample: %.04f'%average_return)
        
        if self.config.eval_mode in ('max', 'sample_and_max'):
            returns = evaluate_policy(
                self.model,
                self.eval_env,
                self.config.eval_episodes,
                visualize=self.config.visualize_evaluation,
                visualization_directory=visualization_subdirectory,
                sample_max=True,
            )
            
            average_return = numpy.mean(returns)
            self.logs['eval_return_max'].log(average_return)
            self.progress.write(
                'Evaluation Return Max: %.04f'%average_return)
    
    def status(self, index, steps):
        charts = []
        lr_chart = plot_logs(
            {
                'lr_%i'%i:self.logs['lr_%i'%i]
                for i in range(len(self.optimizer.param_groups))
            },
            colors={
                'lr_%i'%i:['RED','BLUE','GREEN','YELLOW'][i%4]
                for i in range(len(self.optimizer.param_groups))
            },
            legend=True,
            border='line',
            height=20,
            min_max_y=True,
        )
        charts.append(lr_chart)
        
        total_loss_chart = plot_logs(
            {'total_loss':self.logs['total_loss']},
            colors={'total_loss' : 'RED'},
            legend=True,
            border='line',
            height=20,
            min_max_y=True,
        )
        charts.append(total_loss_chart)
        
        train_return_chart = plot_logs(
            {'train_return':self.logs['train_return']},
            colors={'train_return' : 'YELLOW'},
            legend=True,
            border='line',
            height=20,
            min_max_y=True,
        )
        charts.append(train_return_chart)
        
        eval_return_sample_chart = plot_logs(
            #{'eval_return_sample':self.logs['eval_return_sample']},
            #colors={'eval_return_sample' : 'GREEN'},
            {k:v for k,v in self.logs.items() if 'eval_return' in k},
            colors={'eval_return_sample' : 'GREEN', 'eval_return_max' : 'BLUE'},
            legend=True,
            border='line',
            height=20,
            min_max_y=True,
        )
        charts.append(eval_return_chart)
        
        if self.eval_mode in ('sample', 'sample_and_max'):
            recent_return = self.logs['eval_return_sample'].y[-1]
            charts.append(
                'Most Recent Evaluation Return (Sample Mode): %.04f'%
                recent_return
            )
        if self.eval_mode in ('max', 'sample_and_max'):
            recent_return = self.logs['eval_return_max'].y[-1]
            charts.append(
                'Most Recent Evaluation Return (Max Mode): %.04f'%
                recent_return
            )
        
        self.progress.write('\n'.join(charts))
    
    def collect_data(self):
        self.model.train()
        data_progress = tqdm.tqdm(
            range(self.config.steps_per_epoch//self.config.parallel_envs))
        data_progress.set_description('Collecting Data')
        
        if self.config.visualize_train:
            visualization_directory = os.path.join(
                self.config.visualization_directory, 'train_%08i'%self.steps)
            if not os.path.exists(visualization_directory):
                os.makedirs(visualization_directory)
            running_images = [[] for _ in range(self.train_env.num_envs)]
            trajectory_index = 0
        else:
            visualization_directory = None
        
        for i in data_progress:
            # get expert advice
            expert = self.rollout_model.expert(
                self.rollout_observation, self.rollout_info)
            
            # forward
            model_kwargs = self.rollout_model.observation_to_kwargs(
                self.rollout_observation,
                self.rollout_info,
                self.rollout_terminal | self.rollout_truncated,
                self.rollout_model_output,
            )
            
            # sample an action
            with torch.no_grad():
                model_output = self.rollout_model(**model_kwargs)
                if i / len(data_progress) < self.config.off_policy_sprinkle:
                    on_policy_prob = 0.
                else:
                    try:
                        start, end = self.config.on_policy
                        if self.steps < start:
                            on_policy_prob = 0.
                        elif self.steps >= end:
                            on_policy_prob = 1.
                        else:
                            on_policy_prob = (self.steps-start) / (end-start)
                    except TypeError:
                        on_policy_prob = self.config.on_policy
                on_policy = random.random() < on_policy_prob
                if on_policy:
                    action = self.rollout_model.compute_action(model_output)
                else:
                    expert_actions, num_expert_actions = expert
                    expert_index = torch.cat([
                        torch.randint((n or 1), (1,))
                        for n in num_expert_actions],
                        dim=0,
                    )
                    b = num_expert_actions.shape[0]
                    action = hierarchy_getitem(
                        expert_actions, (range(b), expert_index))
            
            # step
            observation, reward, terminal, truncated, info = (
                self.train_env.step(action))
            done = terminal | truncated
            
            # store experience
            data_kwargs = hierarchy_to_device(model_kwargs, 'cpu')
            self.dataset[i] = (data_kwargs, numpy_to_torch(expert))
            
            if self.config.visualize_train:
                images = self.rollout_model.visualize(
                    self.rollout_observation,
                    action,
                    reward,
                    model_output,
                    next_image = observation['image'],
                )
                for image, image_list in zip(images, running_images):
                    image_list.append(image)
                
                for d, image_list in zip(done, running_images):
                    if d:
                        trajectory_subdirectory = os.path.join(
                            visualization_directory, '%04i'%trajectory_index)
                        if not os.path.exists(trajectory_subdirectory):
                            os.makedirs(trajectory_subdirectory)
                        for i, image in enumerate(image_list):
                            image_path = '%s/vis.%04i.%04i.png'%(
                                trajectory_subdirectory,
                                trajectory_index,
                                i,
                            )
                            save_image(image, image_path)
                        image_list.clear()
                        trajectory_index += 1
            
            # update rollout variables
            self.rollout_model_output = model_output
            self.rollout_observation = observation
            self.rollout_terminal = terminal
            self.rollout_truncated = truncated
            self.rollout_info = info
            
            self.rollout_returns += reward
            done_returns = self.rollout_returns[done]
            for r in done_returns:
                self.logs['train_return'].log(r)
            self.rollout_returns *= ~done
        
        new_steps = self.config.steps_per_epoch
        self.progress.write('Collected %i Total Transitions'%new_steps)
        
        return new_steps
    
    def state_dict(self):
        state_dict = {}
        state_dict['steps'] = self.steps
        state_dict['logs'] = {
            name : log.get_state() for name, log in self.logs.items()
        }
        state_dict['checkpointer'] = self.checkpointer.state_dict()
        state_dict['evaluator'] = self.evaluator.state_dict()
        state_dict['status_scheduler'] = self.status_scheduler.state_dict()
        state_dict['model'] = self.model.state_dict()
        
        return state_dict
    
    def load_state_dict(self, state_dict):
        self.steps = state_dict['steps']
        for name, log in self.logs.items():
            log.set_state(state_dict['logs'][name])
        self.checkpointer.load_state_dict(state_dict['checkpointer'])
        self.evaluator.load_state_dict(state_dict['evaluator'])
        self.status_scheduler.load_state_dict(state_dict['status_scheduler'])
        self.model.load_state_dict(state_dict['model'])
