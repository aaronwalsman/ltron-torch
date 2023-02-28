import numpy

import torch
import torch.nn.functional as F

from splendor.masks import color_index_to_byte

from avarice.trainers import PPOTrainerConfig, PPOTrainer, env_fn_wrapper

from ltron.gym.envs.break_env import BreakEnvConfig
from ltron.gym.envs.make_env import MakeEnvConfig
from ltron.gym.components import ViewpointActions

from ltron_torch.models.ltron_visual_transformer import (
    LtronVisualTransformerConfig,
    LtronVisualTransformer,
    cursor_islands,
)
from ltron_torch.models.equivalence import equivalent_outcome_categorical

from ltron.constants import SHAPE_CLASS_LABELS
from ltron.visualization.drawing import (
    write_text,
    draw_crosshairs,
    draw_square,
    draw_line,
    heatmap_overlay,
    stack_images_horizontal,
)

class LtronPPOTrainerConfig(
    BreakEnvConfig,
    MakeEnvConfig,
    PPOTrainerConfig,
    LtronVisualTransformerConfig
):
    # override defaults
    channels = 256
    heads = 4
    
    batch_size = 32
    parallel_train_envs = 16
    parallel_eval_envs = 16
    
    recurrent = 1
    
    train_env = 'LTRON/Break-v0'
    
    eval_env = 'LTRON/Break-v0'

class LtronPPOTrainer(PPOTrainer):
    def make_single_train_env_fn(self, parallel_index):
        env_fn = env_fn_wrapper(
            self.config.train_env,
            config=self.config,
            train=True,
        )
        return env_fn
    
    def make_single_eval_env_fn(self, parallel_index):
        env_fn = env_fn_wrapper(
            self.config.eval_env,
            config=self.config,
            train=False,
        )
        return env_fn
    
    def compute_losses_OFF(self, model_output, target):
        target_sample = target['sample']
        synthetic_sample = {}
        synthetic_sample['action_primitives'] = (
            target_sample['action_primitives'])
        synthetic_sample['cursor'] = {}
        synthetic_sample['cursor']['button'] = torch.torch.ones_like(
            target_sample['cursor']['button'])
        
        #105,127 -> 77,126
        click_location = torch.zeros_like(target_sample['cursor']['click'])
        click_location[:,0] = 105
        click_location[:,1] = 127
        release_location = torch.zeros_like(target_sample['cursor']['release'])
        release_location[:,0] = 80
        release_location[:,1] = 126
        
        #breakpoint()
        
        synthetic_sample['cursor']['click'] = click_location
        synthetic_sample['cursor']['release'] = release_location
        
        sample_output = self.model.sample_log_prob(
            model_output, sample=synthetic_sample)
        
        button_loss = F.cross_entropy(
            sample_output['logits']['cursor']['button'],
            synthetic_sample['cursor']['button'],
            reduction='none',
        )
        click_logits = sample_output['logits']['cursor']['click']
        b,h,w = click_logits.shape
        click_logits = click_logits.view(b,-1)
        click_target = synthetic_sample['cursor']['click']
        click_target = click_target[:,0] * w + click_target[:,1]
        click_loss = F.cross_entropy(
            click_logits, click_target, reduction='none')
        release_logits = sample_output['logits']['cursor']['release']
        release_logits = release_logits.view(b,-1)
        release_target = synthetic_sample['cursor']['release']
        release_target = release_target[:,0] * w + release_target[:,1]
        release_loss = F.cross_entropy(
            release_logits, release_target, reduction='none')
        ce_policy_loss = (button_loss + click_loss + release_loss).mean()
        lp_policy_loss = -sample_output['log_prob'].mean()
        
        print('button loss', button_loss)
        print('click loss', click_loss)
        print('release loss', release_loss)
        
        print('ce', ce_policy_loss)
        print('lp', lp_policy_loss)
        #breakpoint()
        
        #value = self.model.value(model_output)
        device = sample_output['log_prob'].device
        losses = {}
        #losses['policy_loss'] = 
        losses['policy_loss'] = lp_policy_loss
        losses['value_loss'] = torch.zeros(1, device=device)
        losses['entropy_loss'] = torch.zeros(1, device=device)
        
        return losses
    
    def visualize(self,
        observation,
        action,
        reward,
        model_output,
        sample_output,
    ):
        images = observation['image']
        
        mode = action['action_primitives']['mode']
        mode_space = (
            self.eval_env.single_action_space['action_primitives']['mode'])
        
        button_sample = sample_output['sample']['cursor']['button'].view(-1,1,1)
        device = button_sample.device
        pos_islands = torch.LongTensor(
            cursor_islands(observation['pos_snap_render'])).to(device)
        neg_islands = torch.LongTensor(
            cursor_islands(observation['neg_snap_render'])).to(device)
        click_islands = (
            pos_islands * button_sample + neg_islands * (1-button_sample))
        release_islands = (
            pos_islands * (1-button_sample) + neg_islands * button_sample)
        
        if 'cursor' in sample_output['logits']:
            click_logits = sample_output['logits']['cursor']['click']
            release_logits = sample_output['logits']['cursor']['release']
            b,h,w = click_logits.shape
            click_heatmaps = torch.softmax(
                click_logits.view(b,h*w), dim=-1).view(b,h,w,1)
            release_heatmaps = torch.softmax(
                release_logits.view(b,h*w), dim=-1).view(b,h,w,1)
            click_eq_dist = equivalent_outcome_categorical(
                click_logits,
                click_islands,
            )
            release_eq_dist = equivalent_outcome_categorical(
                release_logits,
                release_islands,
            )
        else:
            b = action['action_primitives']['mode'].shape[0]
            h, w = 256,256
            click_heatmaps = torch.zeros((b,h,w,1))
            release_heatmaps = torch.zeros((b,h,w,1))
            click_eq_dist = TODO
            release_eq_dist = TODO
        
        visualizations = []
        
        for i, m in enumerate(mode):
            image = images[i]
            
            click_heatmap_background = images[i].copy()
            click_heatmap = click_heatmaps[i].cpu().numpy()
            click_heatmap = heatmap_overlay(
                click_heatmap_background,
                click_heatmap,
                [255,0,0],
                background_scale=0.25,
                max_normalize=True,
            )
            
            release_heatmap_background = images[i].copy()
            release_heatmap = release_heatmaps[i].cpu().numpy()
            release_heatmap = heatmap_overlay(
                release_heatmap_background,
                release_heatmap,
                [0,0,255],
                background_scale=0.25,
                max_normalize=True,
            )
            
            click_p_map = click_eq_dist.probs[i][click_islands[i]].detach()
            click_min = torch.min(click_p_map)
            click_max = torch.max(click_p_map)
            click_p_map = (click_p_map - click_min) / (click_max - click_min)
            click_p_map = (click_p_map.cpu().numpy() * 255).astype(numpy.uint8)
            h, w = click_p_map.shape
            click_p_map = click_p_map.reshape(h,w,1).repeat(3, axis=2)
            
            release_p_map = release_eq_dist.probs[i][
                release_islands[i]].detach()
            release_min = torch.min(release_p_map)
            release_max = torch.max(release_p_map)
            release_p_map = (
                (release_p_map - release_min) / 
                (release_max - release_min)
            )
            release_p_map = (release_p_map.cpu().numpy() * 255).astype(
                numpy.uint8)
            h, w = release_p_map.shape
            release_p_map = release_p_map.reshape(h,w,1).repeat(3, axis=2)
            
            #pos_eq = color_index_to_byte(pos_eqs[i])
            #neg_eq = color_index_to_byte(neg_eqs[i])
            #mode_name = switch_map[m]
            mode_name = mode_space.names[m]
            
            '''
            if action['interface']['cursor']['button'][i]:
                click_eq = pos_eq
                release_eq = neg_eq
            else:
                click_eq = neg_eq
                release_eq = pos_eq
            '''
            
            if mode_name == 'viewpoint':
                v = action['action_primitives']['viewpoint'][i]
                action_str = '%s.%s'%(mode_name, ViewpointActions(value=v).name)
            elif mode_name == 'remove':
                r = action['action_primitives']['remove'][i]
                ci, cs = observation['cursor']['click_snap'][i]
                cy, cx = action['cursor']['click'][i]
                action_str = (
                    '%s.%i\n'
                    'Click x/y: %i,%i\n'
                    'Click instance/snap: %i,%i\n'%(
                        mode_name, r, cx, cy, ci, cs)
                )
            elif mode_name == 'insert':
                SHAPE_NAMES = {
                    value:key for key, value in SHAPE_CLASS_LABELS.items()}
                s, c = action['action_primitives']['insert'][i]
                shape_name = SHAPE_NAMES.get(s, 'NONE')
                action_str = (
                    '%s.%i/%i\n'
                    'Shape: %s\n'
                    'Color: %i\n'%(
                        mode_name, s, c, shape_name, c))
            else:
                action_str = mode_name
            
            action_str = 'Reward: %.04f\n%s'%(reward[i], action_str)
            image = write_text(image, action_str)
            
            if mode_name in ('remove', 'pick_and_place', 'rotate'):
                click_polarity = action['cursor']['button'][i]
                click_yx = action['cursor']['click'][i]
                if click_polarity:
                    draw_crosshairs(image, *click_yx, 3, (255,0,0))
                    draw_crosshairs(click_heatmap, *click_yx, 3, (255,0,0))
                    draw_crosshairs(click_p_map, *click_yx, 3, (255,0,0))
                else:
                    draw_square(image, *click_yx, 3, (255,0,0))
                    draw_square(click_heatmap, *click_yx, 3, (255,0,0))
                    draw_square(click_p_map, *click_yx, 3, (255,0,0))
            
            if mode_name in ('pick_and_place',):
                release_polarity = ~bool(click_polarity)
                release_yx = action['cursor']['release'][i]
                if release_polarity:
                    draw_crosshairs(image, *release_yx, 3, (0,0,255))
                    draw_crosshairs(release_heatmap, *release_yx, 3, (0,0,255))
                    draw_crosshairs(release_p_map, *release_yx, 3, (0,0,255))
                else:
                    draw_square(image, *release_yx, 3, (0,0,255))
                    draw_square(release_heatmap, *release_yx, 3, (0,0,255))
                    draw_square(release_p_map, *release_yx, 3, (0,0,255))
                draw_line(image, *click_yx, *release_yx, (255,0,255))
            
            visualization = stack_images_horizontal([
                image,
                click_heatmap,
                release_heatmap,
                click_p_map,
                release_p_map,
            ])#, click_eq, release_eq])
            visualizations.append(visualization)
        
        return visualizations
