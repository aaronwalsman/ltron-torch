import os
import copy

import numpy

import torch
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits

import tqdm

from splendor.image import save_image

from ltron.config import Config
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.gym.envs.break_and_make_env import BreakAndMakeEnv
from ltron.visualization.drawing import write_text

from ltron_torch.models.padding import make_padding_mask
from ltron_torch.interface.utils import (
    bernoulli_or_max, categorical_or_max, categorical_or_max_2d)

class BreakAndMakeInterfaceConfig(Config):
    mode_loss_weight = 1.
    shape_loss_weight = 1.
    color_loss_weight = 1.
    table_spatial_loss_weight = 1.
    table_polarity_loss_weight = 1.
    hand_spatial_loss_weight = 1.
    hand_polarity_loss_weight = 1.
    
    visualization_seqs = 10

class BreakAndMakeInterface:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        dummy_env = BreakAndMakeEnv(config)
        self.no_op_action = dummy_env.no_op_action()
    
    def observation_to_tensors(self, observation, pad):
        raise NotImplementedError
    
    def action_to_tensors(self, action, pad):
        
        # initialize y
        y = {}
        
        # get the device
        device = next(self.model.parameters()).device
        
        # build the mode tensor
        y_mode = numpy.zeros(action['phase'].shape, dtype=numpy.long)
        y_mode[action['phase'] == 1] = 0
        y_mode[action['phase'] == 2] = 1
        y_mode[action['disassembly'] == 1] = 2
        y_mode[action['pick_and_place'] == 1] = 3
        y_mode[action['pick_and_place'] == 2] = 4
        y_mode[action['rotate'] == 1] = 5
        y_mode[action['rotate'] == 2] = 6
        y_mode[action['rotate'] == 3] = 7
        y_mode[action['insert_brick']['shape'] != 0] = 8
        y_mode[action['table_viewpoint'] == 1] = 9
        y_mode[action['table_viewpoint'] == 2] = 10
        y_mode[action['table_viewpoint'] == 3] = 11
        y_mode[action['table_viewpoint'] == 4] = 12
        y_mode[action['table_viewpoint'] == 5] = 13
        y_mode[action['table_viewpoint'] == 6] = 14
        y_mode[action['table_viewpoint'] == 7] = 15
        y_mode[action['hand_viewpoint'] == 1] == 16
        y_mode[action['hand_viewpoint'] == 2] == 17
        y_mode[action['hand_viewpoint'] == 3] == 18
        y_mode[action['hand_viewpoint'] == 4] == 19
        y_mode[action['hand_viewpoint'] == 5] == 20
        y_mode[action['hand_viewpoint'] == 6] == 21
        y_mode[action['hand_viewpoint'] == 7] == 22
        y['mode'] = torch.LongTensor(y_mode).to(device)
        
        for region in 'table', 'hand':
            # get the active sequence indices
            y['%s_i'%region] = torch.BoolTensor(
                action['%s_cursor'%region]['activate']).to(device)
            
            # get the click locations
            y['%s_yx'%region] = torch.LongTensor(
                action['%s_cursor'%region]['position']).to(device)
            
            # get the click polarity
            y['%s_p'%region] = torch.LongTensor(
                action[region + '_cursor']['polarity']).to(device)
        
        y['insert_i'] = y['mode'] == 8
        y['shape'] = torch.LongTensor(
            action['insert_brick']['shape']).to(device) - 1
        y['color'] = torch.LongTensor(
            action['insert_brick']['color']).to(device) - 1
        
        return y
    
    def loss(self, x, y, pad, log=None, clock=None):
        
        # initialize loss
        loss = 0
        
        # get shape and device
        s, b, m = x['mode'].shape
        device = x['mode'].device
        
        # mode supervision
        pad = torch.LongTensor(pad).to(device)
        loss_mask = make_padding_mask(pad, (s,b), mask_value=True)
        mode_loss = cross_entropy(
            x['mode'].view(-1,m), y['mode'].view(-1), reduction='none')
        mode_loss = mode_loss.view(s,b) * loss_mask
        mode_loss = mode_loss.mean() * self.config.mode_loss_weight
        loss = loss + mode_loss
        
        if log is not None:
            log.add_scalar('train/mode_loss', mode_loss, clock[0])
        
        # table and hand supervision
        for region in 'table', 'hand':
            h, w = x[region].shape[-2:]
            i = y['%s_i'%region].view(-1)
            x_region = x[region].reshape(s*b, 2, h*w)[i]
            
            # spatial
            x_spatial = x_region[:,0]
            y_y = y['%s_yx'%region][:,:,0]
            y_x = y['%s_yx'%region][:,:,1]
            y_spatial = (y_y * w + y_x).view(-1)[i]
            spatial_loss = cross_entropy(x_spatial, y_spatial)
            spatial_loss = spatial_loss * getattr(
                self.config, '%s_spatial_loss_weight'%region)
            loss = loss + spatial_loss
            
            # polarity
            x_p = x_region[:,1].view(-1, h*w)
            x_p = x_p[range(x_p.shape[0]), y_spatial]
            y_p = y['%s_p'%region].view(-1)[i].float()
            polarity_loss = binary_cross_entropy_with_logits(x_p, y_p)
            polarity_loss = polarity_loss * getattr(
                self.config, '%s_polarity_loss_weight'%region)
            loss = loss + polarity_loss
            
            if log is not None:
                log.add_scalar(
                    'train/%s_spatial_loss'%region, spatial_loss, clock[0])
                log.add_scalar(
                    'train/%s_polarity_loss'%region, polarity_loss, clock[0])
        
        # shape and color loss
        i = y['insert_i'].view(-1)
        for region in 'shape', 'color':
            n = x[region].shape[-1]
            x_region = x[region].view(s*b, n)[i]
            y_region = y[region].view(-1)[i]
            region_loss = cross_entropy(x_region, y_region)
            region_loss = region_loss * getattr(
                self.config, '%s_loss_weight'%region)
            loss = loss + region_loss
            
            if log is not None:
                log.add_scalar('train/%s_loss'%region, region_loss, clock[0])
        
        if log is not None:
            log.add_scalar('train/total_loss', loss, clock[0])
        
        return loss
    
    def tensor_to_actions(self, x, env, mode='sample'):
        s, b, num_modes = x['mode'].shape
        assert s == 1
        x_mode = x['mode'].view(b,-1)
        x_shape = x['shape'].view(b,-1)
        x_color = x['color'].view(b,-1)
        
        mode_action = categorical_or_max(x_mode, mode=mode).cpu().numpy()
        shape_action = categorical_or_max(x_shape, mode=mode).cpu().numpy()
        color_action = categorical_or_max(x_color, mode=mode).cpu().numpy()
        
        #import pdb
        #pdb.set_trace()
        
        region_yx = []
        region_polarity = []
        for region in 'table', 'hand':
            h, w = x[region].shape[-2:]
            x_region = x[region].view(b, 2, h, w)
            
            # spatial
            x_spatial = x_region[:,0]
            region_y, region_x = categorical_or_max_2d(x_spatial, mode=mode)
            region_y = region_y.cpu().numpy()
            region_x = region_x.cpu().numpy()
            region_yx.append((region_y, region_x))
            
            # polarity
            x_polarity = x_region[:,1]
            x_polarity = x_polarity[range(b), region_y, region_x]
            polarity = bernoulli_or_max(x_polarity, mode=mode).cpu().numpy()
            region_polarity.append(polarity)
        
        (table_y, table_x), (hand_y, hand_x) = region_yx
        table_polarity, hand_polarity = region_polarity
        
        actions = []
        for i in range(b):
            action = copy.deepcopy(self.no_op_action)
            mode = mode_action[i]
            if mode == 0:
                action['phase'] = 1
            elif mode == 1:
                action['phase'] = 2
            elif mode == 2: # disassembly
                action['disassembly'] = 1
                action['table_cursor']['activate'] = True
            elif mode == 3:
                action['pick_and_place'] = 1
                action['table_cursor']['activate'] = True
                action['hand_cursor']['activate'] = True
            elif mode == 4:
                action['pick_and_place'] = 2
                action['hand_cursor']['activate'] = True
            elif mode == 5:
                action['rotate'] = 1
                action['table_cursor']['activate'] = True
            elif mode == 6:
                action['rotate'] = 2
                action['table_cursor']['activate'] = True
            elif mode == 7:
                action['rotate'] = 3
                action['table_cursor']['activate'] = True
            elif mode == 8:
                action['insert_brick']['shape'] = shape_action[i] + 1
                action['insert_brick']['color'] = color_action[i] + 1
            elif mode >= 9 and mode < 16:
                action['table_viewpoint'] = mode - 8
            elif mode >= 16:
                action['hand_viewpoint'] = mode - 15
            
            if action['table_cursor']['activate']:
                action['table_cursor']['position'] = numpy.array(
                    (table_y[i], table_x[i]), dtype=numpy.long)
                action['table_cursor']['polarity'] = table_polarity[i]
            if action['hand_cursor']['activate']:
                action['hand_cursor']['position'] = numpy.array(
                    (hand_y[i], hand_x[i]), dtype=numpy.long)
                action['hand_cursor']['polarity'] = hand_polarity[i]
            
            actions.append(action)
        
        return actions
    
    def visualize_episodes(self, epoch, episodes, visualization_directory):
        num_seqs = min(
            self.config.visualization_seqs, episodes.num_seqs())
        for seq_id in tqdm.tqdm(range(num_seqs)):
            seq_path = os.path.join(
                visualization_directory, 'seq_%06i'%seq_id)
            if not os.path.exists(seq_path):
                os.makedirs(seq_path)
            
            seq = episodes.get_seq(seq_id)
            seq_len = len_hierarchy(seq)
            table_frames = seq['observation']['table_color_render']
            hand_frames = seq['observation']['hand_color_render']
            for frame_id in range(seq_len):
                table_frame = table_frames[frame_id]
                hand_frame = hand_frames[frame_id]
                th, tw = table_frame.shape[:2]
                hh, hw = hand_frame.shape[:2]
                w = tw + hw
                joined_image = numpy.zeros((th, w, 3), dtype=numpy.uint8)
                joined_image[:,:tw] = table_frame
                joined_image[th-hh:,tw:] = hand_frame
                
                frame_action = index_hierarchy(seq['action'], frame_id)
                #frame_mode = int(frame_action['mode'])
                #frame_shape_id = int(frame_action['shape'])
                #frame_color_id = int(frame_action['color'])
                
                ty, tx = frame_action['table_cursor']['position']
                ty = int(ty)
                tx = int(tx)
                hy, hx = frame_action['hand_cursor']['position']
                hy = int(hy)
                hx = int(hx)
                
                joined_image[ty*4:(ty+1)*4, tx*4:(tx+1)*4] = (0,0,0)
                yy = th - hh
                joined_image[
                    yy+hy*4:yy+(hy+1)*4, tw+(hx)*4:tw+(hx+1)*4] = (0,0,0)
                
                if frame_action['insert_brick']['shape']:
                    shape = frame_action['insert_brick']['shape']
                    color = frame_action['insert_brick']['color']
                    mode_string = 'Insert Brick [%i] [%i]'%(shape, color)
                elif frame_action['pick_and_place'] == 1:
                    mode_string = 'Pick And Place [%i,%i] [%i,%i]'%(ty,tx,hy,hx)
                elif frame_action['pick_and_place'] == 2:
                    mode_string = 'Pick And Place ORIGIN [%i,%i]'%(hy,hx)
                elif frame_action['rotate']:
                    mode_string = 'Rotate [%i] [%i,%i]'%(
                        frame_action['rotate'], ty, tx)
                elif frame_action['disassembly']:
                    mode_string = 'Disassembly [%i,%i]'%(ty,tx)
                elif frame_action['table_viewpoint']:
                    mode_string = 'Table Viewpoint [%i]'%(
                        frame_action['table_viewpoint'])
                elif frame_action['hand_viewpoint']:
                    mode_string = 'Hand Viewpoint [%i]'%(
                        frame_action['hand_viewpoint'])
                elif frame_action['phase']:
                    mode_string = 'Phase [%i]'%frame_action['phase']
                else:
                    mode_string = 'UNWRITTEN'
                
                if frame_id:
                    reward = seq['reward'][frame_id-1]
                else:
                    reward = 0.
                mode_string += '\nReward: %.04f'%reward
                
                try:
                    joined_image = write_text(joined_image, mode_string)
                except OSError:
                    pass
                
                frame_path = os.path.join(
                    seq_path,
                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
                )
                save_image(joined_image, frame_path)
