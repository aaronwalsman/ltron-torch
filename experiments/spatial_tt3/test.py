#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import ltron_torch.train.spatial_step as spatial_step

#run = 'Jan24_01-28-20_mechagodzilla'
#epoch = 200
#run = 'Feb10_11-50-54_gpu3'
#run = 'Feb23_00-57-59_mechagodzilla' #'Feb17_22-57-19_gpu3'
#run = 'Mar05_21-48-25_mechagodzilla'
run = 'May19_10-52-18_mechagodzilla'
epoch = 675

if __name__ == '__main__':
    spatial_step.test_checkpoint(
        # load checkpoints
        step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch),
        
        # dataset settings
        dataset = 'tiny_turbos3',
        num_processes = 8,
        split = 'train',
        subset = None,
        
        # model settings
        model_backbone='simple_fcn',
        decoder_channels=512,
        
        # output settings
        output_path='./test_output',
    )
