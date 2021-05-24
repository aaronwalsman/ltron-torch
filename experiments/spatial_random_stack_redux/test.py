#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import ltron_torch.train.spatial_step as spatial_step

#run = 'Jan24_01-28-20_mechagodzilla'
#epoch = 200
#run = 'Feb10_11-50-54_gpu3'
#run = 'Feb23_00-57-59_mechagodzilla' #'Feb17_22-57-19_gpu3'
#run = 'Mar05_21-48-25_mechagodzilla'
#run = 'May17_15-40-58_mechagodzilla'
run = 'May22_15-41-01_mechagodzilla'
epoch = 825

if __name__ == '__main__':
    spatial_step.test_checkpoint(
        # load checkpoints
        step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch),
        
        # dataset settings
        dataset = 'random_stack_redux',
        num_processes = 8,
        split = 'test',
        subset = 16,
        
        # model settings
        model_backbone='simple_fcn',
        decoder_channels=512,
        
        # output settings
        output_path='./test_output',
    )
