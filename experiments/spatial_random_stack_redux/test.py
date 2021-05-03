#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import ltron_torch.train.spatial_graph as spatial_graph

#run = 'Jan24_01-28-20_mechagodzilla'
#epoch = 200
#run = 'Feb10_11-50-54_gpu3'
#run = 'Feb23_00-57-59_mechagodzilla' #'Feb17_22-57-19_gpu3'
#run = 'Mar05_21-48-25_mechagodzilla'
run = 'May01_13-39-04_mechagodzilla'
epoch = 500

if __name__ == '__main__':
    spatial_graph.test_checkpoint(
        # load checkpoints
        step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch),
        
        # dataset settings
        dataset = 'random_stack_redux',
        num_processes = 8,
        split = 'test',
        subset = 128,
        
        # model settings
        step_model_name='nth_try',
        step_model_backbone='simple',
        decoder_channels=512,
        
        # output settings
        output_path='./test_output',
    )
