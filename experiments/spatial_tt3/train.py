#!/usr/bin/env python
#import brick_gym.torch.train.graph_d as graph_d
import ltron_torch.train.spatial_step as train_spatial_step

#run = 'Feb24_11-25-43_mechagodzilla'
#epoch = 380
#run = 'Feb24_19-26-26_mechagodzilla' #None #'Feb23_00-57-59_mechagodzilla'
#epoch = 35
run = None #'May16_00-44-51_mechagodzilla'
epoch = 50

if run is not None:
    step_checkpoint = './checkpoint/%s/step_model_%04i.pt'%(run, epoch)
    edge_checkpoint = './checkpoint/%s/edge_model_%04i.pt'%(run, epoch)
    optimizer_checkpoint = './checkpoint/%s/optimizer_%04i.pt'%(run, epoch)
else:
    step_checkpoint = None
    edge_checkpoint = None
    optimizer_checkpoint = None

if __name__ == '__main__':
    train_spatial_step.train_label_confidence(
        # load checkpoints
        step_checkpoint = step_checkpoint,
        edge_checkpoint = edge_checkpoint,
        optimizer_checkpoint = optimizer_checkpoint,
        
        # general settings
        num_epochs = 1000,
        mini_epochs_per_epoch = 1,
        
        # dataset settings
        dataset='tiny_turbos3',
        augment_dataset='rando_tt3',
        train_split='train',
        train_subset=None,
        test_split='test',
        test_subset=128,
        num_processes = 16,
        randomize_viewpoint=True,
        random_floating_bricks=False,
        random_floating_pairs=False,
        random_bricks_rotation_mode='local_identity',
        controlled_viewpoint=False,
        
        # rollout settings
        train_steps_per_epoch = 1024,
        
        # train settings
        learning_rate = 1e-4,
        weight_decay = 1e-6,
        mini_epoch_sequences = 2048, #2048//4,
        mini_epoch_sequence_length = 1, #4,
        batch_size = 32, #8,
        viewpoint_loss_weight = 0.5,
        entropy_loss_weight = 0.1,
        max_instances_per_step=2,
        
        # model settings
        #step_model_backbone = 'smp_fpn_rnxt50',
        #step_model_name = 'nth_try', #'center_voting',
        model_backbone = 'simple_fcn',
        
        # test settings
        test_frequency = 0,
        test_steps_per_epoch = 8, #512,
        
        # logging settings
        log_train=8,
        
        # checkpoint settings
        checkpoint_frequency=25,
    )
