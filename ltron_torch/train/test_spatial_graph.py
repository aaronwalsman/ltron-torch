import time
import math
import os
import json

import torch
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import binary_cross_entropy

import numpy

import PIL.Image as Image

import tqdm

import torch_geometric.utils as tg_utils

import renderpy.masks as masks
from renderpy.json_numpy import NumpyEncoder

import ltron_torch.evaluation as evaluation
from ltron.dataset.paths import get_dataset_info
from ltron.gym.ltron_env import async_ltron
from ltron_torch.envs.spatial_env import pose_estimation_env
#from ltron.gym.standard_envs import graph_supervision_env

from ltron_torch.gym_tensor import (
        gym_space_to_tensors, gym_space_list_to_tensors, graph_to_gym_space)
from ltron_torch.gym_log import gym_log
import ltron_torch.models.named_models as named_models
from ltron_torch.models.frozen_batchnorm import FrozenBatchNormWrapper
from ltron_torch.train.loss import (
        dense_instance_label_loss, dense_score_loss, cross_product_loss)

#edge_threshold = 0.05

def train_label_confidence(
        # load checkpoints
        step_checkpoint = None,
        edge_checkpoint = None,
        optimizer_checkpoint = None,
        
        # general settings
        num_epochs = 9999,
        mini_epochs_per_epoch = 1,
        mini_epoch_sequences = 2048,
        mini_epoch_sequence_length = 2,
        
        # dasaset settings
        dataset = 'random_stack',
        num_processes = 8,
        train_split = 'train',
        train_subset = None,
        augment_dataset = None,
        image_resolution = (256,256),
        randomize_viewpoint = True,
        randomize_colors = True,
        random_floating_bricks = False,
        random_floating_pairs = False,
        random_bricks_per_scene = (10,20),
        random_bricks_subset = None,
        random_bricks_rotation_mode = None,
        load_scenes = True,
        
        # train settings
        train_steps_per_epoch = 4096,
        batch_size = 6,
        learning_rate = 3e-4,
        weight_decay = 1e-5,
        instance_label_loss_weight = 0.8,
        instance_label_background_weight = 0.05,
        score_loss_weight = 0.2,
        score_background_weight = 0.05,
        score_ratio = 0.1,
        matching_loss_weight = 1.0,
        edge_loss_weight = 1.0,
        #-----------------
        rotation_loss_weight = 1.0,
        translation_loss_weight = 1.0,
        translation_scale_factor = 0.01,
        #-----------------
        center_local_loss_weight = 0.1,
        center_cluster_loss_weight = 0.0,
        center_separation_loss_weight = 1.0,
        center_separation_distance = 5,
        #-----------------
        removability_loss_weight = 0.0,
        #-----------------
        viewpoint_loss_weight = 0.0,
        #-----------------
        max_instances_per_step = 8,
        multi_hide = False,
        
        # model settings
        step_model_name = 'nth_try',
        step_model_backbone = 'smp_fpn_r18',
        decoder_channels = 512,
        edge_model_name = 'subtract',
        segment_id_matching = False,
        brick_vector_mode = 'average',
        
        # test settings
        test_frequency = 1,
        test_steps_per_epoch = 1024,
        
        # checkpoint settings
        checkpoint_frequency = 1,
        
        # logging settings
        log_train = 0,
        log_test = 0):
    
    print('='*80)
    print('Setup')
    
    print('-'*80)
    print('Logging')
    step_clock = [0]
    log = SummaryWriter()
    
    dataset_info = get_dataset_info(dataset)
    num_classes = max(dataset_info['class_ids'].values()) + 1
    max_instances_per_scene = dataset_info['max_instances_per_scene']
    
    if random_floating_bricks:
        max_instances_per_scene += 20
    if random_floating_pairs:
        max_instances_per_scene += 40
    
    print('-'*80)
    print('Building the step model')
    step_model = named_models.named_graph_step_model(
            step_model_name,
            backbone_name = step_model_backbone,
            decoder_channels = decoder_channels,
            num_classes = num_classes,
            input_resolution = image_resolution,
            pose_head = True,
            removability_head = removability_loss_weight > 0.).cuda()
    if step_checkpoint is not None:
        print('Loading step model checkpoint from:')
        print(step_checkpoint)
        step_model.load_state_dict(torch.load(step_checkpoint))
    
    print('-'*80)
    print('Building the edge model')
    edge_model = named_models.named_edge_model(
            edge_model_name,
            input_dim=decoder_channels).cuda()
    if edge_checkpoint is not None:
        print('Loading edge model checkpoint from:')
        print(edge_checkpoint)
        edge_model.load_state_dict(torch.load(edge_checkpoint))
    
    print('-'*80)
    print('Building the optimizer')
    optimizer = torch.optim.Adam(
            list(step_model.parameters()) + list(edge_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay)
    if optimizer_checkpoint is not None:
        print('Loading optimizer checkpoint from:')
        print(optimizer_checkpoint)
        optimizer.load_state_dict(torch.load(optimizer_checkpoint))
    
    print('-'*80)
    print('Building the background class weight')
    instance_label_class_weight = torch.ones(num_classes)
    instance_label_class_weight[0] = instance_label_background_weight
    instance_label_class_weight = instance_label_class_weight.cuda()
    
    print('-'*80)
    print('Building the train environment')
    if step_model_backbone == 'simple':
        segmentation_width, segmentation_height = (
                image_resolution[0] // 4, image_resolution[0] // 4)
    else:
        segmentation_width, segmentation_height = image_resolution
    '''
    train_env = async_ltron(
            num_processes,
            graph_supervision_env,
            dataset=dataset,
            augment_dataset=augment_dataset,
            split=train_split,
            subset=train_subset,
            load_scenes=load_scenes,
            dataset_reset_mode='multi_pass',
            width = image_resolution[0],
            height = image_resolution[1],
            segmentation_width = segmentation_width,
            segmentation_height = segmentation_height,
            multi_hide=multi_hide,
            visibility_mode='instance',
            randomize_viewpoint=randomize_viewpoint,
            controlled_viewpoint=viewpoint_loss_weight > 0.,
            randomize_viewpoint_frequency='reset',
            randomize_colors=randomize_colors,
            random_floating_bricks=random_floating_bricks,
            random_floating_pairs=random_floating_pairs,
            random_bricks_per_scene=random_bricks_per_scene,
            random_bricks_subset=random_bricks_subset,
            random_bricks_rotation_mode=random_bricks_rotation_mode)
    '''
    train_env = async_ltron(
        num_processes,
        pose_estimation_env,
        dataset=dataset,
        split=train_split,
        subset=train_subset,
        width=image_resolution[0],
        height=image_resolution[1],
        segmentation_width=segmentation_width,
        segmentation_height=segmentation_height)
    
    for epoch in range(1, num_epochs+1):
        epoch_start = time.time()
        print('='*80)
        print('Epoch: %i'%epoch)
        
        train_label_confidence_epoch(
                epoch,
                step_clock,
                log,
                math.ceil(train_steps_per_epoch / num_processes),
                #train_steps_per_epoch,
                mini_epochs_per_epoch,
                mini_epoch_sequences,
                mini_epoch_sequence_length,
                batch_size,
                step_model,
                edge_model,
                optimizer,
                train_env,
                #-----------------
                instance_label_loss_weight,
                instance_label_class_weight,
                score_loss_weight,
                score_background_weight,
                score_ratio,
                matching_loss_weight,
                edge_loss_weight,
                #-----------------
                rotation_loss_weight,
                translation_loss_weight,
                translation_scale_factor,
                #-----------------
                center_local_loss_weight,
                center_cluster_loss_weight,
                center_separation_loss_weight,
                center_separation_distance,
                #-----------------
                removability_loss_weight,
                #-----------------
                viewpoint_loss_weight,
                #-----------------
                segmentation_width,
                segmentation_height,
                max_instances_per_step,
                multi_hide,
                max_instances_per_scene,
                segment_id_matching,
                brick_vector_mode,
                dataset_info,
                log_train)
        
        if (checkpoint_frequency is not None and
                epoch % checkpoint_frequency) == 0:
            checkpoint_directory = os.path.join(
                    './checkpoint', os.path.split(log.log_dir)[-1])
            if not os.path.exists(checkpoint_directory):
                os.makedirs(checkpoint_directory)
            
            print('-'*80)
            step_model_path = os.path.join(
                    checkpoint_directory, 'step_model_%04i.pt'%epoch)
            print('Saving step_model to: %s'%step_model_path)
            torch.save(step_model.state_dict(), step_model_path)
            
            edge_model_path = os.path.join(
                    checkpoint_directory, 'edge_model_%04i.pt'%epoch)
            print('Saving edge_model to: %s'%edge_model_path)
            torch.save(edge_model.state_dict(), edge_model_path)
            
            optimizer_path = os.path.join(
                    checkpoint_directory, 'optimizer_%04i.pt'%epoch)
            print('Saving optimizer to: %s'%optimizer_path)
            torch.save(optimizer.state_dict(), optimizer_path)
        
        '''
        if test_frequency is not None and epoch % test_frequency == 0:
            test_label_confidence_epoch(
                    epoch,
                    step_clock,
                    log,
                    math.ceil(test_steps_per_epoch / num_processes),
                    step_model,
                    edge_model,
                    test_env,
                    log_test)
        '''
        
        print('-'*80)
        print('Elapsed: %.04f'%(time.time() - epoch_start))


def train_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        mini_epochs,
        mini_epoch_sequences,
        mini_epoch_sequence_length,
        batch_size,
        step_model,
        edge_model,
        optimizer,
        train_env,
        instance_label_loss_weight,
        instance_label_class_weight,
        score_loss_weight,
        score_background_weight,
        score_ratio,
        matching_loss_weight,
        edge_loss_weight,
        #------------------
        rotation_loss_weight,
        translation_loss_weight,
        translation_scale_factor,
        #------------------
        center_local_loss_weight,
        center_cluster_loss_weight,
        center_separation_loss_weight,
        center_separation_distance,
        #------------------
        removability_loss_weight,
        #------------------
        viewpoint_loss_weight,
        #------------------
        segmentation_width,
        segmentation_height,
        max_instances_per_step,
        multi_hide,
        max_instances_per_scene,
        segment_id_matching,
        brick_vector_mode,
        dataset_info,
        log_train):
    
    print('-'*80)
    print('Train')
    
    #===========================================================================
    # rollout
    
    print('- '*40)
    print('Rolling out episodes to generate data')
    
    seq_terminal = []
    seq_observations = []
    #seq_graph_state = []
    seq_rewards = []
    seq_viewpoint_actions = []
    
    device = torch.device('cuda:0')
    
    step_model.eval()
    edge_model.eval()
    
    step_observations = train_env.reset()
    step_terminal = torch.ones(train_env.num_envs, dtype=torch.bool)
    step_rewards = numpy.zeros(train_env.num_envs)
    graph_states = [None] * train_env.num_envs
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            
            #-------------------------------------------------------------------
            # data storage and conversion
            
            # store observation
            seq_terminal.append(step_terminal)
            seq_observations.append(step_observations)
            
            # convert gym observations to torch tensors
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    train_env.single_observation_space,
                    device)
            
            #-------------------------------------------------------------------
            # step model forward pass
            head_features = step_model(step_tensors['color_render'])
            
            '''
            #-------------------------------------------------------------------
            # build new graph state for all terminal sequences
            # (do this here so we can use the brick_feature_spec from the model)
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    empty_brick_list = BrickList(
                            step_brick_lists[i].brick_feature_spec())
                    graph_states[i] = BrickGraph(
                            empty_brick_list, edge_attr_channels=1).cuda()
            
            # store the graph_state before it gets updated
            # (we want the graph state that was input to this step)
            seq_graph_state.append([g.detach().cpu() for g in graph_states])
            '''
            
            '''
            #-------------------------------------------------------------------
            # edge model forward pass
            (graph_states,
             step_step_logits,
             step_state_logits,
             extra_extra_logits,
             step_extra_logits,
             state_extra_logits) = edge_model(
                    step_brick_lists,
                    graph_states,
                    max_edges=max_edges,    
                    segment_id_matching=segment_id_matching)
            '''
            
            '''
            #-------------------------------------------------------------------
            # act
            hide_logits = [sbl.hide_action.view(-1) for sbl in step_brick_lists]
            actions = []
            for i, logits in enumerate(hide_logits):
                action = {}
                
                do_hide = True
                if 'viewpoint' in head_features:
                    action['viewpoint'] = random.randint(0,6)
                    
                if multi_hide:
                    visibility_sample = numpy.zeros(
                            max_instances_per_scene+1, dtype=numpy.bool)
                    if do_hide:
                        selected_instances = (
                            step_brick_lists[i].segment_id[:,0].cpu().numpy())
                        visibility_sample[selected_instances] = True
                else:
                    if logits.shape[0]:
                        distribution = Categorical(
                                logits=logits, validate_args=True)
                        segment_sample = distribution.sample()
                        visibility_sample = int(
                                step_brick_lists[i].segment_id[segment_sample])
                    else:
                        visibility_sample = 0
                
                action['visibility'] = visibility_sample
                
                graph_data = graph_to_gym_space(
                        graph_states[i].cpu(),
                        train_env.single_action_space['graph_task'],
                        process_instance_logits=True,
                        segment_id_remap=True,
                )
                action['graph_task'] = graph_data
                
                actions.append(action)
            '''
            
            # act --------------------------------------------------------------
            actions = [{} for _ in range(train_env.num_envs)]
            unrolled_confidence_logits = head_features['confidence'].view(
                train_env.num_envs, -1)
            unrolled_confidence = torch.sigmoid(unrolled_confidence_logits)
            unrolled_segmentation = step_tensors['segmentation_render'].view(
                train_env.num_envs, -1)
            unrolled_confidence = (
                unrolled_confidence * (unrolled_segmentation != 0))
            select_actions = numpy.zeros(
                (train_env.num_envs, segmentation_height, segmentation_width),
                dtype=numpy.bool)
            for i in range(max_instances_per_step):
                max_confidence_indices = torch.argmax(
                    unrolled_confidence, dim=-1)
                y = (max_confidence_indices // segmentation_height).cpu()
                x = (max_confidence_indices % segmentation_height).cpu()
                select_actions[range(train_env.num_envs), y, x] = True
                
                # turn off the selected brick instance
                instance_ids = unrolled_segmentation[
                    range(train_env.num_envs), max_confidence_indices]
                unrolled_confidence = (
                    unrolled_confidence *
                    (unrolled_segmentation != instance_ids.unsqueeze(1)))
            
            for i in range(train_env.num_envs):
                actions[i]['visibility'] = select_actions[i]
            
            #-------------------------------------------------------------------
            # prestep logging
            if step < log_train:
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock,
                        log,
                        step_observations,
                        space,
                        actions)
            
            '''
            ######
            all_accuracy = []
            for i, graph_state in enumerate(graph_states):
                true_labels = (
                        step_observations['graph_label']['instances']['label'])
                indices = graph_state.segment_id.view(-1).cpu().numpy()
                true_labels = true_labels[i][indices].reshape(-1)
                prediction = torch.argmax(graph_state.instance_label, dim=-1)
                prediction = prediction.cpu().numpy()
                correct = true_labels == prediction
                accuracy = float(numpy.sum(correct)) / correct.shape[0]
                all_accuracy.append(accuracy)
                #import pdb
                #pdb.set_trace()
            ######
            '''
            
            #-------------------------------------------------------------------
            # step
            (step_observations,
             step_rewards,
             step_terminal,
             step_info) = train_env.step(actions)
            step_terminal = torch.BoolTensor(step_terminal)
            
            #-------------------------------------------------------------------
            # poststep log
            '''
            log.add_scalar(
                    'train_rollout/reward',
                    sum(step_rewards)/len(step_rewards),
                    step_clock[0])
            all_edge_ap = [
                    info['graph_task']['edge_ap'] for info in step_info]
            all_instance_ap = [
                    info['graph_task']['instance_ap'] for info in step_info]
            
            log.add_scalar(
                    'train_rollout/step_edge_ap',
                    sum(all_edge_ap)/len(all_edge_ap),
                    step_clock[0])
            log.add_scalar(
                    'train_rollout/step_instance_ap',
                    sum(all_instance_ap)/len(all_instance_ap),
                    step_clock[0])
            
            num_terminal = sum(step_terminal)
            if num_terminal:
                sum_terminal_edge_ap = sum(
                        ap * t for ap, t in zip(all_edge_ap, step_terminal))
                sum_terminal_instance_ap = sum(
                        ap * t for ap, t in zip(all_instance_ap, step_terminal))
                log.add_scalar(
                        'train_rollout/terminal_edge_ap',
                        sum_terminal_edge_ap/num_terminal, step_clock[0])
                log.add_scalar(
                        'train_rollout/terminal_instance_ap',
                        sum_terminal_instance_ap/num_terminal, step_clock[0])
            
            seq_rewards.append(step_rewards)
            '''
            step_clock[0] += 1
    
    print('- '*40)
    print('Converting rollout data to tensors')
    
    # when joining these into one long list, make sure sequences are preserved
    seq_tensors = gym_space_list_to_tensors(
            seq_observations, train_env.single_observation_space)
    seq_terminal = torch.stack(seq_terminal, axis=1).reshape(-1)
    
    '''
    seq_returns = []
    ret = 0.
    gamma = 0.9
    for reward, terminal in zip(reversed(seq_rewards), reversed(seq_terminal)):
        ret += reward
        seq_returns.append(ret)
        ret *= gamma
        if terminal:
            ret = 0.
    seq_returns = torch.FloatTensor(list(reversed(seq_returns)))
    seq_norm_returns = (
            (seq_returns - torch.mean(seq_returns)) /
            (seq_returns.std() + 0.001))
    '''
    
    #===========================================================================
    # supervision
    
    print('- '*40)
    print('Supervising rollout data')
    
    running_node_loss = 0.
    running_confidence_loss = 0.
    total_correct_segments = 0
    total_correct_correct_segments = 0
    total_segments = 0
    
    step_model.train()
    edge_model.train()
    
    dataset_size = seq_tensors['color_render'].shape[0]
    tlast = 0
    for mini_epoch in range(1, mini_epochs+1):
        print('-   '*20)
        print('Training Mini Epoch: %i'%mini_epoch)
        
        # episode subsequences
        iterate = tqdm.tqdm(range(mini_epoch_sequences//batch_size))
        for seq_id in iterate:
            start_indices = torch.randint(dataset_size, (batch_size,))
            step_terminal = [True for _ in range(batch_size)]
            #graph_states = [None for _ in range(batch_size)]
            seq_loss = 0.
            
            # steps per episode subsequence
            for step in range(mini_epoch_sequence_length):
                step_indices = (start_indices + step) % dataset_size
                
                #--------------------------------------
                # RECENTLY MOVED
                # select graph state from memory for all terminal sequences
                # TODO: Is there more we need to do here to make sure gradients
                #       get clipped here?
                #for i, terminal in enumerate(step_terminal):
                #    if terminal:
                #        graph_states[i] = seq_graph_state[
                #                step_indices[i]].detach().cuda()
                
                # update step terminal
                step_terminal = seq_terminal[step_indices]
                #-------------------------------------
                
                # put the data on the graphics card
                x_im = seq_tensors['color_render'][step_indices].cuda()
                x_seg = seq_tensors['segmentation_render'][step_indices].cuda()
                #y_graph = seq_tensors['graph_label'][step_indices].cuda()
                
                # step forward pass
                '''
                viewpoint_input = None
                if 'viewpoint' in seq_tensors:
                    a = seq_tensors['viewpoint']['azimuth'][step_indices]
                    a = a.cuda()
                    e = seq_tensors['viewpoint']['elevation'][step_indices]
                    e = e.cuda()
                    d = seq_tensors['viewpoint']['distance'][step_indices]
                    d = d.cuda()
                    viewpoint_input = torch.stack((a,e,d), dim=1)
                '''
                head_features = step_model(x_im)
                # MURU: 'fcos_features' should exist as a key in head_features
                # here.  This is what should get losses applied.
                
                step_loss = 0.
                
                bs = head_features['x'].shape[0]
                '''
                (new_graph_states,
                 step_step_logits,
                 step_state_logits,
                 extra_extra_logits,
                 step_extra_logits,
                 state_extra_logits) = edge_model(
                        step_brick_lists,
                        graph_states,
                        extra_brick_vectors=input_extra_brick_vectors,
                        max_edges=max_edges,
                        segment_id_matching=segment_id_matching)
                '''
                #---------------------------------------------------------------
                # dense supervision
                
                '''
                # instance_label supervision
                instance_labels = seq_tensors['instance_labels']['label']
                instance_labels = instance_labels[step_indices][:,:,0].cuda()
                dense_instance_label_target = torch.stack([
                    label[seg] for label, seg in zip(instance_labels, x_seg)])
                '''
                
                #y_instance_label = [
                #        graph['instance_label'][:,0] for graph in y_graph]
                #dense_instance_label_target = torch.stack([
                #        y[seg] for y,seg in zip(y_instance_label, x_seg)])
                ##dense_instance_label_target = instance_labels 
                instance_label_logits = head_features['instance_label']
                instance_label_target = seq_tensors['dense_instance_labels'][
                    step_indices, :, :, 0].cuda()
                instance_label_loss = dense_instance_label_loss(
                        instance_label_logits,
                        instance_label_target,
                        instance_label_class_weight)
                step_loss = (step_loss +
                        instance_label_loss * instance_label_loss_weight)
                log.add_scalar(
                        'loss/instance_label',
                        instance_label_loss, step_clock[0])
                
                foreground = x_seg != 0
                foreground_total = torch.sum(foreground)
                if foreground_total:
                    
                    # rotation supervision -------------------------------------
                    rotation_prediction = head_features['pose']['rotation']
                    # this is both moving the matrix dimensions to the end
                    # and also transposing (inverting) the rotation matrix
                    rotation_prediction = rotation_prediction.permute(
                        (0, 3, 4, 2, 1))
                    
                    transform_labels = seq_tensors['dense_pose_labels']
                    transform_labels = transform_labels[step_indices].cuda()
                    rotation_labels = transform_labels[:,:,:,:,:3]
                    
                    rotation_offset = torch.matmul(
                        rotation_labels, rotation_prediction)
                    rotation_trace = (
                        rotation_offset[:,:,:,0,0] +
                        rotation_offset[:,:,:,1,1] +
                        rotation_offset[:,:,:,2,2]
                    )
                    # angle surrogate is a value between 0. and 1. that is
                    # monotonically consistent with the angle of the
                    # rotation offset matrix
                    angle_surrogate = (rotation_trace + 1.)/4.
                    dense_rotation_loss = 1. - angle_surrogate
                    dense_rotation_loss = dense_rotation_loss * foreground
                    rotation_loss = (
                        torch.sum(dense_rotation_loss) / foreground_total)
                    
                    step_loss = step_loss + rotation_loss * rotation_loss_weight
                    log.add_scalar(
                        'loss/rotation', rotation_loss, step_clock[0])
                    
                    # translation supervision ----------------------------------
                    translation_prediction = (
                        head_features['pose']['translation'])
                    translation_prediction = translation_prediction.permute(
                        (0, 2, 3, 1))
                    translation_prediction = (
                        translation_prediction * translation_scale_factor)
                    
                    translation_labels = (
                        transform_labels[:,:,:,:,3] * translation_scale_factor)
                    offset = translation_prediction - translation_labels
                    dense_translation_loss = (
                        offset[...,0]**2 + offset[...,1]**2 + offset[...,2]**2)
                    dense_translation_loss = dense_translation_loss * foreground
                    translation_loss = (
                        torch.sum(dense_translation_loss) / foreground_total)
                    
                    step_loss = (
                        step_loss + translation_loss * translation_loss_weight)
                    log.add_scalar(
                        'loss/translation', translation_loss, step_clock[0])
                    
                    # score supervision ----------------------------------------
                    instance_label_prediction = torch.argmax(
                            instance_label_logits, dim=1)
                    correct = instance_label_prediction == instance_label_target
                    score_target = correct
                    
                    score_loss = dense_score_loss(
                            head_features['confidence'],
                            score_target,
                            foreground)
                    
                    step_loss = step_loss + score_loss * score_loss_weight
                    log.add_scalar('loss/score', score_loss, step_clock[0])
                    log.add_scalar('train_accuracy/dense_instance_label',
                            float(torch.sum(correct)) /
                            float(torch.numel(correct)),
                            step_clock[0])
                    
                    '''
                    # center voting
                    if center_cluster_loss_weight > 0.:
                        center_data = head_features['cluster_center']
                        center_offsets = center_data[:,:2]
                        center_attention = center_data[:,[2]]
                        h, w = center_offsets.shape[2:]
                        y = torch.arange(h).view(1,1,h,1).expand(1,1,h,w).cuda()
                        x = torch.arange(w).view(1,1,1,w).expand(1,1,h,w).cuda()
                        positions = torch.cat((y,x), dim=1) + center_offsets
                        b = positions.shape[0]
                        
                        #normalizer = torch.ones_like(center_attention)
                        
                        exp_attention = torch.exp(center_attention)
                        exp_positions = positions * exp_attention
                        
                        #positions_1d = positions[:,0] * w + positions[:,1]
                        #positions_1d = positions_1d.view(b,2,-1)
                        summed_positions = scatter_add(
                                exp_positions.view(b, 2, -1),
                                x_seg.view(b, 1, -1))
                        normalizer = scatter_add(
                                exp_attention.view(b, 1, -1),
                                x_seg.view(b, 1, -1))
                        cluster_centers = (
                                summed_positions / normalizer)
                        
                        #--------
                        # losses
                        #--------
                        # make the offsets small
                        # (encourage cluster center to be near object center)
                        offset_norm = torch.norm(center_offsets, dim=1)
                        center_loss = torch.nn.functional.smooth_l1_loss(
                                offset_norm,
                                torch.zeros_like(offset_norm),
                                reduction='none')
                        center_loss = center_loss * (x_seg != 0)
                        center_loss = torch.mean(center_loss)
                        
                        # make the offsets as close as possible to the centers
                        offset_targets = []
                        for i in range(b):
                            offset_target_b = torch.index_select(
                                    cluster_centers[i], 1, x_seg[i].view(-1))
                            offset_targets.append(offset_target_b.view(2, h, w))
                        position_targets = torch.stack(offset_targets, dim=0)
                        cluster_loss = torch.nn.functional.smooth_l1_loss(
                                positions, position_targets, reduction='none')
                        cluster_loss = cluster_loss * (x_seg != 0).unsqueeze(1)
                        cluster_loss = torch.mean(cluster_loss)
                        
                        cluster_total_loss = (
                                center_loss * center_local_loss_weight +
                                cluster_loss * center_cluster_loss_weight)
                        
                        log.add_scalar('loss/center_loss',
                                center_loss * center_local_loss_weight,
                                step_clock[0])
                        log.add_scalar('loss/cluster_loss',
                                cluster_loss * center_cluster_loss_weight,
                                step_clock[0])
                        step_loss = step_loss + cluster_total_loss
                    
                    # removability loss
                    if removability_loss_weight > 0.:
                        removable_instances = seq_tensors[
                                'removability']['removable'][step_indices]
                        removable_instances = removable_instances.bool()
                        removable_directions = seq_tensors[
                                'removability']['direction'][step_indices]
                        
                        good_direction = removable_directions[:,:,2] >= 0.
                        #good_direction_b = torch.norm(
                        #        removable_directions, dim=2) < 1e-5
                        #good_direction = good_direction_a | good_direction_b
                        removable_instances = (
                                removable_instances & good_direction)
                        if not torch.all(
                                torch.sum(removable_instances, dim=-1)>0):
                            print('NO REMOVABLE INSTANCES!!!')
                        
                        removability_target = []
                        for i in range(len(y_graph)):
                            target = removable_instances[i][x_seg[i]]
                            removability_target.append(target)
                        removability_target = torch.stack(removability_target)
                        removability_target = removability_target.cuda()
                        #removability_loss = (
                        # torch.nn.functional.binary_cross_entropy_with_logits(
                        #        head_features['removability'],
                        #        removability_target.unsqueeze(1)))
                        removability_loss = dense_score_loss(
                                head_features['removability'],
                                removability_target,
                                foreground)
                        
                        step_loss = step_loss + (
                                removability_loss * removability_loss_weight)
                    else:
                        removability_target = None
                    '''
                
                '''
                instance_correct = 0.
                total_instances = 0.
                for brick_list, target_graph in zip(step_brick_lists, y_graph):
                    instance_label_target = (
                            target_graph.instance_label[
                                brick_list.segment_id[:,0]])[:,0]
                    if brick_list.num_nodes:
                        instance_label_pred = torch.argmax(
                                brick_list.instance_label, dim=-1)
                        instance_correct += float(torch.sum(
                                instance_label_target ==
                                instance_label_pred).cpu())
                        total_instances += instance_label_pred.shape[0]
                if total_instances:
                    log.add_scalar('train_accuracy/step_instance_label',
                            instance_correct / total_instances,
                            step_clock[0])
                else:
                    print('no instances?')
                '''
                log.add_scalar('loss/total', step_loss, step_clock[0])
                
                seq_loss = seq_loss + step_loss
                
                '''
                if seq_id < log_train:
                    log_train_supervision_step(
                            # log
                            step_clock,
                            log,
                            # input
                            x_im.cpu().numpy(),
                            x_seg.cpu().numpy(),
                            # predictions
                            instance_label_logits.cpu().detach(),
                            torch.sigmoid(dense_score_logits).cpu().detach(),
                            step_brick_lists.cpu().detach(),
                            graph_states,
                            [torch.sigmoid(l).cpu().detach()
                             for l in step_step_logits],
                            [torch.sigmoid(l).cpu().detach()
                             for l in step_state_logits],
                            # ground truth
                            y_graph.cpu(),
                            score_target,
                            removability_target,
                            head_features.get('removability', None))
                '''
                step_clock[0] += 1
            
            seq_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

def log_train_rollout_step(
        step_clock,
        log,
        step_observations,
        space,
        actions):
    #log_step_observations('train', step_clock, log, step_observations, space)
    log.add_text('train/actions',
            json.dumps(actions, cls=NumpyEncoder, indent=2))

def log_train_supervision_step(
        # log
        step_clock,
        log,
        # input
        images,
        segmentations,
        # predictions
        dense_label_logits,
        dense_scores,
        pred_brick_lists,
        pred_graph_states,
        pred_step_step,
        pred_step_state,
        # ground truth
        true_graph,
        score_targets,
        removability_target,
        removability_prediction):
    
    '''
    for (image, segmentation, label_logits, scores) in (
            images,
            segmentations,
            dense_label_logits,
            dense_scores):
        make_image_strip(
    '''
    
    #image_strip = make_image_strip(images
    
    #input
    log.add_image('train/train_images', images, step_clock[0],
            dataformats='NCHW')
    segmentation_mask = masks.color_index_to_byte(segmentations)
    log.add_image('train/train_segmentation', segmentation_mask, step_clock[0],
            dataformats='NHWC')
    
    # prediction
    dense_label_prediction = torch.argmax(dense_label_logits, dim=1).numpy()
    dense_label_mask = masks.color_index_to_byte(dense_label_prediction)
    log.add_image(
            'train/pred_dense_label_mask',
            dense_label_mask,
            step_clock[0],
            dataformats='NHWC')
    
    log.add_image(
            'train/pred_dense_scores',
            dense_scores,
            step_clock[0],
            dataformats='NCHW')
    
    if removability_prediction is not None:
        log.add_image(
                'train/pred_removability',
                torch.sigmoid(removability_prediction),
                step_clock[0],
                dataformats='NCHW')
    
    true_class_label_images = []
    pred_class_label_images = []
    true_instance_label_lookups = []
    pred_instance_label_lookups = []
    for i in range(len(true_graph)):
        true_instance_label_lookup = true_graph[i].instance_label.numpy()[:,0]
        true_instance_label_lookups.append(true_instance_label_lookup)
        class_label_segmentation = true_instance_label_lookup[segmentations[i]]
        class_label_image = masks.color_index_to_byte(class_label_segmentation)
        true_class_label_images.append(class_label_image)
        
        if not pred_brick_lists[i].num_nodes:
            #pred_class_label_images.append(numpy.ones((256,256,3)))
            pred_class_label_images.append(numpy.ones_like(class_label_image))
            continue
        pred_instance_label_lookup = numpy.zeros(
                true_instance_label_lookup.shape, dtype=numpy.long)
        pred_lookup_partial = (
                torch.argmax(pred_brick_lists[i].instance_label, dim=-1))
        pred_instance_label_lookup[pred_brick_lists[i].segment_id[:,0]] = (
                pred_lookup_partial.numpy())
        pred_instance_label_lookups.append(pred_instance_label_lookup)
        class_label_segmentation = pred_instance_label_lookup[segmentations[i]]
        class_label_image = masks.color_index_to_byte(class_label_segmentation)
        pred_class_label_images.append(class_label_image)
    
    pred_class_label_images = numpy.stack(pred_class_label_images)
    log.add_image(
            'train/pred_instance_label_mask',
            pred_class_label_images,
            step_clock[0],
            dataformats='NHWC')
    
    max_size = max(step_step.shape[0] for step_step in pred_step_step)
    step_step_match_image = torch.zeros(
            len(pred_step_step), 3, max_size+1, max_size+1)
    step_step_edge_image = torch.zeros(
            len(pred_step_step), 3, max_size+1, max_size+1)
    for i, step_step in enumerate(pred_step_step):
        size = step_step.shape[0]
        # borders
        step_colors = masks.color_index_to_byte(
                pred_brick_lists[i].segment_id[:,0]).reshape(-1,3)
        step_step_match_image[i, :, 0, 1:size+1] = torch.FloatTensor(
                step_colors).T/255.
        step_step_match_image[i, :, 1:size+1, 0] = torch.FloatTensor(
                step_colors).T/255.
        step_step_edge_image[i, :, 0, 1:size+1] = torch.FloatTensor(
                step_colors).T/255.
        step_step_edge_image[i, :, 1:size+1, 0] = torch.FloatTensor(
                step_colors).T/255.
        
        # content
        step_step_match_image[i, :, 1:size+1, 1:size+1] = (
                step_step[:,:,0].unsqueeze(0))
        step_step_edge_image[i, :, 1:size+1, 1:size+1] = (
                step_step[:,:,1].unsqueeze(0))
    
    log.add_image(
            'train/step_step_match_prediction',
            step_step_match_image,
            step_clock[0],
            dataformats='NCHW')
            
    log.add_image(
            'train/pred_step_step_edge_score',
            step_step_edge_image,
            step_clock[0],
            dataformats='NCHW')
    
    max_h = max(step_state.shape[0] for step_state in pred_step_state)
    max_h = max(1, max_h)
    max_w = max(step_state.shape[1] for step_state in pred_step_state)
    max_w = max(1, max_w)
    step_state_match_image = torch.zeros(
            len(pred_step_state), 3, max_w+1, max_h+1)
    step_state_edge_image = torch.zeros(
            len(pred_step_state), 3, max_w+1, max_h+1)
    for i, step_state in enumerate(pred_step_state):
        h,w = step_state.shape[:2]
        # borders
        step_colors = masks.color_index_to_byte(
                pred_brick_lists[i].segment_id[:,0]).reshape(
                    -1,3)
        state_colors = masks.color_index_to_byte(
                pred_graph_states[i].cpu().detach().segment_id[:,0]).reshape(
                    -1,3)
        
        step_state_match_image[i, :, 0, 1:h+1] = torch.FloatTensor(
                step_colors).T/255.
        step_state_match_image[i, :, 1:w+1, 0] = torch.FloatTensor(
                state_colors).T/255.
        
        step_state_edge_image[i, :, 0, 1:h+1] = torch.FloatTensor(
                step_colors).T/255
        step_state_edge_image[i, :, 1:w+1, 0] = torch.FloatTensor(
                state_colors).T/255.
        
        # content
        step_state_match_image[i, :, 1:w+1, 1:h+1] = (
                step_state[:,:,0].T.unsqueeze(0))
        step_state_edge_image[i, :, 1:w+1, 1:h+1] = (
                step_state[:,:,1].T.unsqueeze(0))
    log.add_image(
            'train/pred_step_state_match_score',
            step_state_match_image,
            step_clock[0],
            dataformats='NCHW')
    log.add_image(
            'train/pred_step_state_edge_score',
            step_state_edge_image,
            step_clock[0],
            dataformats='NCHW')
    
    
    log.add_text('train/pred_instance_labels',
            json.dumps(pred_instance_label_lookups,
                    cls=NumpyEncoder, indent=2),
            step_clock[0])
    
    # ground truth
    true_class_label_images = numpy.stack(true_class_label_images)
    log.add_image(
            'train/true_instance_label_mask',
            true_class_label_images,
            step_clock[0],
            dataformats='NHWC')
    
    log.add_image('train/true_score_labels',
            score_targets.unsqueeze(1),
            step_clock[0],
            dataformats='NCHW')
    
    if removability_target is not None:
        log.add_image('train/true_removability_targets',
                removability_target.unsqueeze(1),
                step_clock[0],
                dataformats='NCHW')
    
    log.add_text('train/true_instance_labels',
            json.dumps(true_instance_label_lookups,
                    cls=NumpyEncoder, indent=2),
            step_clock[0])

def log_step_observations(split, step_clock, log, step_observations, space):
    label = '%s/observations'%split
    gym_log(label, step_observations, space, log, step_clock[0])

def test_label_confidence_epoch(
        epoch,
        step_clock,
        log,
        steps,
        step_model,
        edge_model,
        test_env,
        debug_dump=False):
    
    print('-'*80)
    print('Test')
    
    step_observations = test_env.reset()
    step_terminal = numpy.ones(test_env.num_envs, dtype=numpy.bool)
    graph_states = [None] * test_env.num_envs
    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            # gym -> torch
            step_tensors = gym_space_to_tensors(
                    step_observations,
                    test_env.single_observation_space,
                    torch.cuda.current_device())
            
            # step model forward pass
            step_brick_lists, _, dense_score_logits, head_features = step_model(
                    step_tensors['color_render'],
                    step_tensors['segmentation_render'])
            
            # build new graph state for all terminal sequences
            # (do this here so we can use the brick_feature_spec from the model)
            for i, terminal in enumerate(step_terminal):
                if terminal:
                    graph_states[i] = BrickGraph(
                            BrickList(step_brick_lists[i].brick_feature_spec()),
                            edge_attr_channels=1).cuda()
            
            # upate the graph state using the edge model
            graph_states, _, _ = edge_model(step_brick_lists, graph_states)
            
            print([brick_list.num_nodes for brick_list in step_brick_lists])
            print([graph_state.num_nodes for graph_state in graph_states])
            edge_dicts = [graph.edge_dict() for graph in graph_states]
            
            print(edge_dicts)
            lkjlkjljklkj
            
            
            # sample an action
            hide_logits = [sbl.hide_action.view(-1) for sbl in step_brick_lists]
            hide_distributions = []
            bad_distributions = []
            for i, logits in enumerate(hide_logits):
                try:
                    distribution = Categorical(
                            logits=logits, validate_args=True)
                except ValueError:
                    bad_distrbutions.append(i)
                    distribution = Categorical(probs=torch.ones(1).cuda())
                hide_distributions.append(distribution)
            
            if len(bad_distributions):
                print('BAD DISTRIBUTIONS, REVERTING TO 0')
                print('STEP: %i, GLOBAL_STEP: %i'%(step, step_clock[0]))
                print('BAD INDICES:', bad_distributions)
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock, log, step_observations, space, [])
            
            segment_samples = [dist.sample() for dist in hide_distributions]
            #instance_samples = graph.remap_node_indices(
            #        batch_graphs, segment_samples, 'segment_id')
            instance_samples = [brick_list.segment_id[sample]
                    for brick_list, sample
                    in zip(step_brick_lists, segment_samples)]
            actions = [{'visibility':int(i.cpu())} for i in instance_samples]
            
            if step < log_test:
                space = train_env.single_observation_space
                log_train_rollout_step(
                        step_clock,
                        log,
                        step_observations,
                        space,
                        actions)
            
            (step_observations,
             step_rewards,
             step_terminal,
             _) = train_env.step(actions)
            
            seq_rewards.append(step_rewards)
            
            step_clock[0] += 1










            
            
            instance_label_logits = batch_graphs.instance_label
            segment_index = batch_graphs.segment_index
            ptr = batch_graphs.ptr
            instance_label_targets = torch.cat([
                    y[segment_index[start:end]]
                    for y, start, end in zip(y_batch, ptr[:-1], ptr[1:])])
            prediction = torch.argmax(instance_label_logits, dim=1)
            correct = instance_label_targets == prediction
            total_correct_segments += float(torch.sum(correct).cpu())
            total_segments += correct.shape[0]
            
            #action_prob = torch.exp(action_logits) * segment_weights
            #max_action = torch.argmax(action_prob, dim=-1)
            #max_correct = correct[range(batch_size), max_action]
            #max_is_correct_segments += int(torch.sum(max_correct))
            #max_is_correct_normalizer += batch_size
            # this is bad, but we're not training the hide_actions yet
                
            '''
            action_prob = torch.sigmoid(action_logits) * segment_weights
            max_action = torch.argmax(action_prob, dim=-1)
            add_to_graph = action_prob > 0.5
            add_to_graph_correct += torch.sum(correct * add_to_graph)
            add_to_graph_normalizer += torch.sum(add_to_graph)
            '''
            
            if debug_dump:
                for i in range(batch_size):
                    Image.fromarray(images[i]).save(
                            'image_%i_%i_%i.png'%(epoch, i, step))
                    max_index = segment_ids[i,max_action[i]]
                    max_highlight = segmentations[i] == int(max_index)
                    Image.fromarray(max_highlight.astype(numpy.uint8)*255).save(
                            'highlight_%i_%i_%i.png'%(epoch, i, step))
                    
                    action_weights = (
                            torch.sigmoid(action_logits[i]) *
                            segment_weights[i])
                    #confidence_image = image_generators.segment_weight_image(
                    #        segmentations[i],
                    #        action_weights.cpu().numpy(),
                    #        segment_ids[i].cpu().numpy())
                    #Image.fromarray(confidence_image).save(
                    #        'segment_confidence_%i_%i_%i.png'%(epoch, i, step))
                    #correct_image = image_generators.segment_weight_image(
                    #        segmentations[i],
                    #        correct[i].cpu().numpy(),
                    #        segment_ids[i].cpu().numpy())
                    #Image.fromarray(correct_image).save(
                    #        'correct_%i_%i_%i.png'%(epoch, i, step))
                    
                    metadata = {
                        'max_action' : int(max_action[i]),
                        'max_label' : int(y[i,max_action[i]]),
                        'max_prediction' : int(prediction[i,max_action[i]]),
                        'all_labels' : y[i].cpu().tolist(),
                        'all_predictions' : prediction[i].cpu().tolist()
                    }
                    with open('details_%i_%i_%i.json'%(epoch,i,step), 'w') as f:
                        json.dump(metadata, f)
            
            '''
            confidence_prediction = action_logits > 0.
            correct_correct = (
                    (confidence_prediction == correct) * segment_weights)
            total_correct_correct_segments += float(
                    torch.sum(correct_correct).cpu())
            '''
            '''
            #action_distribution = Categorical(logits=action_logits)
            #hide_actions = action_distribution.sample().cpu()
            hide_actions = max_action
            hide_indices = [segment_ids[i,action]
                    for i, action in enumerate(hide_actions)]
            actions = [{'visibility':int(action)} for action in hide_indices]
            '''
            '''
            # sample an action
            # this is what we should be doing, but we're not training
            # visibility separately yet
            #hide_probs = torch.exp(batch_graph.hide_action.view(-1))
            #==========
            # this is what we are doing, reusing score (for now)
            hide_probs = batch_graph.score
            #hide_logits = torch.log(hide_probs / (1. - hide_probs))
            #==========
            #hide_distributions = graph.batch_graph_categoricals(
            #        batch_graph, logits=hide_logits)
            #segment_samples = [dist.sample() for dist in hide_distributions]
            split_probs = graph.split_node_value(batch_graph, hide_probs)
            segment_samples = [torch.argmax(p) for p in split_probs]
            instance_samples = graph.remap_node_indices(
                    batch_graph, segment_samples, 'segment_index')
            actions = [{'visibility':int(i.cpu())} for i in instance_samples]
            '''
            GET_THE_ABOVE_FROM_TRAIN
            
            step_observations, _, terminal, _ = test_env.step(actions)
    
    print('- '*40)
    node_accuracy = total_correct_segments/total_segments
    #confidence_accuracy = total_correct_correct_segments/total_segments
    #top_confidence_accuracy = (
    #        add_to_graph_correct / add_to_graph_normalizer)
            #max_is_correct_segments / max_is_correct_normalizer)
    
    log.add_scalar('test_accuracy/node_labels', node_accuracy, step_clock[0])
    #log.add_scalar('test_accuracy/confidence_accuracy',
    #        confidence_accuracy, step_clock[0])
    #log.add_scalar('test_accuracy/confident_node_accuracy',
    #        top_confidence_accuracy, step_clock[0])
