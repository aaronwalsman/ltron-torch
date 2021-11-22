from ltron_torch.dataset.rollout_dataset import build_rolloutFrames_train_loader
import random
import time
import os

import numpy

import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from ltron_torch.models.simple_fcn import named_resnet_fcn
from ltron_torch.models.heads import Conv2dMultiheadDecoder

from ltron.dataset.paths import get_dataset_info
from ltron_torch.config import Config
from ltron_torch.train.optimizer import adamw_optimizer
import pdb
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, default_tile_transform, default_image_transform, default_image_untransform)
from splendor.image import save_image

class PretrainResnetBackboneConfig(Config):
    epochs = 8
    batch_size = 16
    loader_workers = 8

    learning_rate = 3e-4
    weight_decay = 0.1
    betas = (0.9, 0.95)

    dataset = 'random_six'
    train_split = 'simple_single_seq'
    train_subset = None
    test_split = 'simple_single'
    test_subset = None

    test_frequency = 1
    checkpoint_frequency = 10
    visualization_frequency = 1


def train_disassembly_behavior_cloning(train_config):
    print('=' * 80)
    print('Setup')
    print('-' * 80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    step = 0
    train_start = time.time()

    model = build_model(train_config)
    optimizer = adamw_optimizer(model, train_config)
    train_loader = build_rolloutFrames_train_loader(train_config)

    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.time()
        print('=' * 80)
        print('Epoch: %i' % epoch)

        train_pass(
            train_config, model, optimizer, train_loader, log, clock, step)
        save_checkpoint(train_config, epoch, model, optimizer, log, clock)
        # episodes = test_epoch(
        #     train_config, epoch, test_env, model, log, clock)
        # visualize_episodes(train_config, epoch, episodes, log, clock)

        train_end = time.time()
        print('=' * 80)
        print('Train elapsed: %0.02f seconds' % (train_end - train_start))


def build_model(config):
    print('-' * 80)
    print('Building resnet disassembly model')
    info = get_dataset_info(config.dataset)
    dense_heads = {"class" : max(info["class_ids"].values())+1, "pos_snap" : 1, "neg_snap" : 1}
    model = named_resnet_fcn(
        'resnet50',
        256,
        dense_heads=Conv2dMultiheadDecoder(256, dense_heads, kernel_size=1)
    )

    return model.cuda()


def train_pass(train_config, model, optimizer, train_loader, log, clock, step):
    model.train()
    counter = 0
    for workspace, label in tqdm.tqdm(train_loader):
        # convert observations to model tensors --------------------------------
        workspace = workspace.cuda()

        # forward --------------------------------------------------------------
        # xg, xd = model(xw, xh)
        output, = model(workspace)
        class_id, pos_snap, neg_snap = output["class"], output['pos_snap'], output['neg_snap']
        pos_snap = pos_snap.view(pos_snap.shape[0], pos_snap.shape[2], pos_snap.shape[3])
        neg_snap = neg_snap.view(neg_snap.shape[0], neg_snap.shape[2], neg_snap.shape[3])
        pos_snap = torch.squeeze(pos_snap)
        neg_snap = torch.squeeze(neg_snap)

        # loss -----------------------------------------------------------------
        label = label.cuda()
        id_label = label[:, :, :, 0]
        pos_label = label[:, :, :, 1]
        neg_label = label[:, :, :, 2]
        loss_id, loss_pos, loss_neg = cross_entropy(class_id, id_label), binary_cross_entropy_with_logits(pos_snap, pos_label.float()), binary_cross_entropy_with_logits(neg_snap, neg_label.float())
        loss = loss_id + loss_pos + loss_neg
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        log.add_scalar('train/class_loss', loss_id, clock[0])
        log.add_scalar('train/pos_loss', loss_pos, clock[0])
        log.add_scalar('train/neg_loss', loss_neg, clock[0])
        clock[0] += 1

        # Accuracy -----------------------------------------------------------------
        id_pred = torch.argmax(class_id, dim=1)
        pos_pred = torch.where(torch.sigmoid(pos_snap) > 0.5, 1, 0)
        neg_pred = torch.where(torch.sigmoid(pos_snap) > 0.5, 1, 0)
        non_back = torch.sum(torch.where(label[:,:,:,0]>0, 1, 0))
        id_corr = torch.sum(torch.where((label[:,:,:,0] == id_pred) & (label[:,:,:,0] != 0), 1, 0))
        pos_corr = torch.sum(torch.where((label[:,:,:,1] == pos_pred) & (label[:,:,:,1] != 0), 1, 0))
        neg_corr = torch.sum(torch.where((label[:, :, :, 2] == neg_pred) & (label[:, :, :, 2] != 0), 1, 0))
        counter += 1
        if counter == 1000:
            log.add_scalar("train/id_acc", id_corr/non_back, clock[0], step)
            log.add_scalar("train/pos_acc", pos_corr / non_back, clock[0], step)
            log.add_scalar("train/neg_acc", neg_corr / non_back, clock[0], step)
            step += 1
            counter = 0



    # torch.save(model.state_dict(), "pretrain_models/standard_fcn.pth")
    # torch.save(optimizer.state_dict(), "pretrain_models/standard_fcn_optim.pth")

def test_model(test_config, model):
    test_loader = build_rolloutFrames_train_loader(test_config)
    with torch.no_grad():
        for workspace, label in test_loader:
            workspace = workspace.cuda()
            output, = model(workspace)
            class_id, pos_snap, neg_snap = output["class"], output['pos_snap'], output['neg_snap']
            id_pred = torch.argmax(class_id[0], dim=0)
            pos_pred = torch.where(torch.sigmoid(pos_snap[0]) > 0.5, 1, 0)
            neg_pred = torch.where(torch.sigmoid(pos_snap[0]) > 0.5, 1, 0)


def save_checkpoint(train_config, epoch, model, optimizer, log, clock):
    frequency = train_config.checkpoint_frequency
    if frequency is not None and epoch % frequency == 0:
        checkpoint_directory = os.path.join(
            './checkpoint', os.path.split(log.log_dir)[-1])
        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)

        print('-' * 80)
        model_path = os.path.join(
            checkpoint_directory, 'model_%04i.pt' % epoch)
        print('Saving model to: %s' % model_path)
        torch.save(model.state_dict(), model_path)

        optimizer_path = os.path.join(
            checkpoint_directory, 'optimizer_%04i.pt' % epoch)
        print('Saving optimizer to: %s' % optimizer_path)
        torch.save(optimizer.state_dict(), optimizer_path)

def visualization(train_config):
    model = build_model(train_config)
    model.load_state_dict(torch.load("pretrain_models/standard_fcn.pth"))
    model.eval()
    test_loader = build_rolloutFrames_train_loader(train_config)
    counter = 0
    with torch.no_grad():
        for workspace, label in test_loader:
            workspace = workspace.cuda()
            output, = model(workspace)
            class_id, pos_snap, neg_snap = output["class"], output['pos_snap'], output['neg_snap']
            id_pred = torch.argmax(class_id[0], dim = 0)
            pos_pred = torch.where(torch.sigmoid(pos_snap[0])>0.5, 1, 0)
            neg_pred = torch.where(torch.sigmoid(pos_snap[0])>0.5, 1, 0)
            im = default_image_untransform(id_pred)
            save_image(im, "model_outcome/" + str(counter) + "id_pred.png")
            im = default_image_untransform(workspace[0])
            save_image(im, "model_outcome/" + str(counter) + "workspace.png")
            im = default_image_untransform(pos_pred)
            save_image(im, "model_outcome/" + str(counter) + "pos_snap.png")
            im = default_image_untransform(neg_pred)
            save_image(im, "model_outcome/" + str(counter) + "neg_snap.png")
            im = default_image_untransform(label[0,:,:,0])
            save_image(im, "model_outcome/" + str(counter) + "id_label.png")
            im = default_image_untransform(label[0,:,:,1])
            save_image(im, "model_outcome/" + str(counter) + "pos_label.png")
            im = default_image_untransform(label[0,:,:,2])
            save_image(im, "model_outcome/" + str(counter) + "neg_label.png")
            counter += 1
            if counter == 10:
                pdb.set_trace()

def main():
    config = PretrainResnetBackboneConfig.load_config("../../experiments/pretrainbackbone_resnet/settings.cfg")
    train_disassembly_behavior_cloning(config)

if __name__ == '__main__' :
    config = PretrainResnetBackboneConfig.load_config("../../experiments/pretrainbackbone_resnet/settings.cfg")
    main()
    # visualization(config)
