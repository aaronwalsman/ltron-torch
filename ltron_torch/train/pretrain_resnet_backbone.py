from pathlib import Path
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
from ltron_torch.train.optimizer import build_optimizer
import pdb
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from ltron_torch.gym_tensor import (
    gym_space_to_tensors, default_tile_transform, default_image_transform, default_image_untransform)
from splendor.image import save_image
from ltron_torch.models.deeplabv3 import deeplabv3
import matplotlib.pyplot as plt

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
    optimizer = "adamw"

def train_disassembly_behavior_cloning(train_config):
    print('=' * 80)
    print('Setup')
    print('-' * 80)
    print('Log')
    log = SummaryWriter()
    clock = [0]
    step = 0
    train_start = time.time()

    model_deeplab = build_model_deeplab(train_config)
    # model = build_model(train_config)
    model_deeplab.load_state_dict(torch.load("./checkpoint/Jan01_17-09-55_patillo/model_0050.pt"))
    optimizer = build_optimizer(model_deeplab, train_config)
    optimizer.load_state_dict(torch.load("./checkpoint/Jan01_17-09-55_patillo/optimizer_0050.pt"))
    train_loader = build_rolloutFrames_train_loader(train_config)
    for epoch in range(1, train_config.epochs + 1):
        epoch_start = time.time()
        print('=' * 80)
        print('Epoch: %i' % epoch)

        train_pass(
            train_config, None, optimizer, train_loader, log, clock, step, model_deeplab)
        save_checkpoint(train_config, epoch, model_deeplab, optimizer, log, clock)
        # episodes = test_epoch(
        #     train_config, epoch, test_env, model, log, clock)
        # visualize_episodes(train_config, epoch, episodes, log, clock)

        train_end = time.time()
        print('=' * 80)
        print('Train elapsed: %0.02f seconds' % (train_end - train_start))

    checkpoint_directory = os.path.join(
        './checkpoint', os.path.split(log.log_dir)[-1])
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    print('-' * 80)
    model_path = os.path.join(
        checkpoint_directory, 'model_%04i.pt' % epoch)
    print('Saving model to: %s' % model_path)
    torch.save(model_deeplab.state_dict(), model_path)

    optimizer_path = os.path.join(
        checkpoint_directory, 'optimizer_%04i.pt' % epoch)
    print('Saving optimizer to: %s' % optimizer_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def build_model(config):
    print('-' * 80)
    print('Building resnet disassembly model')
    info = get_dataset_info(config.dataset)
    # dense_heads = {"class" : max(info["class_ids"].values())+1, "pos_snap" : 1, "neg_snap" : 1, "color" : max(info['color_ids'].values())+1}
    dense_heads = {"pos_snap" : 1, "neg_snap" : 1, "color" : max(info['color_ids'].values())+1}
    model = named_resnet_fcn(
        'resnet50',
        256,
        dense_heads=Conv2dMultiheadDecoder(256, dense_heads, kernel_size=1),
        frozen_batchnorm=True,
        pretrained=True
    )


    return model.cuda()

def build_model_deeplab(config):
    print('-' * 80)
    print('Building deeplab disassembly model')
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    # model = torch.nn.Sequential(*list(model.children()))
    info = get_dataset_info(config.dataset)
    # class_num = max(info['class_ids'].values())+1
    color_num = max(info['color_ids'].values())+1
    # model = deeplabv3.deeplabv3_resnet50(pretrained_backbone=True, pretrained=False, num_classes=class_num, heads={"class":class_num, "pos_snap":1, "neg_snap":1})
    model = deeplabv3.deeplabv3_resnet50(pretrained_backbone=True, pretrained=False, num_classes=color_num, heads={"color":color_num})
    # model.classifier[4] = torch.nn.AvgPool2d(kernel_size = (4,4), stride=4)
    # model.classifier.add_module("pred", torch.nn.Conv2d(256, class_num, kernel_size = (3,3), stride=2))
    # model.classifier[4] = torch.nn.Conv2d(256, class_num, kernel_size=1)
    print(model)
    # pdb.set_trace()
    return model.cuda()

def fake_train_pass(model, train_loader):
    model.eval()
    for workspace, label in train_loader:
        workspace = workspace.cuda()
        output = model(workspace)
        pdb.set_trace()
        break

def train_pass(train_config, model, optimizer, train_loader, log, clock, step, model_deeplab):
    # model.train()
    model_deeplab.train()
    counter = 0
    for workspace, label in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        # convert observations to model tensors --------------------------------
        workspace = workspace.cuda()
        # forward --------------------------------------------------------------
        # xg, xd = model(xw, xh)
        # output, = model(workspace)
        output = model_deeplab(workspace)
        # pdb.set_trace()
        # class_id = torch.nn.functional.interpolate(class_id, (64,64), mode="bilinear")
        # class_id, pos_snap, neg_snap, color_pred = output["class"], output['pos_snap'], output['neg_snap'], output['color']
        # class_id, pos_snap, neg_snap = output["class"], output['pos_snap'], output['neg_snap']
        # pos_snap = pos_snap.view(pos_snap.shape[0], pos_snap.shape[2], pos_snap.shape[3])
        # neg_snap = neg_snap.view(neg_snap.shape[0], neg_snap.shape[2], neg_snap.shape[3])
        # pos_snap = torch.squeeze(pos_snap)
        # neg_snap = torch.squeeze(neg_snap)
        color_pred = output['color']

        # loss -----------------------------------------------------------------
        label = label.cuda()
        # id_label = label[:, :, :, 0]
        # pos_label = label[:, :, :, 1]
        # neg_label = label[:, :, :, 2]
        color_label = label[:, :, :, 3]
        # id_weight = torch.ones(class_id.shape[1]).cuda()
        # id_weight[0] = 0.03
        color_weight = torch.ones(color_pred.shape[1]).cuda()
        color_weight[0] = 0.03
        # pdb.set_trace()
        # pos_weight = torch.ones(1).cuda()
        # neg_weight = torch.ones(1).cuda()
        # pos_weight[0] = 3
        # neg_weight[0] = 3
        # loss_id, loss_pos, loss_neg = cross_entropy(class_id, id_label, weight=id_weight), binary_cross_entropy_with_logits(pos_snap, pos_label.float(), pos_weight=pos_weight), \
        #                          binary_cross_entropy_with_logits(neg_snap, neg_label.float(), pos_weight=neg_weight)
        # loss_id = cross_entropy(class_id, id_label, weight=id_weight)
        # loss_id_t, loss_pos_t, loss_neg_t = cross_entropy(class_id, id_label,
        #                                             weight=id_weight), binary_cross_entropy_with_logits(pos_snap,
        #                                                                                                 pos_label.float(),), \
        #                               binary_cross_entropy_with_logits(neg_snap, neg_label.float(),)
        # loss = loss_id + loss_pos + loss_neg
        loss_color = cross_entropy(color_pred, color_label, weight=color_weight)
        loss_color.backward()
        optimizer.step()
        # log.add_scalar('train/class_loss', loss_id*1000, clock[0])
        # log.add_scalar('train/pos_loss', loss_pos*1000, clock[0])
        # log.add_scalar('train/neg_loss', loss_neg*1000, clock[0])
        log.add_scalar('train/color_loss', loss_color*1000, clock[0])
        clock[0] += 1

        # Accuracy -----------------------------------------------------------------
        counter += 1
        if counter == 12:
            # id_pred = torch.argmax(class_id, dim=1)
            # pos_pred = torch.where(torch.sigmoid(pos_snap) > 0.5, 1, 0)
            # neg_pred = torch.where(torch.sigmoid(neg_snap) > 0.5, 1, 0)
            color_pred = torch.argmax(color_pred, dim=1)
            non_back = torch.sum(torch.where(label[:,:,:,0]>0, 1, 0))
            # id_corr = torch.sum(torch.where((label[:,:,:,0] == id_pred) & (label[:,:,:,0] != 0), 1, 0))
            # pos_corr = torch.sum(torch.where((label[:,:,:,1] == pos_pred) & (label[:,:,:,0] != 0), 1, 0))
            # neg_corr = torch.sum(torch.where((label[:, :, :, 2] == neg_pred) & (label[:, :, :, 0] != 0), 1, 0))
            color_corr = torch.sum(torch.where((label[:, :, :, 3] == color_pred) & (label[:, :, :, 0] != 0), 1, 0))
            # log.add_scalar("train/id_acc", id_corr/non_back, clock[0], step)
            # log.add_scalar("train/pos_acc", pos_corr / non_back, clock[0], step)
            # log.add_scalar("train/neg_acc", neg_corr / non_back, clock[0], step)
            log.add_scalar("train/color_acc", color_corr / non_back, clock[0], step)
            step += 1
            counter = 0
    # log.add_scalar("train/id_acc", id_corr / non_back, clock[0], step)
    # log.add_scalar("train/pos_acc", pos_corr / non_back, clock[0], step)
    # log.add_scalar("train/neg_acc", neg_corr / non_back, clock[0], step)
    # log.add_scalar("train/color_acc", color_corr / non_back, clock[0], step)
    step += 1


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

def visualization(train_config, model_path):
    model = build_model_deeplab(train_config)
    model.load_state_dict(torch.load(model_path))
    model.train()
    test_loader = build_rolloutFrames_train_loader(train_config, batch_overload=None)
    print(len(test_loader))
    destination = "model_outcome_color"
    Path(destination).mkdir(parents=False, exist_ok=True)
    counter = 0
    with torch.no_grad():
        for workspace, label in test_loader:
            workspace = workspace.cuda()
            output = model(workspace)
            # class_id = torch.nn.functional.interpolate(output, (64,64), mode="bilinear")
            # class_id, pos_snap, neg_snap = output["class"], output['pos_snap'], output['neg_snap']
            # class_id = output
            # id_pred = torch.argmax(class_id[0], dim = 0)
            # pos_pred = torch.sigmoid(pos_snap[0]).squeeze().cpu().numpy()
            # neg_pred = torch.sigmoid(neg_snap[0]).squeeze().cpu().numpy()
            # plt.imshow(pos_pred, cmap="gray", interpolation="nearest")
            # plt.colorbar()
            # plt.show()
            # pdb.set_trace()
            # exit(0)
            # pos_pred_t = torch.where(torch.sigmoid(pos_snap[0])>0.5, 1, 0)
            # neg_pred_t = torch.where(torch.sigmoid(neg_snap[0])>0.5, 1, 0)
            color_pred = output['color']
            color_pred = torch.argmax(color_pred[0], dim = 0)
            # im = default_image_untransform(id_pred)
            # save_image(im, destination + "/" + str(counter) + "id_pred.png")
            im = default_image_untransform(workspace[0])
            save_image(im, destination + "/" + str(counter) + "workspace.png")
            # im = default_image_untransform(pos_pred_t)
            # save_image(im, destination + "/" + str(counter) + "pos_snap.png")
            # im = default_image_untransform(neg_pred_t)
            # save_image(im, destination + "/" + str(counter) + "neg_snap.png")
            # plt.imshow(pos_pred, cmap="gray", interpolation="nearest")
            # plt.clim(0, 1)
            # plt.colorbar()
            # plt.savefig(destination + "/" + str(counter) + "pos_snap_c.png")
            # plt.clf()
            # plt.imshow(neg_pred, cmap="gray", interpolation="nearest")
            # plt.clim(0, 1)
            # plt.colorbar()
            # plt.savefig(destination + "/" + str(counter) + "neg_snap_c.png")
            # plt.clf()
            # im = default_image_untransform(label[0,:,:,0])
            # save_image(im, destination + "/" + str(counter) + "id_label.png")
            # im = default_image_untransform(label[0,:,:,1])
            # save_image(im, destination + "/" + str(counter) + "pos_label.png")
            # im = default_image_untransform(label[0,:,:,2])
            # save_image(im, destination + "/" + str(counter) + "neg_label.png")
            im = default_image_untransform(label[0, :, :, 3])
            save_image(im, destination + "/" + str(counter) + "color_label.png")
            im = default_image_untransform(color_pred)
            save_image(im, destination + "/" + str(counter) + "color_pred.png")
            counter += 1
            if counter == 12:
                exit()

def main():
    config = PretrainResnetBackboneConfig.load_config("../../experiments/pretrainbackbone_resnet/settings.cfg")
    train_disassembly_behavior_cloning(config)

if __name__ == '__main__' :
    config = PretrainResnetBackboneConfig.load_config("../../experiments/pretrainbackbone_resnet/settings.cfg")
    # main()
    visualization(config, model_path="./checkpoint/Jan01_17-09-55_patillo/model_0050.pt")
