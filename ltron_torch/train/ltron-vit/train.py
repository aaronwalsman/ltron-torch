import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import atexit
import sys
import os
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from vit import LTron_ViT
from dataset import LTronPatchDataset


def train():

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    WEIGHT_PATH = 'weights/ltron_vit.pt'
    NUM_EPOCHS = 100
    LR = 1e-4
    WEIGHT_DECAY = 0.0
    OPTIMIZER = "Adam"
    PATCH_SIZE = 8*8
    HIDDEN_DIM = 256
    BATCH_SIZE = 8

    START_EPOCH = 0

    # Initialize tensorboard loger
    logname = "ltron_vit-lr=%f-wd=%f-optim=%s-patch_size=%d-hidden_dim=%d-batch_size=%d" % \
        (LR, WEIGHT_DECAY, OPTIMIZER, PATCH_SIZE, HIDDEN_DIM, BATCH_SIZE)
    logname = "%s-%s" % (logname, sys.argv[1]) if len(sys.argv) > 1 else logname
    writer = SummaryWriter(log_dir="logs/%s/" % logname)

    # Initialize data loader
    ### TODO There is a bug that only lets me run with with num_workers=0 (proabbly in the dataset class)
    ### that needs to be fixed
    train_dataset = LTronPatchDataset(train=True)#, root='../../envs/fork/')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_dataset = LTronPatchDataset(train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    model = LTron_ViT(lr=LR, weight_decay=WEIGHT_DECAY, optimizer_type=OPTIMIZER, device=DEVICE).to(DEVICE)
    if os.path.exists(WEIGHT_PATH):
        print("Loading Weights For %s" % WEIGHT_PATH)
        checkpoint = torch.load(WEIGHT_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["weights"])
        START_EPOCH = checkpoint["epoch"] +1
    print("Starting training at epoch %d" % START_EPOCH)

    for e in tqdm(range(NUM_EPOCHS)):
        train_losses = model.train_epoch(train_dataloader)
        writer.add_scalar("Train/Classification Loss", train_losses["classification"], e)
        writer.add_scalar("Train/Classification Error", train_losses["classification_error"], e)

        test_losses = model.test_epoch(test_dataloader)
        writer.add_scalar("Test/Classification Loss", test_losses["classification"], e)
        writer.add_scalar("Test/Classification Error", test_losses["classification_error"], e)

        torch.save({"weights": model.state_dict(), "epoch": e}, WEIGHT_PATH)


if __name__ == '__main__':
    train()
