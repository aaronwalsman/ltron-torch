import torch
from torch.utils.data import Dataset
import numpy as np
import glob

KEYS = ["images", "categories", "image_positions"]


class LTronPatchDataset(Dataset):

    def __init__(self, train=True, root='data/'):
        path = root + 'train/*.npy' if train else root + 'test/*.npy'
        self.datapoints = [np.load(f, allow_pickle=True) for f in glob.glob(path)]
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, index: int):
        a = self.datapoints[index]
        return {name: a.item().get(name) for name in KEYS}

