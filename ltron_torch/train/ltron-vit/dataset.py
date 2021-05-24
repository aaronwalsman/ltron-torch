import torch
from torch.utils.data import Dataset
import numpy as np
import glob

KEYS = ["images", "categories", "image_positions"]


class LTronPatchDataset(Dataset):

    def __init__(self, train=True, root='data/'):
        super().__init__()
        path = root + 'train/*.torch' if train else root + 'test/*.torch'
        self.datapoints = glob.glob(path)
    
    def __len__(self):
        return len(self.datapoints)
    
    def __getitem__(self, index: int):
        d = torch.load(self.datapoints[index])
        return d

