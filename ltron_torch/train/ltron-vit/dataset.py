from torch.utils.data import Dataset
import numpy as np
import glob
import torch

KEYS = ["images", "categories", "image_positions"]


class LTronPatchDataset(Dataset):

    def __init__(self, train=True, root='data/'):
        super().__init__()
        path = root + 'train/data-images*.npy' if train else root + 'test/data-images*.npy'
        self.image_datapoints = glob.glob(path)
        self.image_datapoints.sort()
        self.categories_datapoints = [p.replace("data-images", "data-categories") for p in self.image_datapoints]
        self.image_positions_datapoints = [p.replace("data-images", "data-image-positions") for p in self.image_datapoints]
    
    def __len__(self):
        return len(self.image_datapoints)
    
    def __getitem__(self, index: int):
        d = {
            "images": np.load(self.image_datapoints[index]),
            "categories": np.load(self.categories_datapoints[index]),
            "image_positions": np.load(self.image_positions_datapoints[index]),
        }
        return d

