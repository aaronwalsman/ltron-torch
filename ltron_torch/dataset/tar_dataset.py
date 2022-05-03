import io
import tarfile

import numpy

from torch.utils.data import Dataset, DataLoader

from gym.vector.async_vector_env import AsyncVectorEnv

from ltron.config import Config
from ltron.hierarchy import index_hierarchy

from ltron_torch.dataset.collate import pad_stack_collate

class TarDataset(Dataset):
    def __init__(self, tar_paths):
        self.tar_paths = tar_paths
        self.tar_files = None
        _, self.names = get_tarfiles_and_names(self.tar_paths)
    
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, i):
        if self.tar_files is None:
            self.tar_files, _ = get_tarfiles_and_names(self.tar_paths)
        
        tar_path, name = self.names[i]
        #bytestream = io.BytesIO(self.zipfile.open(name).read())
        data = self.tar_files[tar_path].extractfile(name)
        data = numpy.load(data, allow_pickle=True)
        data = data['episode'].item()
        
        return data

def get_tarfiles_and_names(tar_paths):
    tar_files = {tp:tarfile.open(tp, 'r') for tp in tar_paths}
    names = []
    for tar_path, tar_file in tar_files.items():
        names.extend([(tar_path, name) for name in tar_file.getnames()])
    
    return tar_files, names
    

def build_episode_loader(dataset, batch_size, workers, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=pad_stack_collate,
        shuffle=shuffle,
    )
    
    return loader
