from torch.utils.data import DataLoader

from ltron.dataset.info import get_split_shards
from ltron.dataset.webdataset import (
    get_episode_webdataset,
    get_episode_webdataset_from_shards,
)

def build_episode_loader(dataset, split, batch_size, workers, **kwargs):
    shards = get_split_shards(dataset, split)
    return build_episode_loader_from_shards(
        shards, batch_size, workers, **kwargs)

def build_episode_loader_from_shards(
    shards,
    batch_size,
    workers,
    **kwargs,
):
    dataset = get_episode_webdataset_from_shards(
        shards,
        batch_size=batch_size,
        **kwargs,
    )
    
    #loader = DataLoader(
    #    dataset, batch_size=None, num_workers=workers, collate_fn=lambda x : x)
    
    return dataset #loader
