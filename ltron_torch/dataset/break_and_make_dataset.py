import numpy

from ltron.hierarchy import index_hierarchy

from ltron_torch.dataset.episode_dataset import (
    EpisodeDatasetConfig, EpisodeDataset)

class BreakAndMakeDatasetConfig(EpisodeDatasetConfig):
    pass

class BreakAndMakeDataset(EpisodeDataset):
    pass

class BreakOnlyDataset(EpisodeDataset):
    def __getitem__(self, i):
        data = super().__getitem__(i)
        
        i = numpy.where(data['actions']['phase'])[0]
        if len(i):
            data = index_hierarchy(data, slice(0, i[0]+1))
        
        return data
