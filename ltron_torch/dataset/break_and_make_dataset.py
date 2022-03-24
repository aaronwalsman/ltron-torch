import copy

import numpy

from ltron.hierarchy import index_hierarchy, concatenate_numpy_hierarchies

from ltron_torch.dataset.episode_dataset import (
    EpisodeDatasetConfig, EpisodeDataset)

class BreakAndMakeDatasetConfig(EpisodeDatasetConfig):
    pass

class BreakAndMakeDataset(EpisodeDataset):
    def __getitem__(self, i):
        sequence = super().__getitem__(i)
        
        # fix cursor observations
        # why is this happening?  can we fix the next round of episodes we
        # produce to take this out?
        if isinstance(
            sequence['observations']['table_cursor']['position'], list):
            sequence['observations']['table_cursor']['position'] = (
                numpy.stack((
                    sequence['observations']['table_cursor']['position'][0],
                    sequence['observations']['table_cursor']['position'][1],
                ), axis=-1))
        
        if isinstance(
            sequence['observations']['hand_cursor']['position'], list):
            sequence['observations']['hand_cursor']['position'] = (
                numpy.stack((
                    sequence['observations']['hand_cursor']['position'][0],
                    sequence['observations']['hand_cursor']['position'][1],
                ), axis=-1))
        
        return sequence

class BreakOnlyDataset(BreakAndMakeDataset):
    def __getitem__(self, i):
        data = super().__getitem__(i)
        
        i = numpy.where(data['actions']['phase'])[0]
        if len(i):
            data = index_hierarchy(data, slice(0, i[0]+1))
        
        return data
