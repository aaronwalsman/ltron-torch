import sys

import torch
import torch.nn as nn

from gymnasium.spaces import Discrete

from steadfast.config import Config

default_image_mean = [0.485, 0.456, 0.406]
default_image_std = [0.229, 0.224, 0.225]

'''
class DictEmbedding(nn.ModuleDict):
    def __init__(self, config, observation_space):
        super().__init__({
            name : AutoEmbedding(config, subspace)
            for name, subspace in observation_space.items()
            if not isinstance(subspace, IgnoreSpace)
        })
    
    def forward(self, x):
        return {name : module(**x[name]) for name, module in self.items()}
    
    def observation_to_kwargs(self, obs, info, done, model_output):
        if not isinstance(info, dict):
            breakpoint()
        return {'x' : {
            name : module.observation_to_kwargs(
                obs[name], info.get(name, {}), done, None)
            for name, module in self.items()
        }}

class TupleEmbedding(nn.ModuleList):
    def __init__(self, config, observation_space):
        super().__init__([
            AutoEmbedding(config, subspace)
            for i, subspace in enumerate(observation_space)
            if not isinstance(subspace, IgnoreSpace)
        ])
       
    def forward(self, x):
        return [module(**x[i]) for i, module in enumerate(self)]
    
    def observation_to_kwargs(self, observation, info, done, model_output):
        model_kwargs = []
        for i, module in enumerate(self):
            try:
                module_info = info[i]
            except (KeyError, IndexError):
                module_info = {}
            model_kwargs.append(module.observation_to_kwargs(
                observation[i], module_info, done, None))
        return {'x' : model_kwargs}
'''
class DiscreteEmbedding(nn.Embedding):
    def __init__(self, config, observation_space):
        super().__init__(observation_space.n, config.channels)
    
    def observation_to_kwargs(self, o, i, d, model_output):
        return {'input' : torch.LongTensor(o).to(self.weight.device)}
    
    def forward(self, x):
        return super().forward(x)

class ImageSpaceEmbedding(nn.Module):
    def __init__(self, observation_space, tile_width, tile_height, channels):
        super().__init__()
        assert observation_space.height % tile_height == 0
        assert observation_space.width % tile_width == 0
        self.y_tiles = observation_space.height // tile_height
        self.x_tiles = observation_space.width // tile_width
        self.total_tiles = self.y_tiles * self.x_tiles

        self.tile_conv = nn.Conv2d(
            in_channels=observation_space.channels,
            out_channels=channels,
            kernel_size=(tile_height, tile_width),
            stride=(tile_height, tile_width)
        )
        self.tile_embedding = nn.Embedding(self.total_tiles, channels)

    def forward(self, x):
        x = self.tile_conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c, h*w).permute(2,0,1)
        
        x = x + self.tile_embedding.weight.view(h*w, 1, c)
        
        return x
    
    def observation_to_kwargs(self, o, i, d, model_output):
        x = torch.FloatTensor(
            ((o / 255.) - default_image_mean) / default_image_std
        ).to(self.tile_conv.weight.device)
        x = x.permute(0,3,1,2)
        
        return {'x' : x}
'''
class AutoEmbeddingConfig(Config):
    channels = 768
    tile_height = 16
    tile_width = 16

def AutoEmbedding(config, observation_space):
    if isinstance(observation_space, IgnoreSpace):
        return None
    
    this_module = sys.modules[__name__]
    EmbeddingClass = getattr(
        this_module, type(observation_space).__name__ + 'Embedding')
    return EmbeddingClass(config, observation_space)
'''
