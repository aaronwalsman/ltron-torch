import sys

import torch
import torch.nn as nn

import torchvision.transforms.functional as tF

from gymnasium.spaces import Discrete

from steadfast.config import Config

#default_image_mean = [0.485, 0.456, 0.406]
#default_image_std = [0.229, 0.224, 0.225]

class AutoEmbeddingConfig(Config):
    channels = 768
    tile_height = 16
    tile_width = 16

class DictEmbedding(nn.ModuleDict):
    def __init__(self, config, observation_space):
        super().__init__({
            name : AutoEmbedding(config, subspace)
            for name, subspace in observation_space.items()
            if 'equivalence' not in name
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

class DiscreteEmbedding(nn.Embedding):
    def __init__(self, config, observation_space):
        super().__init__(observation_space.n, config.channels)
    
    def observation_to_kwargs(self, o, i, d, model_output):
        return {'input' : torch.LongTensor(o).to(self.weight.device)}
    
    def forward(self, x):
        return super().forward(x)

class ImageSpaceEmbedding(nn.Module):
    def __init__(self, config, observation_space):
        super().__init__()
        self.config = config
        assert observation_space.height % config.tile_height == 0
        assert observation_space.width % config.tile_width == 0
        self.y_tiles = observation_space.height // config.tile_height
        self.x_tiles = observation_space.width // config.tile_width
        self.total_tiles = self.y_tiles * self.x_tiles
        self.tile_pixels = config.tile_height * config.tile_width
        
        self.in_channels = observation_space.channels * self.tile_pixels
        self.pre_norm = nn.LayerNorm(self.in_channels)
        self.linear = nn.Linear(self.in_channels, config.channels)
        self.post_norm = nn.LayerNorm(config.channels)
        #self.dropout = nn.Dropout(config.embedding_dropout)
        
        self.positional_encoding = nn.Parameter(
            torch.randn(self.total_tiles, 1, config.channels))
        
        #self.tile_conv = nn.Conv2d(
        #    in_channels=in_channels,
        #    out_channels=config.channels,
        #    kernel_size=(1,1),
        #    #kernel_size=(config.tile_height, config.tile_width),
        #    #stride=(config.tile_height, config.tile_width)
        #)
        #self.tile_embedding = nn.Embedding(self.total_tiles, config.channels)
        #self.embedding_norm = nn.LayerNorm(config.channels)

    def forward(self, x):
        #x = self.tile_conv(x)
        #b, c, h, w = x.shape
        #x = x.view(b, c, h*w).permute(2,0,1)
        
        #x = x + self.embedding_norm(self.tile_embedding.weight.view(h*w, 1, c))
        
        b, h, w, c = x.shape
        x = x.view(
            b,
            self.y_tiles,
            self.config.tile_height,
            self.x_tiles,
            self.config.tile_width,
            c,
        )
        x = x.permute(1,3,0,2,4,5)
        x = x.reshape(self.total_tiles, b, self.tile_pixels*c)
        
        x = self.pre_norm(x)
        x = self.linear(x)
        x = self.post_norm(x)
        x = x + self.positional_encoding
        
        return x
    
    def observation_to_kwargs(self, o, i, d, model_output):
        x = torch.FloatTensor(
        #    ((o / 255.) - default_image_mean) / default_image_std
            (o / 255.)).to(self.linear.weight.device)
        #x = x.permute(0,3,1,2)
        #).to(self.tile_conv.weight.device)
        #x = x.permute(0,3,1,2)
        #x = tF.to_tensor(o)
        
        
        return {'x' : x}

def AutoEmbedding(config, observation_space):
    this_module = sys.modules[__name__]
    auto_embedding_name = type(observation_space).__name__ + 'Embedding'
    EmbeddingClass = getattr(this_module, auto_embedding_name)
    return EmbeddingClass(config, observation_space)
