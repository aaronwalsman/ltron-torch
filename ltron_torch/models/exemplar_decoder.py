import torch
from torch.nn import Module, Linear, Parameter

from ltron.config import Config

class ExemplarDecoderConfig(Config):
    encoder_channels = 768
    exemplars = 256
    exemplar_dimensions = 2
    exemplar_min = 0
    exemplar_max = 255
    exemplar_channels = 1

class ExemplarDecoder(Module):
    def __init__(self, config):
        self.config = config
        self.linear = Linear(config.encoder_channels, config.exemplars)
        self.exemplars = Parameter(
            torch.rand((config.exemplars, config.exemplar_dimensions)))
        
        self.t = 1. / config.exemplars ** 0.5
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.softmax(x*self.t, dim=-1)
        x = torch.einsum('sbc,cn->sbn', x, self.exemplars)
        exemplar_scale = self.config.exemplar_max - self.config.exemplar_min
        x = x * exemplar_scale + self.config.exemplar_min
        
        return x
        
