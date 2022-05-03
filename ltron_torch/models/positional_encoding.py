import torch
from torch.nn import Module

from ltron_torch.models.parameter import NoWeightDecayParameter

class LearnedPositionalEncoding(Module):
    def __init__(self, channels, max_length, init='zero'):
        super(LearnedPositionalEncoding, self).__init__()
        if init == 'zero':
            encoding = torch.zeros(max_length, channels)
        if init == 'normal':
            encoding = torch.zeros(max_length, channels)
            encoding.normal_(mean=0., std=0.02)
        
        self.encoding = NoWeightDecayParameter(encoding)
    
    def forward(self, i):
        s, b = i.shape
        c = self.encoding.shape[-1]
        return self.encoding[i.reshape(-1)].view(s,b,c)
