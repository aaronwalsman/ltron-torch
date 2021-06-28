import torch

from ltron.hierarchy import map_hierarchies

class MultiHead(torch.nn.Module):
    def __init__(self, modules):
        super(MultiHead, self).__init__()
        self.modules = modules
    
    def forward(x):
        def multi_forward(module):
            return module(x)
        return map_hierarchies(multi_forward, self.modules)
