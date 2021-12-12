from torch.nn import Module

from ltron_torch.model.parameter import NoWeightDecayParameter

class LearnedPositionalEncoding(Module):
    def __init__(self, length, channels):
        super(LearnedPositionalEncoding, self).__init__()
        self.encoding = NoWeightDecayParameter(torch.zeros(length, channels))
    
    def forward(self, i):
        s, b = i.shape
        return self.encoding[i]
