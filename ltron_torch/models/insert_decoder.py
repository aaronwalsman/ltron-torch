import numpy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from gymnasium.spaces import Discrete

from ltron.constants import (
    SHAPE_CLASS_LABELS,
    COLOR_CLASS_LABELS,
    NUM_SHAPE_CLASSES,
    NUM_COLOR_CLASSES,
)
from ltron_torch.models.auto_decoder import AutoDecoder

class InsertDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.shape_class_labels is None:
            num_shape_classes = NUM_SHAPE_CLASSES
        else:
            num_shape_classes = len(config.shape_class_labels) + 1
        if config.color_class_labels is None:
            num_color_classes = NUM_COLOR_CLASSES
        else:
            num_color_classes = len(config.color_class_labels) + 1
        self.shape_decoder = AutoDecoder(config, Discrete(num_shape_classes))
        self.color_decoder = AutoDecoder(config, Discrete(num_color_classes))
    
    def forward(self,
        x, sample=None, sample_max=False, shape_eq=None, color_eq=None
    ):
        out_sample = []
        log_prob = 0.
        entropy = 0.
        
        shape_sample = None if sample is None else sample[:,0]
        #s,lp,e,_ = self.shape_decoder( # TEMP
        s,lp,e,x = self.shape_decoder(
            x,
            sample=shape_sample,
            sample_max=sample_max,
            equivalence=shape_eq,
            equivalence_dropout=0.,
        )
        out_sample.append(s)
        log_prob = log_prob + lp
        entropy = entropy + e
        
        color_sample = None if sample is None else sample[:,1]
        s,lp,e,x = self.color_decoder(
            x,
            sample=color_sample,
            sample_max=sample_max,
            equivalence=color_eq,
            equivalence_dropout=0.,
        )
        out_sample.append(s)
        log_prob = log_prob + lp
        entropy = entropy + e
        
        out_sample = torch.stack(out_sample, dim=1)
        
        return out_sample, log_prob, entropy, x
