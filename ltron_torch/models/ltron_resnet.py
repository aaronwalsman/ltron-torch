import numpy

import torch
import torch.nn as nn
from torch.distributions import Categorical

from ltron_torch.models.simple_fcn import named_resnet_fcn
from ltron_torch.models.mlp import linear_stack
from ltron_torch.models.equivalence import (
    equivalent_outcome_categorical,
)

default_image_mean = torch.FloatTensor([0.485, 0.456, 0.406])
default_image_std = torch.FloatTensor([0.229, 0.224, 0.225])

class LtronResNet(nn.Module):
    def __init__(self, config, observation_space, action_space):
        super().__init__()
        value_head = linear_stack(
            3,
            2048,
            hidden_channels=256,
            out_channels=1,
        )
        self.fcn = named_resnet_fcn(
            'resnet50',
            decoder_channels=256,
            global_heads=value_head,
        )
        
        self.click_decoder = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256,64,kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64,32,kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32,2,kernel_size=3,padding=1)
            #nn.Conv2d(256,32,kernel_size=3,padding=1),
        )
        
        self.observation_space = observation_space
        self.action_space = action_space
    
    def observation_to_kwargs(self, observation, info, done, model_output):
        x = torch.FloatTensor(observation['image']) / 255.
        x = (x - default_image_mean) / default_image_std
        x = x.permute(0,3,1,2)
        device = next(iter(self.parameters())).device
        x = x.to(device)
        
        pos_eq = observation['interface']['pos_equivalence']
        pos_eq = torch.LongTensor(pos_eq).to(device)
        neg_eq = observation['interface']['neg_equivalence']
        neg_eq = torch.LongTensor(neg_eq).to(device)
        
        return {
            'x' : x,
            'pos_eq' : pos_eq,
            'neg_eq' : neg_eq,
        }
    
    def forward(self, x, pos_eq, neg_eq):
        value, x = self.fcn(x)
        x = self.click_decoder(x)
        
        #b, c, h, w = x.shape
        ## what shape is x?  64x64 probably?
        #x = x.view(b, 2, 4, 4, h, w).permute(0, 1, 4, 2, 5, 3).reshape(
        #    b, 2, h*4, w*4)
        
        return {
            'value' : value,
            'x' : x,
            'pos_eq' : pos_eq,
            'neg_eq' : neg_eq,
        }
    
    def sample_log_prob(self, output, sample=None):
        x = output['x']
        b, c, h, w = x.shape
        
        logits = x.reshape(b, -1)
        distribution = Categorical(logits=logits)
        if sample is None:
            sample = distribution.sample()
        
        button, click_y, click_x = numpy.unravel_index(
            sample.cpu().numpy(), (2,256,256))
        
        device = x.device
        button_eq = torch.zeros((b, 2, 256, 256), dtype=torch.long).to(device)
        button_eq[:,1] = 1
        button_eq_distribution = equivalent_outcome_categorical(
            logits, button_eq.view(b, -1))
        button_islands = button_eq.view(b,-1)[range(b), sample]
        
        eq = torch.stack((output['neg_eq'], output['pos_eq']), dim=1)
        click_logits = x[range(b), button]
        click_eq = eq[range(b), button]
        eq_distribution = equivalent_outcome_categorical(
            click_logits.view(b,-1), click_eq.view(b,-1), dropout=0.)
        click_islands = click_eq[range(b), click_y, click_x]
        
        entropy = eq_distribution.entropy() + button_eq_distribution.entropy()
        log_prob = (
            eq_distribution.log_prob(click_islands) +
            button_eq_distribution.log_prob(button_islands)
        )
        
        return {
            'log_prob' : log_prob,
            'sample' : sample,
            'entropy' : entropy,
            'action' : {
                'interface' : {
                    'cursor' : {
                        'button' : button,
                        'click' : numpy.stack((click_y, click_x), axis=1),
                        'release' : numpy.zeros((b,2), dtype=numpy.int64),
                    },
                    'primitives': {
                        'mode' : numpy.ones((32,), dtype=numpy.int64),
                        'viewpoint' : numpy.zeros((32,), dtype=numpy.int64),
                        'remove' : numpy.ones((32,), dtype=numpy.int64),
                    }
                }
            },
            'click_logits' : click_logits,
            'release_logits' : torch.zeros_like(click_logits),
        }
    
    def sample_output_to_log_prob(self, sample_output):
        return sample_output['log_prob']
    
    def sample_output_to_entropy(self, sample_output):
        return sample_output['entropy']
    
    def sample_output_to_sample(self, sample_output):
        return sample_output['sample']
    
    def sample_output_to_action(self, sample_output):
        return sample_output['action']
    
    def value(self, output):
        return output['value'].view(-1)
    
    #def loss(self, x, y, seq_pad):
    #    loss = torch.sum(-torch.log_softmax(x, dim=-1) * y, dim=-1)
