from collections import OrderedDict

import torch
import torch.nn as nn

from torch.distributions import Categorical

from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Tuple

from avarice.data import torch_to_numpy

from steadfast.config import Config
from steadfast.hierarchy import flatten_hierarchy
from steadfast.name_span import NameSpan

#from ltron.gym.components.cursor import CursorActionSpace

from ltron_torch.models.mlp import linear_stack

class DictAutoDecoder(nn.Module):
    def __init__(self, config, space):
        super().__init__()
        submodules = OrderedDict()
        for name, subspace in space.items():
            submodules[name] = AutoDecoder(config, subspace)
        self.submodules = nn.ModuleDict(submodules)
    
    def forward(self, x):
        out = {}
        for name, module in self.submodules.items():
            module_out, x = module(x)
            out[name] = module_out
        
        return out, x
    
    def sample_action(self, output, observation=None, info=None):
        return {
            name : submodule.sample_action(output[name])
            for name, submodule in self.submodules.items()
        }
    
    def log_prob(self, output, action):
        return sum(
            submodule.log_prob(output[name], action[name])
            for name, submodule in self.submodules.items()
        )
    
    def entropy(self, output):
        return sum(
            submodule.entropy(output[name])
            for name, submodule in self.submodules.items()
        )

class TupleAutoDecoder(nn.Module):
    def __init__(self, config, space):
        super().__init__()
        submodules = []
        for subspace in space:
            submodules.append(AutoDecoder(config, subspace))
        self.submodules = nn.ModuleList(submodules)
    
    def forward(self, x):
        out = []
        for i, submodule in enumerate(self.submodules):
            module_out, x = submodule(x)
            out.append(module_out)
        return out, x
    
    def sample_action(self, output, observation=None, info=None):
        return [
            submodule.sample_action(output[i])
            for i, submodule in enumerate(self.submodules)
        ]
    
    def log_prob(self, output, action):
        return sum(
            submodule.log_prob(output[i], action[i])
            for i, submodule in enumerate(self.submodules)
        )
    
    def entropy(self, output, observation, info, action):
        return sum(
            submodule.entropy(output[i])
            for i, submodule in enumerate(self.submodules)
        )

class DiscreteAutoDecoder(nn.Module):
    def __init__(self, config, space):
        super().__init__()
        self.config = config
        self.mlp = linear_stack(
            self.config.decoder_layers,
            self.config.channels,
            out_channels=space.n,
            nonlinearity=self.config.nonlinearity,
        )
        self.embedding = nn.Embedding(space.n, self.config.channels)
    
    def forward(self, x, sample=None):
        logits = self.mlp(x)
        if torch.any(~torch.isfinite(logits)):
            breakpoint()
        distribution = Categorical(logits=logits)
        if sample is None:
            sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        entropy = distribution.entropy()
        
        embedding = self.embedding(sample)
        x = x + embedding
        return sample, log_prob, entropy, x
    
    '''
    def sample_action(self, output, observation=None, info=None):
        return output['sample']
    
    def log_prob(self, output, action):
        return output['distribution'].log_prob(action)
    
    def entropy(self, output):
        return output['distribution'].entropy()
    '''

class MultiDiscreteAutoDecoder(TupleAutoDecoder):
    def __init__(self, config, space):
        super().__init__(config, Tuple(Discrete(n) for n in space.nvec))
    
    #def sample_action(self, output, observation, info):
    #    action = torch_to_numpy(output['sample'])
    #    breakpoint()
    #    return action
    
    def forward(self, x):
        o, x = super().forward(x, [{} for _ in self.submodules])
        out = {}
        out['logits'] = torch.stack([oo['logits'] for oo in o], dim=1)
        out['distribution'] = [oo['distribution'] for oo in o]
        out['sample'] = torch.stack([oo['sample'] for oo in o], dim=1)
        #out['logits'] = torch.stack(out['logits'], dim=1)
        #out['sample'] = torch.stack(out['sample'], dim=1)
        return out, x
    
    def sample_action(self, output, observation=None, info=None):
        return output['sample']
    
    def log_prob(self, output, action):
        return sum(
            distribution.log_prob(output['sample'][:,i])
            for i, distribution in enumerate(output['distribution'])
        )
    
    def entropy(self, output, observation, info, action):
        return sum(
            distribution.entropy()
            for distribution in output['distribution']
        )

'''
class CursorAutoDecoder(DictAutoDecoder):
    def __init__(self, config, space):
        super().__init__(config, space)
    
    def forward(self, x, snap_map=None):
        #out = super().forward(x, submodule_kwargs={name:None for name in self.submodules})
        breakpoint()
    
    def log_prob(self, output, action):
        breakpoint()
    
    def entropy(self, output, target):
        breakpoint()
    
    def observation_to_kwargs(self, observation, info, done, output):
        if 'snap_map' in observation:
            return {'snap_map':observation['snap_map']}
        else:
            return {}
'''

class AutoDecoderConfig(Config):
    channels = 768
    decoder_layers = 3
    nonlinearity = 'gelu'
    image_attention_channels = 16

def AutoDecoder(config, space):
    #if isinstance(space, CursorActionSpace):
    #    return CursorAutoDecoder(config, space)
    if isinstance(space, Dict):
        return DictAutoDecoder(config, space)
    elif isinstance(space, Tuple):
        return TupleAutoDecoder(config, space)
    elif isinstance(space, MultiDiscrete):
        return MultiDiscreteAutoDecoder(config, space)
    elif isinstance(space, Discrete):
        return DiscreteAutoDecoder(config, space)

class CriticDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stack = linear_stack(
            config.decoder_layers,
            config.channels,
            out_channels=1,
            nonlinearity=config.nonlinearity,
        )
    
    def forward(self, x):
        return self.stack(x).view(-1)

class AutoActorCriticDecoder(nn.Module):
    def __init__(self, config, space):
        super().__init__()
        self.actor = AutoDecoder(config, space)
        self.critic = CriticDecoder(config)
    
    def forward(self, x, actor_kwargs):
        a, _ = self.actor(x, **actor_kwargs)
        c = self.critic(x)
        return {'actor' : a, 'critic' : c}
    
    def sample_action(self, output, observation, info):
        return self.actor.sample_action(output['actor'], observation, info)
    
    def log_prob(self, output, action):
        return self.actor.log_prob(output['actor'], action)
    
    def entropy(self, output):
        return self.actor.entropy(output['actor'])
    
    def value(self, output):
        return output['critic']
    
    def observation_to_kwargs(self, observation, info, done, output):
        return {
            'actor_kwargs' : self.actor.observation_to_kwargs(
                observation, info, done, output)
        }
