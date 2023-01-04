from collections import OrderedDict

import torch
import torch.nn as nn
#from torch.nn import Module, ModuleList, ModuleDict

from torch.distributions import Categorical

from gymnasium.spaces import Discrete, MultiDiscrete, Dict

from avarice.data import torch_to_numpy

from steadfast.config import Config
from steadfast.hierarchy import flatten_hierarchy
from steadfast.name_span import NameSpan

from ltron_torch.models.mlp import linear_stack

'''
We have an action space made up of a bunch of individual components.  We need
to decode a value for each component.

In order to model the joint distribution of these variables, we want a
sequential decoding of individual conditional distributions:
p(a[0]) -> sample a0 -> p(a[1]|a0) -> sample a1 -> p(a[2]|a0,a1)

Where a[i] is the individual action components.

The simplest version (UnsharedMLP) is:
Each component is it's own MLP with an input x and it's own output logits.
An action component a[i] is sampled from these logits, then passed to an
embedding which is then added to x to form the input for the next MLP.

The next slightly more complex version (SharedMLP) is:
There is a single shared MLP with output logits for every conditional
distribution.  If there are three action components with 3, 4 and 7
sub-actions each the shared MLP will have 3+4+7=14 outputs.  Then we do
the same thing as before and decode them one by one, feeding each into a
shared embedding before adding them to x as input for the next pass.  This
would let you decode in arbitrary order, and you'd have to devise ways to
supervise this when doing imitation learning.

The next more complex version (SharedTransformer) is:
Just use a small conditional (GPT-style) transformer instead of SharedMLP
with a shared vocabulary space.

All of these methods will run a fixed number of sequential steps.  I'd be
worried that the transformer is overkill and will be slow, but if it's small
enough, maybe it'd be fine.  Although, you'd also have to run it sequentially
with gradients flowing all the way through, which I haven't done before.
Let's restrict ourselves to the first two for now, and only push up to the
transformer if we can convince ourselves that it won't work otherwise.

The final much more involved version would be to make each conditional
distribution decode step part of the main sensory processing transformer.
The problem is that we'd then have to decode multiple full transformer
steps for each output action, and would have to use the same size of
transformer for everything.  This also commits us pretty heavily to the
transformer architecture.  All the previous methods can be used with
anything, whereas this requires you to be in transformer land all the
way through.  Although I guess you could take the output of whatever
network and use it to seed the input token of a transformer decoder, but at
that point it feels like we are back at SharedTransformer.  The nice thing
about this though is that it allows you to just do regular GPT-style training
where you predict the next token all the way through, and get some benefit
from training it to predict future observations as well.  This is all way too
big for what I have at the moment though, and there are a lot of other
considerations.  For example, what does it look like to mix the "predict
the future" loss with some RL/policy gradient loss?  Seems challenging.
If you have direct supervision everything lines up nicely, but if not...
less nice.  I think I have to start with the simple MLP decoders for now.
This is the easiest thing to spin up, and for this problem, it should just
work.

One last consideration is where the input to the decoder comes from.
The simplest thing is to just have a single decoder token from the main
transformer.  That plays nicely with other backbones too, all you need to
do is provide a single feature vector as input to the decoder.  For a second
I thought about making a single vector by doing attention over the outputs of
the entire history of tokens so far... but guess what that is?  It's the same
as a single decoder token.  Sheesh.  Ok, so single decoder token it is.
'''

class UnsharedMLPAutoDecoderConfig(Config):
    channels = 768
    layers = 3
    nonlinearity = 'gelu'

class AutoDecoderHead(nn.Module):
    def __init__(self, config, num_outputs):
        super().__init__()
        self.config = config
        self.mlp = linear_stack(
            self.config.layers,
            self.config.channels,
            hidden_channels=self.config.channels,
            out_channels=num_outputs,
            nonlinearity=self.config.nonlinearity,
        )
        self.embedding = nn.Embedding(num_outputs, self.config.channels)

class UnsharedMLPAutoDecoder(nn.Module):
    def __init__(self,
        config,
        action_space,
        build_value_head = False,
    ):
        super().__init__()
        self.config = config
        self.action_space = action_space
        self.conditional_mlps = self.build_nested_conditional_mlp(action_space)
        
        if build_value_head:
            self.value_head = linear_stack(
                self.config.layers,
                self.config.channels,
                hidden_channels=self.config.channels,
                out_channels=1,
                nonlinearity=self.config.nonlinearity,
            )
    
    def build_nested_conditional_mlp(self, space):
        if isinstance(space, Discrete):
            return AutoDecoderHead(self.config, space.n)
        elif isinstance(space, MultiDiscrete):
            return nn.ModuleList([
                AutoDecoderHead(self.config, n)
                for n in space.nvec
            ])
        elif isinstance(space, Dict):
            submodules = OrderedDict()
            for name, subspace in space.items():
                submodules[name] = self.build_nested_conditional_mlp(subspace)
            return nn.ModuleDict(submodules)
        elif isinstance(space, Tuple):
            return nn.ModuleList([
                self.build_nested_conditional_mlp(subspace)
                for subspace in space
            ])
        else:
            raise ValueError(
                'AutoDecoder only accepts hierarchies of Dicts, Tuples,'
                'Discrete and MultiDiscrete action spaces'
            )
    
    def compute_single_output(self, x, module):
        logits = module.mlp(x)
        distribution = Categorical(logits=logits)
        sample = distribution.sample()
        log_prob = distribution.log_prob(sample)
        embedding = module.embedding(sample)
        x = x + embedding
        return logits, distribution, sample, log_prob, x #embedding, x
    
    def compute_nested_output(self, x, module):
        if isinstance(module, nn.ModuleDict):
            logits = OrderedDict()
            distributions = OrderedDict()
            samples = OrderedDict()
            log_prob = 0.
            #embeddings = OrderedDict()
            for name, sub_module in module.items():
                #l,d,s,p,e,x = self.compute_nested_output(x, sub_module)
                l,d,s,p,x = self.compute_nested_output(x, sub_module)
                logits[name] = l
                distributions[name] = d
                samples[name] = s
                log_prob = log_prob + p
                #embeddings[name] = e
            return logits, distributions, samples, log_prob, x #embeddings, x
        elif isinstance(module, nn.ModuleList):
            logits = []
            distributions = []
            samples = []
            log_prob = 0.
            #embeddings = []
            for sub_module in module:
                #l,d,s,p,e,x = self.compute_nested_output(x, sub_module)
                l,d,s,p,x = self.compute_nested_output(x, sub_module)
                logits.append(l)
                distributions.append(d)
                samples.append(s)
                log_prob = log_prob + p
                #embeddings.append(e)
            return logits, distributions, samples, log_prob, x, #embeddings, x
        else:
            return self.compute_single_output(x, module)
    
    def forward(self, x):
        b, c = x.shape
        
        out = {}
        #l,d,s,p,e,_ = self.compute_nested_output(x, self.conditional_mlps)
        l,d,s,p,_ = self.compute_nested_output(x, self.conditional_mlps)
        out['logits'] = l
        out['distributions'] = d
        out['samples'] = s
        out['log_prob'] = p
        #out['embeddings'] = e
        
        if hasattr(self, 'value_head'):
            out['value'] = self.value_head(x)
        
        return out
    
    def sample_action(self, output, observation, info):
        action = torch_to_numpy(output['samples'])
        return action
    
    def log_probs(self, output, action):
        return output['log_prob']

class AutoDecoderConfig(UnsharedMLPAutoDecoderConfig):
    mode = 'UnsharedMLPAutoDecoder'

def AutoDecoder(config, action_space, *args, **kwargs):
    if config.mode == 'UnsharedMLPAutoDecoder':
        return UnsharedMLPAutoDecoder(config, action_space, *args, **kwargs)
    else:
        raise NotImplementedError
