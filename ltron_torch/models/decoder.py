import torch
import torch.nn as nn
from torch.distributions import Categorical
from steadfast.config import Config

from ltron_torch.models.mlp import linear_stack
from ltron_torch.models.equivalence import equivalent_outcome_categorical

class DecoderConfig(Config):
    channels = 768
    decoder_layers = 3
    nonlinearity = 'gelu'
    image_attention_channels = 16
    sigmoid_screen_attention = False
    screen_equivalence = True

class DiscreteDecoder(nn.Module):
    def __init__(self, config, num_classes): #, sample_offset=0):
        super().__init__()
        self.config = config
        self.mlp = linear_stack(
            self.config.decoder_layers,
            self.config.channels,
            out_channels=num_classes,
            first_norm=True,
            nonlinearity=self.config.nonlinearity,
            Norm=nn.LayerNorm,
        )
        self.embedding = nn.Embedding(num_classes, self.config.channels)
        #self.embedding_norm = nn.LayerNorm(self.config.channels)
        #self.sample_offset = sample_offset

    def forward(self,
        x,
        sample=None,
        equivalence=None,
        equivalence_dropout=0.5,
        sample_max=False,
    ):
        logits = self.mlp(x)
        if torch.any(~torch.isfinite(logits)):
            breakpoint()

        distribution = Categorical(logits=logits)
        if sample is None:
            if sample_max:
                physical_index = torch.argmax(distribution.probs, dim=-1)
                sample = physical_index
            else:
                physical_index = distribution.sample()
                sample = physical_index # + self.sample_offset
        else:
            physical_index = sample # - self.sample_offset

        if equivalence is not None:
            eq_distribution = equivalent_outcome_categorical(
                logits, equivalence, dropout=equivalence_dropout)
            b = logits.shape[0]
            eq_sample = equivalence[range(b),sample]
            log_prob = eq_distribution.log_prob(eq_sample)
            entropy = eq_distribution.entropy()
        else:
            log_prob = distribution.log_prob(physical_index)
            entropy = distribution.entropy()

        embedding = self.embedding(physical_index)
        #x = x + self.embedding_norm(embedding)
        x = x + embedding
        return sample, log_prob, entropy, x, logits

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

class ConstantDecoder(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.config = config
        self.index = index

    def forward(self, x, sample=None):
        b = x.shape[0]
        device = x.device
        if sample is None:
            sample = torch.full((b,), self.index, device=device)

        return sample, 0., 0., x, torch.zeros(b, 1, device=device)
