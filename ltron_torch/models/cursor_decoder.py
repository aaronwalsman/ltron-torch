import torch
import torch.nn as nn
from torch.distributions import Categorical

from gymnasium.spaces import Discrete

from ltron_torch.models.mlp import linear_stack, conv2d_stack
from ltron_torch.models.equivalence import equivalent_outcome_categorical
from ltron_torch.models.auto_decoder import AutoDecoder

class CursorDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.button_decoder = AutoDecoder(config, Discrete(2))
        self.click_decoder = ScreenDecoder(config)
        self.release_decoder = ScreenDecoder(config)
    
    def forward(self,
        x,
        image_x,
        click_eq,
        release_eq,
        do_release,
        sample=None,
    ):
        out_sample = {}
        log_prob = 0.
        entropy = 0.
        
        button_sample = None if sample is None else sample['button']
        s,lp,e,x = self.button_decoder(x, sample=button_sample)
        out_sample['button'] = s
        log_prob = log_prob + lp
        entropy = entropy + e
        
        click_sample = None if sample is None else sample['click']
        s,lp,e,x = self.click_decoder(x, image_x, click_eq)
        out_sample['click'] = s
        log_prob = log_prob + lp
        entropy = entropy + e
        
        release_sample = None if sample is None else sample['release']
        s,lp,e,rx = self.release_decoder(x, image_x, release_eq)
        out_sample['release'] = s
        x = x * ~do_release.view(-1,1) + rx * do_release.view(-1,1)
        log_prob = log_prob + lp * do_release
        entropy = entropy + e * do_release
        
        return out_sample, log_prob, entropy, x
    '''
    def forward(self, decode_x, image_x, click_eq, release_eq):
        button_out, decode_x = self.button_decoder(decode_x)
        click_out, decode_x = self.click_decoder(
            decode_x, image_x, click_eq)
        release_out, decode_x = self.release_decoder(
            decode_x, image_x, release_eq)
        
        return {
            'button' : button_out,
            'click' : click_out,
            'release' : release_out,
        }
    '''
    '''
    def log_prob(self, output, action, include_release=True):
        b = self.button_decoder.log_prob(output['button'], action['button'])
        c = self.click_decoder.log_prob(output['click'], action['click'])
        r = self.release_decoder.log_prob(
            output['release'], action['release']) * include_release
        return b + c + r
    
    def entropy(self, output, include_release=True):
        b = self.button_decoder.entropy(output['button'])
        c = self.click_decoder.entropy(output['click'])
        r = self.release_decoder.entropy(output['release']) * include_release
        return b + c + r
    '''

class ScreenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        th = config.tile_height
        tw = config.tile_width
        self.config = config
        self.q_head = linear_stack(
            self.config.decoder_layers,
            self.config.channels,
            out_channels=config.image_attention_channels,
            nonlinearity=config.nonlinearity,
        )
        self.k_head = conv2d_stack(
            self.config.decoder_layers,
            self.config.channels,
            out_channels=config.image_attention_channels*th*tw,
            nonlinearity=config.nonlinearity,
        )
        self.content_linear = nn.Linear(
            config.image_attention_channels, config.channels)
        self.y_embedding = nn.Embedding(
            config.image_height, config.channels)
        self.x_embedding = nn.Embedding(
            config.image_width, config.channels)
        self.norm = nn.LayerNorm(config.channels)
    
    #def forward(self, decode_x, image_x, equivalence=None):
    #    return (decode_x, image_x, equivalence)
    #
    #def log_prob_entropy(self, model_output, action):
        #decode_x, image_x, equivalence = model_output
    
    def forward(self,
        x,
        image_x,
        equivalence,
        sample=None,
    ):
        
        b,c,h,w = image_x.shape
        th, tw = self.config.tile_height, self.config.tile_width
        ac = self.config.image_attention_channels
        ih = self.config.image_height
        iw = self.config.image_width
        
        q = self.q_head(x)
        k = self.k_head(image_x)
        k = k.view(b,ac,th,tw,h,w).permute(0,1,4,2,5,3).reshape(b,ac,ih,iw)
        
        qk = torch.einsum('bc,bchw->bhw', q, k)
        
        distribution = Categorical(logits=qk.view(b,ih*iw))
        eq_distribution = equivalent_outcome_categorical(qk, equivalence)
        
        if sample is None:
            flat_sample = distribution.sample()
            sample_y = torch.div(flat_sample, iw, rounding_mode='floor')
            sample_x = flat_sample % iw
            sample = torch.stack((sample_y, sample_x), dim=-1)
        else:
            sample_y = sample[:,0]
            sample_x = sample[:,1]
        
        eq_sample = equivalence[range(b),sample_y,sample_x]
        log_prob = eq_distribution.log_prob(eq_sample)
        entropy = eq_distribution.entropy()
        
        x_pe = self.x_embedding(sample_x)
        y_pe = self.y_embedding(sample_y)
        content_x = self.content_linear(k[range(b),:,sample_y,sample_x])
        
        x = x + self.norm(content_x + x_pe + y_pe)
        
        #out = {}
        #out['sample'] = yx
        #out['equivalence'] = equivalence
        #out['distribution'] = eq_distribution
        
        return sample, log_prob, entropy, x
    
    def log_prob(self, output, action):
        b = output['equivalence'].shape[0]
        eq_index = output['equivalence'][range(b),action[:,0],action[:,1]]
        return output['distribution'].log_prob(eq_index)
    
    def entropy(self, output):
        return output['distribution'].entropy()
