import torch
import torch.nn as nn
from torch.distributions import Categorical

from gymnasium.spaces import Discrete

from ltron_torch.models.mlp import linear_stack, conv2d_stack
from ltron_torch.models.equivalence import (
    equivalent_outcome_categorical,
    avg_equivalent_logprob,
)
from ltron_torch.models.decoder import DiscreteDecoder

class CursorDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.button_decoder = DiscreteDecoder(config, 2)
        self.click_decoder = ScreenDecoder(config)
        self.release_decoder = ScreenDecoder(config)
        self.forward_passes = 0
    
    def forward(self,
        x,
        image_x,
        pos_snap_eq,
        neg_snap_eq,
        do_release,
        sample=None,
    ):
        out_sample = {}
        log_prob = 0.
        entropy = 0.
        logits = {}
        
        button_sample = None if sample is None else sample['button']
        s,lp,e,x,b_logits = self.button_decoder(x, sample=button_sample)
        out_sample['button'] = s
        log_prob = log_prob + lp
        entropy = entropy + e
        logits['button'] = b_logits
        
        if pos_snap_eq is None:
            raise Exception('Turns out this is bad!')
            click_eq = None
            release_eq = None
        else:
            both_eq = torch.stack((neg_snap_eq, pos_snap_eq), dim=-1)
            b = out_sample['button'].shape[0]
            click_eq = both_eq[range(b), ..., out_sample['button']]
            release_eq = both_eq[range(b), ..., ~out_sample['button']]
        
        click_sample = None if sample is None else sample['click']
        s,lp,e,x,c_logits = self.click_decoder(
            x, image_x, click_eq, sample=click_sample)
        out_sample['click'] = s
        log_prob = log_prob + lp
        entropy = entropy + e
        logits['click'] = c_logits
        
        release_sample = None if sample is None else sample['release']
        s,lp,e,rx,r_logits = self.release_decoder(
            x, image_x, release_eq, sample=release_sample)
        out_sample['release'] = s
        x = x * ~do_release.view(-1,1) + rx * do_release.view(-1,1)
        log_prob = log_prob + lp * do_release
        entropy = entropy + e * do_release
        logits['release'] = r_logits
        
        #if self.forward_passes > 1024:
        #    breakpoint()
        #self.forward_passes += 1
        
        return out_sample, log_prob, entropy, x, logits

class ScreenDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        th = config.tile_height
        tw = config.tile_width
        self.config = config
        
        c = self.config.channels
        c_out = config.image_attention_channels
        self.k_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c, c//2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(c//2, c//2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c//2, c//4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(c//4, c//4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c//4, c//8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(c//8, c//8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(c//8, c//16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(c//16, c_out, kernel_size=3, padding=1),
        )
        
        self.q_head = linear_stack(
            self.config.decoder_layers,
            self.config.channels,
            out_channels=config.image_attention_channels,
            nonlinearity=config.nonlinearity,
        )
        #self.k_head = conv2d_stack(
        #    self.config.decoder_layers,
        #    self.config.channels,
        #    out_channels=config.image_attention_channels*th*tw,
        #    nonlinearity=config.nonlinearity,
        #)
        self.content_linear = nn.Linear(
            config.image_attention_channels, config.channels)
        
        #self.y_embedding = nn.Embedding(
        #    config.image_height, config.channels)
        #self.x_embedding = nn.Embedding(
        #    config.image_width, config.channels)
        
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
        th = self.config.tile_height 
        tw = self.config.tile_width
        ac = self.config.image_attention_channels
        ih = self.config.image_height
        iw = self.config.image_width
        
        q = self.q_head(x)
        k = self.k_head(image_x)
        #k = k.view(b,ac,th,tw,h,w).permute(0,1,4,2,5,3).reshape(b,ac,ih,iw)
        
        qk = torch.einsum('bc,bchw->bhw', q, k)
        temperature = 1. / ac ** 0.5
        a = qk * temperature
        
        click_distribution = Categorical(logits=a.view(b,ih*iw))
        
        if sample is None:
            flat_sample = click_distribution.sample()
            sample_y = torch.div(flat_sample, iw, rounding_mode='floor')
            sample_x = flat_sample % iw
            sample = torch.stack((sample_y, sample_x), dim=-1)
        else:
            sample_y = sample[:,0]
            sample_x = sample[:,1]
            flat_sample = sample_y * iw + sample_x
        
        if equivalence is None:
            raise Exception('NOPE NOPE NOPE, NEED EQUIVALENCE FOR PPO')
            log_prob = click_distribution.log_prob(flat_sample)
            entropy = click_distribution.entropy()
        else:
            if self.training:
                eq_dropout = 0.5
            else:
                eq_dropout = 0.0
            eq_distribution = equivalent_outcome_categorical(
                a, equivalence, dropout=eq_dropout)
            eq_sample = equivalence[range(b),sample_y,sample_x]
            log_prob = eq_distribution.log_prob(eq_sample)
            
            # TMP
            #print('WARNING TURNED OFF EQ')
            #log_prob_eq = eq_distribution.log_prob(eq_sample)
            #log_prob = click_distribution.log_prob(flat_sample)
            # TMP
            
            entropy = eq_distribution.entropy()
        
        #x_pe = self.x_embedding(sample_x)
        #y_pe = self.y_embedding(sample_y)
        content_x = self.content_linear(k[range(b),:,sample_y,sample_x])
        
        #x = x + self.norm(content_x + x_pe + y_pe)
        x = self.norm(content_x)
        
        screen_logits = click_distribution.logits.view(b, ih, iw)
        
        return sample, log_prob, entropy, x, screen_logits
    
    #def log_prob(self, output, action):
    #    b = output['equivalence'].shape[0]
    #    eq_index = output['equivalence'][range(b),action[:,0],action[:,1]]
    #    return output['distribution'].log_prob(eq_index)
    #
    #def entropy(self, output):
    #    return output['distribution'].entropy()
