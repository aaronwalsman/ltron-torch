import torch
from torch.nn import (
    Module, Identity, Conv2d, Sequential, ReLU, MaxPool2d, Upsample,
    MultiheadAttention, LayerNorm, Linear, Parameter, Embedding, Dropout)

from ltron_torch.config import Config
from ltron_torch.models.positional_encoding import sinusoid_positional_encoding
import ltron_torch.models.dvae as dvae

class SlotoencoderConfig(Config):
    encoder_mode = 'token_map'
    
    num_data_tokens = 32*32
    vocabulary = 4096
    
    channels = 256
    residual_channels = None
    output_channels = 4096
    num_slots = 64
    num_encoder_blocks = 4
    num_heads = 1
    
    num_decoder_blocks = 4
    tile_shape = (32,32)
    
    weight_scalar = None
    
    attention_dropout = 0.1
    residual_dropout = 0.1
    decoder_dropout = 0.0
    
    def set_dependent_variables(self, **kwargs):
        if self.residual_channels is None:
            self.residual_channels = self.channels*4
        
        if self.weight_scalar is None:
            self.weight_scalar = 1./self.channels**0.5

class SlotSampler(Module):
    def __init__(self, config):
        super(SlotSampler, self).__init__()
        self.n = config.num_slots
        
        '''
        mu = torch.zeros(config.channels)
        mu.normal_(0., config.weight_scalar)
        self.mu = Parameter(mu)
        #self.mu = Parameter(torch.randn(config.channels))
        #self.log_sigma = Parameter(torch.zeros(config.channels))
        #torch.nn.init.xavier_uniform_(self.log_sigma)
        #self.log_sigma = Parameter(
        #    (torch.rand(config.channels)*2-1)/config.channels**0.5)
        log_sigma = torch.zeros(config.channels)
        log_sigma.normal_(0., config.weight_scalar)
        self.log_sigma = Parameter(log_sigma)
        '''
        
        #self.embedding = torch.nn.Embedding(self.n, config.channels)
        e = torch.randn(self.n, 1, config.channels) * config.weight_scalar
        self.slot_embedding = Parameter(e)
        #self.positional_encoding = Parameter(
        #    torch.zeros(config.num_data_tokens, 1, config.channels))
        
        #p = positional_encoding(config.channels, config.num_data_tokens, 1)
        #self.register_buffer('positional_encoding', p)
        
        p = torch.randn(
            config.num_data_tokens, 1, config.channels) * config.weight_scalar
        self.positional_encoding = Parameter(p)
    
    def forward(self, x):
        t, b, c = x.shape
        
        '''
        mu = self.mu.view(1, 1, c)
        #sigma = self.log_sigma.exp().view(1, 1, c)
        sigma = config.weight_scalar
        
        s = mu + sigma * torch.randn((self.n, b, c), device=x.device)
        x = torch.cat((s, x), dim=0)
        '''
        
        x = x + self.positional_encoding
        
        #s = self.embedding.weight.view(self.n, 1, c).expand(self.n, b, c)
        s = self.slot_embedding.expand(self.n, b, c)
        x = torch.cat((s, x), dim=0)
        
        return x


class EncoderBlock(Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.num_slots = config.num_slots
        
        self.lns1 = LayerNorm(config.channels)
        self.lnd1 = LayerNorm(config.channels)
        self.attention1 = MultiheadAttention(
            config.channels, config.num_heads, dropout=config.attention_dropout)
        
        self.ln2 = LayerNorm(config.channels)
        self.attention2 = MultiheadAttention(
            config.channels, config.num_heads, dropout=config.attention_dropout)
        
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            ReLU(),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
        )
    
    def forward(self, x):
        s = x[:self.num_slots]
        d = x[self.num_slots:]
        
        rs = self.lns1(s)
        rd = self.lnd1(d)
        rs = self.attention1(rs, rd, rd)[0]
        s = s + rs
        
        r = self.ln2(s)
        r = self.attention2(r, r, r)[0]
        s = s + r
        
        s = s + self.projection_residual(s)
        
        # breaks differentiation
        #x[:self.num_slots] = s
        x = torch.cat((s, d), dim=0)
        
        return x


class Encoder(Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.num_slots = config.num_slots
        
        if config.encoder_mode == 'image':
            raise NotImplementedError
            self.embedding = dvae.Encoder(need_to_convert_dvae_to_config)
        
        elif config.encoder_mode == 'token_map':
            self.embedding = Embedding(config.vocabulary, config.channels)
        
        else:
            raise NotImplementedError
        
        self.slot_sampler = SlotSampler(config)
        
        self.encoder_blocks = Sequential(*[
            EncoderBlock(config) for _ in range(config.num_encoder_blocks)])
    
    def forward(self, x, pause=False):
        x = self.embedding(x)
        x = self.slot_sampler(x)
        x = self.encoder_blocks(x)
        #x = x[:self.num_slots]
        
        return x


class TileSlotsB(Module):
    def __init__(self, config):
        super(TileSlotsB, self).__init__()
        p = torch.randn(
            config.num_data_tokens, 1, config.channels) * config.weight_scalar
        self.positional_encoding = Parameter(p)
    
    def forward(self, x):
        t, b, c = x.shape
        
        p = self.positional_encoding.expand(-1, b, c)
        x = torch.cat((x, p), dim=0)
        
        return x
        

class DecoderBlockB(Module):
    def __init__(self, config):
        super(DecoderBlockB, self).__init__()
        self.num_slots = config.num_slots
        
        self.ln1 = LayerNorm(config.channels)
        self.ln2 = LayerNorm(config.channels)
        self.attention1 = MultiheadAttention(
            config.channels, config.num_heads, dropout=config.attention_dropout)
        
        self.projection_residual = Sequential(
            LayerNorm(config.channels),
            Linear(config.channels, config.residual_channels),
            ReLU(),
            Linear(config.residual_channels, config.channels),
            Dropout(config.residual_dropout),
            #Linear(config.channels, config.channels),
        )
    
    def forward(self, x, pause=False):
        s = x[:self.num_slots]
        d = x[self.num_slots:]
        
        if pause:
            import pdb
            pdb.set_trace()
        
        rs = self.ln1(s)
        rd = self.ln1(d)
        rd = self.attention1(rd, rs, rs)[0]
        d = d + rd
        
        d = d + self.projection_residual(d)
        
        # breaks differentiation
        #x[:self.num_slots] = s
        x = torch.cat((s, d), dim=0)
        
        return x


class DecoderB(Module):
    def __init__(self, config):
        super(DecoderB, self).__init__()
        self.num_slots = config.num_slots
        self.tile_shape = config.tile_shape
        
        self.tile_slots = TileSlotsB(config)
        self.decoder_blocks = Sequential(*[
            DecoderBlockB(config) for _ in range(config.num_decoder_blocks)])
        #self.output_conv = Conv2d(
        #    config.channels, config.output_channels, kernel_size=1)
        self.output_linear = Linear(config.channels, config.output_channels)
    
    def forward(self, x, pause=False):
        x1 = self.tile_slots(x)
        #x1 = x
        #x2 = self.decoder_blocks(x1)
        x2 = self.decoder_blocks[0](x1, pause=pause)
        
        #x3 = x2[self.num_slots:].permute(1, 2, 0).contiguous()
        #b, c, hw = x3.shape
        
        #x3 = x3.view(b, c, *self.tile_shape)
        x3 = self.output_linear(x2[self.num_slots:])
        
        #pred = torch.argmax(x4, dim=1).view(b, -1)
        #num_ones = torch.sum(pred, dim=-1)
        #pause0 = num_ones[0] == 1 and pred[0][0] == 1
        #pause1 = num_ones[1] == 1 and pred[0][1] == 1
        #pause = pause0 or pause1
        
        #if pause:
        #    import pdb
        #    pdb.set_trace()
        
        return x3


class TileSlots(Module):
    def __init__(self, config):
        super(TileSlots, self).__init__()
        self.tile_shape = config.tile_shape
        num_tiled_slots = self.tile_shape[0] * self.tile_shape[1]
        
        #p = positional_encoding(config.channels, num_tiled_slots)
        #p = p.permute(1,0).view(1, config.channels, num_tiled_slots)
        #self.register_buffer('positional_encoding', p)
        
        #p = torch.zeros(1, config.channels, num_tiled_slots)
        p = torch.randn(
            1, config.channels, num_tiled_slots) * config.weight_scalar
        self.positional_encoding = Parameter(p)
    
    def forward(self, x):
        s, b, c = x.shape
        x = x.view(s*b, c, 1)
        x = x + self.positional_encoding
        x = x.view(s*b, c, *self.tile_shape)
        #import pdb
        #pdb.set_trace()
        
        return x


class DecoderBlock(Module):
    def __init__(self, config):
        super(DecoderBlock, self).__init__()
        
        self.residual = Sequential(
            LayerNorm((config.channels, *config.tile_shape)),
            Conv2d(config.channels, config.residual_channels, kernel_size=1),
            ReLU(),
            Conv2d(config.residual_channels, config.channels, kernel_size=1),
            Dropout(config.decoder_dropout),
        )
    
    def forward(self, x):
        x = x + self.residual(x)
        return x

class TileCompositor(Module):
    def __init__(self, config):
        super(TileCompositor, self).__init__()
        self.num_slots = config.num_slots
        self.attention_channel = Conv2d(config.channels, 1, kernel_size=1)
    
    def forward(self, x):
        a = self.attention_channel(x)
        
        sb, c, h, w = x.shape
        x = x.view(self.num_slots, -1, c, h, w)
        a = a.view(self.num_slots, -1, 1, h, w)
        a = torch.softmax(a, dim=0)
        x = torch.sum(a*x, dim=0)
        
        return x

class Decoder(Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        
        self.tile_slots = TileSlots(config)
        self.decoder_blocks = Sequential(*[
            DecoderBlock(config) for _ in range(config.num_decoder_blocks)])
        self.tile_compositor = TileCompositor(config)
        self.output_conv = Conv2d(
            config.channels, config.output_channels, kernel_size=1)
    
    def forward(self, x, pause=False):
        x1 = self.tile_slots(x)
        x2 = self.decoder_blocks(x1)
        x3 = self.tile_compositor(x2)
        x4 = self.output_conv(x3)
        
        if pause:
            import pdb
            pdb.set_trace()
        
        return x4

class Slotoencoder(Module):
    def __init__(self, config):
        super(Slotoencoder, self).__init__()
        self.num_slots = config.num_slots
        self.weight_scalar = config.weight_scalar
        
        self.encoder = Encoder(config)
        self.decoder = DecoderB(config)
        
        self.apply(self._init_weights)
        
        self.classifier = torch.nn.Linear(256*config.num_slots, 2)
    
    def _init_weights(self, module):
        if isinstance(module, (Linear, Conv2d, Embedding)):
            module.weight.data.normal_(mean=0., std=self.weight_scalar)
            if isinstance(module, (Linear, Conv2d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.)
    
    def forward(self, x, pause=False):
        x = self.encoder(x, pause=pause)
        
        t,b,c = x.shape
        xc = x[:self.num_slots].permute(1,0,2).contiguous().view(
            b, self.num_slots*c)
        c = self.classifier(xc)
        
        x = self.decoder(x[:self.num_slots], pause=pause)
        
        return x, c
