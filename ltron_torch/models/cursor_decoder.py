import math

import torch
from torch.nn import Module, Linear, Embedding, ReLU, GELU, Dropout, LayerNorm
from torch.distributions import Categorical

from ltron.config import Config

from ltron_torch.models.transformer import make_nonlinearity
from ltron_torch.models.mlp import ResidualBlock, linear_stack
from ltron_torch.models.conditional_decoder import ConditionalDecoder

class CoarseToFineCursorDecoderConfig(Config):
    nonlinearity = 'gelu'
    
    channels = 768
    
    cursor_decoder_dropout = 0.

'''
PLAN:
1. get rid of the config, pass the arguments in
2. make a coarse-to-fine super class
3. make CoarseToFineCursorDecoder inherit from that, and do whatever squirrely
    reshaping thing it needs to do
4. make CoarseToFineBrickTypeDecoder inherit from that if the base class alone
    isn't good enough.
'''

'''
class CoarseToFineBrickTypeDecoder(Module):
    def __init__(self,
        config,
        num_shapes,
        num_colors,
        coarse_layers=3,
        fine_layers=3,
        nonlinearity='gelu',
        channels=768,
        dropout=0.,
    ):
        super().__init__()
        
        self.num_shapes = num_shapes
        self.num_colors = num_colors
        
        self.coarse_head = linear_stack(
            coarse_layers,
            channels,
            out_channels=self.coarse_span.total,
            nonlinearity=nonlinearity,
            hidden_dropout=dropout,
        )
        self.coarse_embedding = Embedding(self.num_shapes, channels)
        
        self.fine_head = linear_stack(
            fine_layers,
            channels*2,
            hidden_channels=channels,
            out_channels=num_colors,
            nonlinearity=nonlinearity,
            hidden_dropout=dropout,
        )
        
        self.input_norm = LayerNorm(channels)
        self.embedding_norm = LayerNorm(channels)
        
        self.no_op_head = Linear(config.channels, 1)
    
    def forward(self, x, sample_mode='top4'):
'''     

class CoarseToFineCursorDecoder(ConditionalDecoder):
    def __init__(self,
        coarse_span,
        fine_shape,
        channels=768,
        **kwargs,
    ):
        fine_cells = 1
        for s in fine_shape:
            fine_cells *= s
        
        super().__init__(
            (coarse_span.total, fine_cells),
            channels=channels,
            **kwargs,
        )
        self.no_op_head = Linear(channels, 1)
        
        self.coarse_span = coarse_span
        self.fine_shape = fine_shape
    
    def forward(self, x, **kwargs):
        no_op_x = self.no_op_head(x)
        
        x = super().forward(x, **kwargs)
        b, c, f = x.shape
        
        x_out = [no_op_x]
        for name in self.coarse_span.keys():
            start, stop = self.coarse_span.name_range(name)
            
            c_shape = self.coarse_span.get_shape(name)
            f_shape = self.fine_shape
            x_name = x[:,start:stop].view(b, *c_shape, *f_shape)
            
            # YIKES
            if len(c_shape) > 1:
                assert len(f_shape) >= len(c_shape)
                permute_order = [0]
                for i in range(len(c_shape)):
                    permute_order.append(i+1)
                    permute_order.append(i+1+len(c_shape))
                for i in range(len(f_shape) - len(c_shape)):
                    permute_order.append(permute_order[-1]+1)
                
                x_name = x_name.permute(*permute_order).contiguous()
            
            x_name = x_name.view(b, -1)
            x_out.append(x_name)
        
        x_out = torch.cat(x_out, dim=-1)
        
        return x_out

class OldCoarseToFineCursorDecoder(Module):
    def __init__(self,
        config,
        coarse_span,
        fine_shape,
        coarse_layers=3,
        fine_layers=3,
    ):
        super().__init__()
        
        self.config = config
        self.coarse_span = coarse_span
        self.fine_shape = fine_shape
        self.fine_total = 1
        for f in fine_shape:
            self.fine_total *= f
        
        self.coarse_head = linear_stack(
            coarse_layers,
            config.channels,
            out_channels=self.coarse_span.total,
            nonlinearity=config.nonlinearity,
            hidden_dropout=config.cursor_decoder_dropout,
        )
        self.coarse_embedding = Embedding(
            self.coarse_span.total, config.channels)
        
        self.fine_head = linear_stack(
            fine_layers,
            config.channels*2,
            hidden_channels=config.channels,
            out_channels=self.fine_total,
            nonlinearity=config.nonlinearity,
            hidden_dropout=config.cursor_decoder_dropout,
        )
        
        self.input_norm = LayerNorm(config.channels)
        self.embedding_norm = LayerNorm(config.channels)
        
        self.no_op_head = Linear(config.channels, 1)
    
    def forward(self, x, sample_mode='top4'):
        assert sample_mode in ('sample', 'max', 'top4')
        sb, c = x.shape
        
        x = self.input_norm(x)
        no_op_x = self.no_op_head(x)
        coarse_x = self.coarse_head(x)
        if sample_mode == 'sample':
            dist = Categorical(logits=coarse_x)
            i = dist.sample().view(sb, 1)
        elif sample_mode == 'max':
            i = torch.argmax(coarse_x, dim=-1).view(sb, 1)
        elif sample_mode == 'top4':
            i = torch.topk(coarse_x, 4, dim=-1).indices
        
        k = i.shape[-1]
        e = self.coarse_embedding(i)
        e = self.embedding_norm(e)
        x = x.view(sb, 1, c).expand(sb, k, c)
        fine_x = torch.cat((x, e), dim=-1)
        fine_x = self.fine_head(fine_x)
        
        b, n = coarse_x.shape
        b, k, f = fine_x.shape
        x = coarse_x.view(b, n, 1)
        x = x.expand(b, n, f).clone() - math.log(self.fine_total)
        
        bb = torch.arange(b).view(b, 1).expand(b, k).reshape(-1).to(x.device)
        
        x[bb, i.view(-1)] = (
            x[bb, i.view(-1)] +
            fine_x.view(sb*k, f) +
            math.log(self.fine_total) -
            torch.logsumexp(fine_x, dim=-1).view(-1, 1)
        )
        
        x_out = [no_op_x]
        for name in self.coarse_span.keys():
            start, stop = self.coarse_span.name_range(name)
            
            c_shape = self.coarse_span.get_shape(name)
            f_shape = self.fine_shape
            x_name = x[:,start:stop].view(b, *c_shape, *f_shape)
            
            # YIKES
            if len(c_shape) > 1:
                assert len(f_shape) >= len(c_shape)
                permute_order = [0]
                for i in range(len(c_shape)):
                    permute_order.append(i+1)
                    permute_order.append(i+1+len(c_shape))
                for i in range(len(f_shape) - len(c_shape)):
                    permute_order.append(permute_order[-1]+1)
                
                x_name = x_name.permute(*permute_order).contiguous()
            
            x_name = x_name.view(b, -1)
            x_out.append(x_name)
        
        x_out = torch.cat(x_out, dim=-1)
        
        return x_out

class CoarseToFineCursorDecoder2d(Module):
    def __init__(self,
        config,
        #height,
        #width,
        #coarse_positions,
        coarse_span,
        fine_shape,
        #fine_height,
        #fine_width,
        #fine_positions,
        #fine_channels,
        coarse_layers=3,
        fine_layers=3,
        #expand_distribution=True,
    ):
        super().__init__()
        
        self.config = config
        self.coarse_span = coarse_span
        self.fine_shape = fine_shape
        self.fine_total = 1
        for f in fine_shape:
            self.fine_total *= f
        #self.coarse_start_stop = {}
        '''
        for name, shape in self.coarse_span.items():
            n = 1
            for s in shape:
                n *= s
            self.coarse_start_stop[name] = (
                self.total_coarse_positions, self.total_coarse_positions+n)
            self.total_coarse_positions += n
        '''
        #self.height = height
        #self.width = width
        #self.coarse_positions = coarse_positions
        #self.fine_height = fine_height
        #self.fine_width = fine_width
        #self.fine_positions = fine_positions
        #self.cursor_channels = cursor_channels
        #self.expand_distribution = expand_distribution
        
        #self.fine_positions = fine_height * fine_width
        #self.fine_channels = self.fine_positions * self.cursor_channels
        
        #self.coarse_height = self.height // self.fine_height
        #self.coarse_width = self.width // self.fine_width
        #self.coarse_positions = self.coarse_height * self.coarse_width
        
        self.coarse_head = linear_stack(
            coarse_layers,
            config.channels,
            out_channels=self.coarse_span.total,
            nonlinearity=config.nonlinearity,
            hidden_dropout=config.cursor_decoder_dropout,
        )
        self.coarse_embedding = Embedding(
            self.coarse_span.total, config.channels)
        
        self.fine_head = linear_stack(
            fine_layers,
            config.channels*2,
            hidden_channels=config.channels,
            out_channels=self.fine_total,
            nonlinearity=config.nonlinearity,
            hidden_dropout=config.cursor_decoder_dropout,
        )
        
        # split this out
        self.input_norm = LayerNorm(config.channels)
        self.embedding_norm = LayerNorm(config.channels)
        
        self.no_op_head = Linear(config.channels, 1)
    
    def forward(self, x, sample_mode='top4'):
        assert sample_mode in ('sample', 'max', 'top4')
        sb, c = x.shape
        
        x = self.input_norm(x)
        no_op_x = self.no_op_head(x)
        coarse_x = self.coarse_head(x)
        if sample_mode == 'sample':
            dist = Categorical(logits=coarse_x)
            i = dist.sample().view(sb, 1)
        elif sample_mode == 'max':
            i = torch.argmax(coarse_x, dim=-1).view(sb, 1)
        elif sample_mode == 'top4':
            i = torch.topk(coarse_x, 4, dim=-1).indices
        
        k = i.shape[-1]
        e = self.coarse_embedding(i)
        e = self.embedding_norm(e)
        x = x.view(sb, 1, c).expand(sb, k, c)
        fine_x = torch.cat((x, e), dim=-1)
        fine_x = self.fine_head(fine_x)
        
        #return {'coarse_x':coarse_x, 'fine_x':fine_x, 'coarse_i':i}
        
        #coarse_maps = {}
        #for name in self.coarse_span.keys():
        #    shape = self.coarse_span.get_shape(name)
        #    start, end = self.coarse_span.name_range(name)
        #    coarse_maps[name] = coarse_x[:,start:end].reshape(sb,*shape)
        
        #if self.expand_distribution:
        b, n = coarse_x.shape
        b, k, f = fine_x.shape
        x = coarse_x.view(b, n, 1)
        x = x.expand(b, n, f).clone() - math.log(self.fine_total)
        
        bb = torch.arange(b).view(b, 1).expand(b, k).reshape(-1).to(x.device)
        
        x[bb, i.view(-1)] = (
            x[bb, i.view(-1)] +
            fine_x.view(sb*k, f) +
            math.log(self.fine_total) -
            torch.logsumexp(fine_x, dim=-1).view(-1, 1)
        )
        
        x_out = [no_op_x] #, x.view(b,-1)]
        #x_out.append(torch.zeros(b, 1, device=x.device))
        for name in self.coarse_span.keys():
            start, stop = self.coarse_span.name_range(name)
            ch, cw = self.coarse_span.get_shape(name)
            fh, fw, fp = self.fine_shape
            x_name = x[:,start:stop].view(b, ch, cw, fh, fw, fp)
            x_name = x_name.permute(0,1,3,2,4,5).contiguous()
            x_name = x_name.view(b, -1)
            x_out.append(x_name)
        
        x_out = torch.cat(x_out, dim=-1)
        
        #for name, (start, stop) in self.coarse_start_stop.items():
        #    x_out[name] = x[:,start:stop].reshape(
        #        b,*self.coarse_shapes[name],f)
        
        return x_out
        #else:
        #    x_out = {}
        #    for name, (start, stop) in self.coarse_start_stop.items():
        #        x_out[name] = coarse_x[:,start:stop].reshape(
        #            b,*self.coarse_shapes[name])
        #    return x_out, x_fine, i
