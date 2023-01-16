import math

import torch
from torch.nn import Module, Linear, Embedding, ReLU, GELU, Dropout, LayerNorm
from torch.distributions import Categorical

from ltron.config import Config
from ltron.name_span import NameSpan
from ltron.constants import MAX_SNAPS_PER_BRICK

from ltron_torch.models.transformer import make_nonlinearity
from ltron_torch.models.mlp import ResidualBlock, linear_stack
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.coarse_to_fine_decoder import CoarseToFineDecoder

class CoarseToFineVisualCursorDecoder(Module):
    def __init__(self,
        #coarse_span,
        #fine_shape,
        space,
        fine_shape,
        channels=768,
        **kwargs,
    ):
        # module super
        super().__init__()
        
        self.space = space
        tile_height, tile_width = fine_shape[:2]
        
        coarse_span = NameSpan()
        for screen in space.layout.keys():
            if screen == 'NO_OP':
                continue
            sh, sw, *_ = space.get_shape(screen).get_shape('screen')
            assert sh % tile_height == 0
            assert sh % tile_width == 0
            ch = sh // tile_height
            cw = sw // tile_width
            coarse_span.add_names(**{screen:(ch, cw)})
        
        fine_cells = 1
        for s in fine_shape:
            fine_cells *= s
        
        #super().__init__(
        self.coarse_to_fine = CoarseToFineDecoder(
            (coarse_span.total, fine_cells),
            channels=channels,
            **kwargs,
        )
        #self.no_op_head = Linear(channels, 1)
        #self.deselect_head = Linear(channels, len(coarse_span.keys()))
        self.no_op_deselect_head = LinearMultiheadDecoder(
            in_channels=channels,
            heads={'NO_OP':1, **{screen:1 for screen in coarse_span.keys()}},
        )
        
        self.coarse_span = coarse_span
        self.fine_shape = fine_shape
    
    def forward(self, x, **kwargs):
        #no_op_x = self.no_op_head(x)
        #deselect_x = self.deselect_head(x)
        
        no_op_deselect_x = self.no_op_deselect_head(x)
        
        #x = super().forward(x, **kwargs)
        x = self.coarse_to_fine(x, **kwargs)
        b, c, f = x.shape
        
        x_out = {'NO_OP':no_op_deselect_x['NO_OP']}
        for screen in self.coarse_span.keys():
            start, stop = self.coarse_span.name_range(screen)
            
            c_shape = self.coarse_span.get_shape(screen)
            f_shape = self.fine_shape
            x_screen = x[:,start:stop].view(b, *c_shape, *f_shape)
            
            # YIKES
            if len(c_shape) > 1:
                assert len(f_shape) >= len(c_shape)
                permute_order = [0]
                for i in range(len(c_shape)):
                    permute_order.append(i+1)
                    permute_order.append(i+1+len(c_shape))
                for i in range(len(f_shape) - len(c_shape)):
                    permute_order.append(permute_order[-1]+1)
                
                x_screen = x_screen.permute(*permute_order).contiguous()
            
            x_screen = x_screen.view(b, -1)
            screen_shape = self.space.get_shape(screen)
            device = x_screen.device
            x_ravel = torch.zeros(b, screen_shape.total, device=device)
            deselect_x = no_op_deselect_x[screen]
            screen_shape.ravel_vector(
                {'deselect':deselect_x, 'screen':x_screen}, out=x_ravel, dim=1)
            
            #x_out.append(x_screen)
            x_out[screen] = x_ravel
        
        #x_out = torch.cat(x_out, dim=-1)
        x_ravel = torch.zeros(b, self.space.total, device=device)
        self.space.ravel_vector(x_out, out=x_ravel, dim=1)
        
        return x_ravel

class CoarseToFineSymbolicCursorDecoder(Module):
    def __init__(self,
        max_instances,
        channels=768,
        **kwargs,
    ):
        # module init
        super().__init__()
        
        total_instances = sum(max_i+1 for max_i in max_instances.values())
        #super().__init__(
        #    (total_instances, MAX_SNAPS_PER_BRICK),
        #    channels=channels,
        #    **kwargs,
        #)
        self.coarse_to_fine = CoarseToFineDecoder(
            (total_instances, MAX_SNAPS_PER_BRICK),
            channels=channels,
            **kwargs,
        )
        self.no_op_head = Linear(channels, 1)
        
    def forward(self, x, **kwargs):
        no_op_x = self.no_op_head(x)
        
        x = self.coarse_to_fine(x, **kwargs)
        b = x.shape[0]
        x = x.view(b, -1)
        
        x = torch.cat((no_op_x, x), dim=-1)
        
        return x

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
