import torch
from torch.nn import Module, ModuleDict, Sequential
from torch.distributions import Categorical

from ltron.config import Config
from ltron.name_span import NameSpan
from ltron.gym.spaces import (
    MultiScreenPixelSpace,
    SymbolicSnapSpace,
    BrickShapeColorSpace,
)

from ltron_torch.models.mlp import linear_stack
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.cursor_decoder import (
    CoarseToFineVisualCursorDecoder,
    CoarseToFineSymbolicCursorDecoder,
)
from ltron_torch.models.brick_decoder import BrickDecoder

class AutoDecoderConfig(Config):
    channels = 768

class AutoDecoder(Module):
    def __init__(self,
        config,
        action_space,
    ):
        super().__init__()
        
        self.action_space = action_space
        
        self.readout_layout = NameSpan(PAD=1)
        
        self.combined_decoder_names = []
        
        decoders = {}
        
        for name, space in action_space.subspaces.items():
            if isinstance(space, MultiScreenPixelSpace):
                raise Exception('TODO')
                #decoders[name] = CoarseToFineVisualCursorDecoder(
                '''
                self.visual_cursor_names.append(name)
                self.readout_layout.add_names(**{name:1})

                # build the cursor fine-grained embedding
                first_key = next(
                    k for k in space.screen_span.keys() if k != 'NO_OP')
                mh = observation_space[first_key].mask_height
                mw = observation_space[first_key].mask_width
                h,w,c = space.screen_span.get_shape(first_key)
                assert h % mh == 0
                assert w % mw == 0
                fh = h//mh
                fw = w//mw
                self.cursor_fine_layout.add_names(**{name:(fh,fw,2)})

                self.cursor_colors[name] = []
                for i in range(c):
                    self.cursor_colors[name].append(visualization_colors[
                            self.cursor_color_n % len(visualization_colors)])
                    self.cursor_color_n += 1
                '''
            elif isinstance(space, SymbolicSnapSpace):
                self.readout_layout.add_names(**{name:1})
                decoders[name] = CoarseToFineSymbolicCursorDecoder(
                    space.max_instances, config.channels,
                )
            elif isinstance(space, BrickShapeColorSpace):
                self.readout_layout.add_names(**{name:1})
                decoders[name] = BrickDecoder(
                    space.num_shapes,
                    space.num_colors,
                    config.channels,
                )
            else:
                self.combined_decoder_names.append(name)
        
        self.readout_layout.add_names(combined=1)
        
        # build the combined decoder
        decoders['combined'] = Sequential(
            linear_stack(
                2,
                config.channels,
                nonlinearity=config.nonlinearity,
                final_nonlinearity=True,
                hidden_dropout=config.action_decoder_dropout,
                out_dropout=config.action_decoder_dropout,
            ),
            LinearMultiheadDecoder(
                config.channels,
                {
                    name:self.action_space.subspaces[name].n
                    for name in self.combined_decoder_names
                }
            ),
        )
        
        self.decoders = ModuleDict(decoders)
    
    def forward(self, x, readout_x, readout_t, seq_pad):
        device = next(iter(self.parameters())).device
        s, b, c = x.shape
        
        decoder_x = {}
        
        max_seq = torch.max(seq_pad)
        
        for name in self.readout_layout.keys():
            if name == 'PAD':
                continue
            
            # get index associated with this readout name
            readout_index = self.readout_layout.ravel(name, 0)
            
            # find the indices of x corresponding to the readout index
            readout_s, readout_b = torch.where(readout_x['x'] == readout_index)
            
            name_x = x[readout_s, readout_b]
            name_x = self.decoders[name](name_x)
            if isinstance(name_x, dict):
                decoder_x.update(name_x)
            else:
                decoder_x[name] = name_x
        
        ts = torch.max(seq_pad)
        tb = seq_pad.shape[0]
        out_x = torch.zeros(ts*tb, self.action_space.n, device=device)
        self.action_space.ravel_vector(decoder_x, out=out_x, dim=1)
        out_x = out_x.view(ts, tb, self.action_space.n)
        
        return out_x
    
    def tensor_to_distribution(self, x):
        s, b, a = x.shape
        assert s == 1
        distribution = Categorical(logits=x)
        
        return distribution
