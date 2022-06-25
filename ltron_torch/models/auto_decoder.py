from ltron.config import Config
from ltron.name_span import NameSpan

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
        
        for name, space in action_space.subspaces.items():
            if isinstance(space, MultiScreenPixelSpace):
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
            elif isinstance(space, SymbolicSnapSpace):
                self.readout_layout.add_names(**{name:1})
                self.symbolic_cursor_names.append(name)
            elif isinstance(space, BrickShapeColorSpace):
                self.readout_layout.add_names(**{name:1})
                self.brick_inserters[name] = (
                    space.num_shapes, space.num_colors)
            else:
                self.noncursor_names.append(name)
        
        self.readout_layout.add_names(noncursor=1)
    
    def forward(self, x, readout_x, readout_t, seq_pad):
        decoder_x = {}
        
        max_seq = torch.max(seq_pad)
        min_t = torch.min(readout_t, dim=0).values
        
        for name in self.readout_layout.keys():
            if name == 'PAD':
                continue
            readout_index = self.readout_layout.ravel(name, 0)
            readout_s, readout_b = torch.where(readout_x == readout_index)
            name_x = x[readout_s, readout_b]
            sb = name_x.shape[0]
            name_x = self.decoders[name](name_x)
            if isinstance(name_x, dict):
                decoder_x.update(name_x)
            else:
                decoder_x[name] = name_x
            
            if readout_i is None:
                readout_i = (readout_t - min_t.view(1,-1))[readout_s, readout_b]
        
        flat_x = torch.zeros(sb, self.action_space.n, device=device)
        last_dim = len(flat_x.shape)-1
        self.action_space.ravel_vector(decoder_x, out=flat_x, dim=last_dim)
        
        out_x = torch.zeros(max_seq, b, self.action_space.n, device=device)
        out_x[readout_i, readout_b] = flat_x
        
        return out_x
    
    def tensor_to_distribution(self, x):
        
