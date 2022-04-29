import torch
from torch.nn import Module, Linear, LayerNorm, Sequential, ModuleDict

from ltron_torch.models.mlp import linear_stack
from ltron_torch.models.heads import LinearMultiheadDecoder
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
    init_weights,
)
from ltron_torch.models.auto_embedding import (
    AutoEmbeddingConfig, AutoEmbedding)
from ltron_torch.models.coarse_to_fine_decoder import (
    CoarseToFineCursorDecoderConfig, CoarseToFineCursorDecoder)
from ltron_torch.models.padding import decat_padded_seq, get_seq_batch_indices

class AutoTransformerConfig(
    AutoEmbeddingConfig,
    TransformerConfig,
    CoarseToFineCursorDecoderConfig,
):
    action_decoder_dropout = 0.1

class AutoTransformer(Module):
    def __init__(self,
        config,
        #interface,
        observation_space,
        action_space,
        checkpoint=None
    ):
        super().__init__()
        self.config = config
        #self.interface = interface
        self.observation_space = observation_space
        self.action_space = action_space
        
        # build the token embedding
        self.embedding = AutoEmbedding(config, observation_space, action_space)
        '''
        self.embedding = AutoEmbedding(
            config,
            interface.tile_shape,
            interface.token_vocabulary,
            interface.tile_spatial_positions,
            interface.max_time_steps,
        )
        '''
        
        # build the transformer
        self.encoder = Transformer(config)
        
        # build cursor decoders
        decoders = {}
        for name in self.embedding.cursor_names:
            subspace = self.action_space.subspaces[name]
            coarse_cursor_keys = [
                k for k in subspace.screen_span.keys() if k != 'NO_OP']
            coarse_cursor_span = self.embedding.tile_position_layout.subspace(
                coarse_cursor_keys)
            fine_shape = self.embedding.cursor_fine_layout.get_shape(name)
            decoders[name] = CoarseToFineCursorDecoder(
                config, coarse_cursor_span, fine_shape)
        
        # build the noncursor decoder
        decoders['noncursor'] = Sequential(
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
                {name:self.action_space.subspaces[name].n
                 for name in self.embedding.noncursor_names
                }
            ),
        )
        
        self.decoders = ModuleDict(decoders)
        
        '''
        # build the cursor decoders
        decoders = {}
        self.decoder_data = {}
        self.fine_shape = (*interface.tile_shape[:2], 2)
        for name, cursor_data in interface.cursor_screen_data.items():
            tile_h, tile_w = interface.tile_shape[:2]
            total_coarse_positions = 0
            self.decoder_data[name] = {}
            coarse_shapes = {}
            for screen_name, screen_data in cursor_data.items():
                screen_h, screen_w, screen_c = screen_data['shape']
                assert screen_h % tile_h == 0
                assert screen_w % tile_w == 0
                coarse_h = screen_h // tile_h
                coarse_w = screen_w // tile_w
                screen_coarse_positions = coarse_h * coarse_w
                self.decoder_data[name][screen_name] = {
                    'start':total_coarse_positions,
                    'end':total_coarse_positions+screen_coarse_positions,
                    #'coarse_shape':(coarse_h, coarse_w),
                    #'fine_shape':(tile_h, tile_w, screen_c),
                }
                coarse_shapes[screen_name] = (coarse_h, coarse_w)
                total_coarse_positions += screen_coarse_positions
            
            fine_positions = tile_h * tile_w * screen_c
            
            decoders[name] = CoarseToFineCursorDecoder(
                config, coarse_shapes, fine_positions)
        '''
        '''
        for name, cursor_screen_data in interface.cursor_screen_data.items():
            for screen_name, screen_data in cursor_screen_data.items():
                cursor_decoder = CoarseToFineCursorDecoder(
                    config,
                    *screen_data['shape'],
                    *interface.tile_shape[:2],
                    2,
                )
                decoders['%s,%s'%(name, screen_name)] = cursor_decoder
        '''
        '''
        # build the noncursor decoder
        decoders['noncursor'] = Sequential(
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
                {name:(b-a)
                 for name, (a, b) in interface.noncursor_name_range.items()
                }
            ),
        )
        
        self.decoders = ModuleDict(decoders)
        '''
        
        # initialize weights
        if checkpoint is not None:
            if isinstance(checkpoint, str):
                checkpoint = torch.load(checkpoint)
            self.load_state_dict(checkpoint)
        elif config.init_weights:
            self.apply(init_weights)
    
    def zero_all_memory(self):
        self.encoder.zero_all_memory()
    
    def forward_rollout(self, terminal, *args, **kwargs):
        device = next(self.parameters()).device
        use_memory = torch.BoolTensor(~terminal).to(device)
        return self(*args, **kwargs, use_memory=use_memory)
    
    def observation_to_tensors(self, batch, sequence_pad, device):
        return self.embedding.observation_to_tensors(
            batch, sequence_pad, device)
    
    def forward(self,
        tile_x,
        tile_t,
        tile_yx,
        tile_pad,
        token_x,
        token_t,
        token_pad,
        cursor_t,
        cursor_fine_yxp,
        cursor_coarse_yx,
        cursor_pad,
        readout_x,
        readout_t,
        readout_pad,
        seq_pad,
        use_memory=None,
    ):
        x, t, pad = self.embedding(
            tile_x,
            tile_t,
            tile_yx,
            tile_pad,
            token_x,
            token_t,
            token_pad,
            cursor_t,
            cursor_fine_yxp,
            cursor_coarse_yx,
            cursor_pad,
            readout_x,
            readout_t,
            readout_pad,
            seq_pad,
        )
        
        # use the encoder to encode
        x = self.encoder(x, t, pad, use_memory=use_memory)[-1]
        s, b, c = x.shape
        
        # extract decoder tokens
        _, x = decat_padded_seq(x, tile_pad, token_pad)
        
        # use the decoders to decode
        max_seq = torch.max(seq_pad)
        min_t = torch.min(token_t, dim=0).values
        
        head_xs = {}
        
        #for name,readout_index in self.interface.readout_token_indices.items():
        for name in self.embedding.readout_layout.keys():
            readout_index = self.embedding.readout_layout.ravel(
                name, 0)
            readout_s, readout_b = torch.where(token_x == readout_index)
            readout_t = (token_t - min_t.view(1,-1))[readout_s, readout_b]
            r_x = x[readout_s, readout_b]
            r_x = self.decoders[name](r_x)
            if name != 'noncursor':
                r_x = {name:r_x}
            
            for head_name, head_x in r_x.items():
                #start, end = self.interface.name_range[head_name]
                # if this is a cursor, we need to unpack a bit
                if name != 'noncursor':
                    maps = {}
                    total_elements = 0
                    import pdb
                    pdb.set_trace()
                    for screen_name, screen_x in head_x.items():
                        sb, ch, cw = screen_x.shape[:3]
                        
                        fh, fw, fc = self.fine_shape
                        screen_x = screen_x.view(sb, ch, cw, fh, fw, fc)
                        screen_x = screen_x.permute(0,1,3,2,4,5)
                        screen_x = screen_x.reshape(sb, ch*fh, cw*fw, fc)
                        maps[screen_name] = screen_x
                        total_elements += ch*fh*cw*fw*fc
                    '''
                    head_decoder_data = self.decoder_data[head_name]
                    coarse_x, fine_x, head_i = head_x
                    sb, c = coarse_x.shape
                    _, f = fine_x.shape
                    dense_x = coarse_x.view(sb, c, 1)
                    dense_x = dense_x.expand(sb, c, f).clone()
                    dense_x[range(sb), head_i.view(-1)] = (
                        dense_x[range(sb), head_i.view(-1)] + fine_x)
                    
                    maps = {}
                    total_elements = 0
                    for screen_name, screen_data in head_decoder_data.items():
                        screen_start = screen_data['start']
                        screen_end = screen_data['end']
                        ch, cw = screen_data['coarse_shape']
                        fh, fw, fc = screen_data['fine_shape']
                        screen_x = dense_x[:,screen_start:screen_end]
                        screen_x = screen_x.view(sb, ch, cw, fh, fw, fc)
                        screen_x = screen_x.permute(0, 1, 3, 2, 4, 5)
                        screen_x = screen_x.reshape(sb, ch*fh, cw*fw, fc)
                        maps[screen_name] = screen_x
                        total_elements += ch*fh*cw*fw*fc
                    '''
                    head_x = torch.zeros(
                        sb, total_elements+1, device=screen_x.device)
                    cursor_space = self.action_space.subspaces[head_name]
                    cursor_space.ravel_maps(maps, out=head_x)
                    head_x = head_x[:,1:]
                
                head_xs[head_name] = head_x
        
        import pdb
        pdb.set_trace()
        
        action_x = torch.zeros(
            sb,
            self.action_space.subspace_span.total,
            device=tile_x.device,
        )
        self.action_space.ravel_subspace(head_xs, out=action_x)
        
        padded_action_x = torch.zeros(
            max_seq,
            b,
            self.action_space.subspace_span.total,
            device=tile_x.device,
        )
        padded_action_x[readout_t, readout_b] = action_x
        
        return padded_action_x
