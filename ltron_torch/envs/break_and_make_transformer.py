raise Exception('Deprecated')

class BreakAndMakeTransformerInterface:
    def __init__(self, config):
        self.config = config
    
    def observation_to_tensors(self, observation, seq_pad, device=None):
        '''
        Convert gym observations to torch tensors
        '''
        # make tiles
        wx, wi, w_pad = batch_deduplicate_tiled_seqs(
            observation['workspace_color_render'],
            seq_pad,
            self.config.tile_width,
            self.config.tile_height,
            background=102,
        )
        wi = numpy.insert(wi, (0,3,3), -1, axis=-1)
        wi[:,:,0] = 0
        b = wx.shape[1]
        
        hx, hi, h_pad = batch_deduplicate_tiled_seqs(
            observation['handspace_color_render'],
            seq_pad,
            self.config.tile_width,
            self.config.tile_height,
            background=102,
        )
        hi= numpy.insert(hi, (0,3,3), -1, axis=-1)
        hi[:,:,0] = 0
        
        # move tiles to torch/cuda
        wx = torch.FloatTensor(wx)
        hx = torch.FloatTensor(hx)
        w_pad = torch.LongTensor(w_pad)
        h_pad = torch.LongTensor(h_pad)
        tile_x, tile_pad = cat_padded_seqs(wx, hx, w_pad, h_pad)
        tile_x = default_tile_transform(tile_x).cuda()
        tile_pad = tile_pad.cuda()
        tile_i, _ = cat_padded_seqs(
            torch.LongTensor(wi), torch.LongTensor(hi), w_pad, h_pad)
        tile_i = tile_i.cuda()
        
        # make additional tokens
        max_seq_len = numpy.max(seq_pad)
        token_x = torch.LongTensor(observation['phase_switch']).cuda()
        token_i = torch.ones((max_seq_len, b, 6), dtype=torch.long) * -1
        token_i[:,:,0] = 0
        token_i[:,:,1] = torch.arange(max_seq_len).unsqueeze(-1)
        token_i = token_i.cuda()
        token_pad = torch.LongTensor(seq_pad).cuda()
        
        # make decoder indices and pad
        decoder_i = (
            torch.arange(max_seq_len).unsqueeze(-1).expand(max_seq_len, b))
        decoder_pad = torch.LongTensor(pad).cuda()
        
        return (
            tile_x, tile_i, tile_pad,
            token_x, token_i, token_pad,
            decoder_i, decoder_pad,
        )
