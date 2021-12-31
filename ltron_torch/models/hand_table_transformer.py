import os

import numpy

import torch
from torch.nn import Module, Embedding, Linear, LayerNorm, Sequential

import tqdm

from splendor.image import save_image

from ltron.config import Config
from ltron.hierarchy import len_hierarchy, index_hierarchy
from ltron.gym.envs.blocks_env import BlocksEnv
from ltron.compression import (
    batch_deduplicate_tiled_seqs, batch_deduplicate_from_masks)
from ltron.visualization.drawing import write_text

from ltron_torch.gym_tensor import default_tile_transform
from ltron_torch.models.embedding import (
    TileEmbedding,
    TokenEmbedding,
)
from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask
from ltron_torch.models.mask import padded_causal_mask
from ltron_torch.models.positional_encoding import LearnedPositionalEncoding
from ltron_torch.models.embedding import TileEmbedding, TokenEmbedding
from ltron_torch.models.transformer import (
    TransformerConfig,
    Transformer,
    TransformerBlock,
    init_weights,
)
from ltron_torch.models.heads import LinearMultiheadDecoder
#from ltron_torch.interface.utils import categorical_or_max, categorical_or_max_2d

class HandTableTransformerConfig(Config):
    tile_h = 16
    tile_w = 16
    tile_c = 3
    table_h = 256
    table_w = 256
    hand_h = 96
    hand_w = 96
    
    table_decode_h = 64
    table_decode_w = 64
    hand_decode_h = 24
    hand_decode_w = 24
    
    global_tokens = 1
    
    max_sequence_length = 1024
    
    embedding_dropout = 0.1
    
    nonlinearity = 'gelu'
    
    encoder_blocks = 12
    encoder_channels = 768
    encoder_residual_channels = None
    encoder_heads = 12
    
    decoder_blocks = 4
    decoder_channels = 768
    decoder_residual_channels = None
    decoder_heads = 12
    
    residual_dropout = 0.1
    attention_dropout = 0.1
    content_dropout = 0.1
    
    spatial_channels = 2
    num_modes = 20
    num_shapes = 6
    num_colors = 6
    
    model_checkpoint = False
    init_weights = True
    
    def set_dependents(self):
        assert self.table_h % self.tile_h == 0
        assert self.table_w % self.table_w == 0
        self.table_tiles_h = self.table_h // self.tile_h
        self.table_tiles_w = self.table_w // self.tile_w
        self.table_tiles = self.table_tiles_h * self.table_tiles_w
        
        assert self.hand_h % self.tile_h == 0
        assert self.hand_w % self.tile_w == 0
        self.hand_tiles_h = self.hand_h // self.tile_h
        self.hand_tiles_w = self.hand_w // self.tile_w
        self.hand_tiles = self.hand_tiles_h * self.hand_tiles_w
        
        self.spatial_tiles = self.table_tiles + self.hand_tiles
        
        self.table_decoder_pixels = self.table_decode_h * self.table_decode_w
        self.hand_decoder_pixels = self.hand_decode_h * self.hand_decode_w
        self.decoder_pixels = (
            self.table_decoder_pixels + self.hand_decoder_pixels)
        
        assert self.table_decode_h % self.table_tiles_h == 0
        assert self.table_decode_w % self.table_tiles_w == 0
        self.upsample_h = self.table_decode_h // self.table_tiles_h
        self.upsample_w = self.table_decode_w // self.table_tiles_w
        assert self.hand_decode_h // self.hand_tiles_h == self.upsample_h
        assert self.hand_decode_w // self.hand_tiles_w == self.upsample_w
        
        #self.global_tokens = self.num_modes + self.num_shapes + self.num_colors

class HandTableTransformer(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # build the tokenizers
        self.tile_embedding = TileEmbedding(
            config.tile_h,
            config.tile_w,
            config.tile_c,
            config.encoder_channels,
            config.embedding_dropout,
        )
        self.token_embedding = TokenEmbedding(
            2, config.encoder_channels, config.embedding_dropout)
        
        self.mask_embedding = Embedding(1, config.encoder_channels)
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.encoder_channels, config.spatial_tiles)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.encoder_channels, config.max_sequence_length)
        
        # build the transformer
        encoder_config = TransformerConfig.translate(
            config,
            blocks='encoder_blocks',
            channels='encoder_channels',
            residual_channels='encoder_residual_channels',
            num_heads='encoder_heads',
        )
        self.encoder = Transformer(encoder_config)
        
        # build the linear layer that converts from encoder to decoder channels
        self.encode_to_decode = Sequential(
            Linear(config.encoder_channels, config.decoder_channels),
            LayerNorm(config.decoder_channels)
        )
        
        # build the decoder
        self.decoder = CrossAttentionDecoder(config)
        
        # initialize weights
        if config.model_checkpoint:
            self.load_state_dict(torch.load(config.model_checkpoint))
        elif config.init_weights:
            self.apply(init_weights)
    
    def forward(self,
        tile_x, tile_t, tile_yx, tile_pad,
        token_x, token_t, token_pad,
        decode_t, decode_pad,
        use_memory=None,
    ):
        
        # make the tile embeddings
        tile_x = self.tile_embedding(tile_x)
        tile_pt = self.temporal_position_encoding(tile_t)
        tile_pyx = self.spatial_position_encoding(tile_yx)
        tile_x = tile_x + tile_pt + tile_pyx
        
        # make the tokens
        token_x = self.token_embedding(token_x)
        token_pt = self.temporal_position_encoding(token_t)
        token_x = token_x + token_pt
        
        # concatenate the tile and discrete tokens
        x, pad = cat_padded_seqs(tile_x, token_x, tile_pad, token_pad)
        t, _ = cat_padded_seqs(tile_t, token_t, tile_pad, token_pad)
        
        # use the encoder to encode
        x = self.encoder(x, t, pad, use_memory=use_memory)
        
        # convert encoder channels to decoder channels
        x = self.encode_to_decode(x)
        
        # use the decoder to decode
        x = self.decoder(decode_t, decode_pad, x, t, pad, use_memory=use_memory)
        
        return x

class CrossAttentionDecoder(Module):
    def __init__(self, config):
        super().__init__()
        
        # store config
        self.config = config
        
        # build the positional encodings
        self.spatial_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.spatial_tiles+config.global_tokens)
        self.temporal_position_encoding = LearnedPositionalEncoding(
            config.decoder_channels, config.max_sequence_length)
        
        # build the cross-attention block
        decoder_config = TransformerConfig.translate(
            config,
            blocks='decoder_blocks',
            channels='decoder_channels',
            residual_channels='decoder_residual_channels',
            num_heads='decoder_heads',
        )
        self.block = TransformerBlock(decoder_config)
        
        upsample_spatial_channels = (
            config.spatial_channels * config.upsample_h * config.upsample_w)
        
        # output
        self.norm = LayerNorm(config.decoder_channels)
        
        self.table_decoder = Linear(
            config.decoder_channels, upsample_spatial_channels)
        
        self.hand_decoder = Linear(
            config.decoder_channels, upsample_spatial_channels)
        
        #self.mode_decoder = Linear(
        #    config.decoder_channels, config.global_channels)
        global_head_spec = {
            'mode' : config.num_modes,
            'shape' : config.num_shapes,
            'color' : config.num_colors,
        }
        self.global_decoder = LinearMultiheadDecoder(
            config.decoder_channels,
            global_head_spec,
        )
    
    def forward(self, tq, pad_q, xk, tk, pad_k, use_memory=None):
        
        # use the positional encoding to generate the query tokens
        x_spatial = self.spatial_position_encoding.encoding
        x_temporal = self.temporal_position_encoding(tq)
        s, b, c = x_temporal.shape
        hw, c = x_spatial.shape
        xq = x_temporal.view(s, 1, b, c) + x_spatial.view(1, hw, 1, c)
        xq = xq.view(s*hw, b, c)
        tq = tq.view(s, 1, b).expand(s, hw, b).reshape(s*hw, b)
        pad_q = pad_q * hw
        
        # compute the mask
        mask = padded_causal_mask(tq, pad_q, tk, pad_k)
        
        # use the transformer block to compute the output
        x = self.block(xq, pad_k, xk=xk, mask=mask, use_memory=use_memory)
        
        # reshape the output
        uh = self.config.upsample_h
        uw = self.config.upsample_w
        uc = self.config.spatial_channels
        uhwc = uh*uw*uc
        x = x.view(s, hw, b, c)
        
        # split off the table tokens, upsample and reshape into a rectangle
        table_start = 0
        table_end = table_start + self.config.table_tiles
        table_x = self.table_decoder(x[:,table_start:table_end])
        th = self.config.table_tiles_h
        tw = self.config.table_tiles_w
        table_x = table_x.view(s, th, tw, b, uh, uw, uc)
        table_x = table_x.permute(0, 3, 6, 1, 4, 2, 5)
        table_x = table_x.reshape(s, b, uc, th*uh, tw*uw)
        
        # split off the hand tokens, upsample and reshape into a rectangle
        hand_start = table_end
        hand_end = hand_start + self.config.hand_tiles
        hand_x = self.hand_decoder(x[:,hand_start:hand_end])
        hh = self.config.hand_tiles_h
        hw = self.config.hand_tiles_w
        hand_x = hand_x.view(s, hh, hw, b, uh, uw, uc)
        hand_x = hand_x.permute(0, 3, 6, 1, 4, 2, 5)
        hand_x = hand_x.reshape(s, b, uc, hh*uh, hw*uw)
        
        # split off the global tokens
        global_start = hand_end
        global_end = global_start + self.config.global_tokens
        global_x = x[:,global_start:global_end].permute(0, 2, 1, 3)
        x = self.global_decoder(global_x)
        x['table'] = table_x
        x['hand'] = hand_x
        #mode_x = global_x['mode']
        #shape_x = global_x['shape']
        #color_x = global_x['color']
        
        return x

class BreakAndMakeTransformerInterface:
    def __init__(self, model, config):
        self.config = config
        self.model = model
    
    def observation_to_tensors(self, observation, pad):
        device = next(self.model.parameters()).device
        
        # process table tiles
        table_tiles, table_tyx, table_pad = batch_deduplicate_tiled_seqs(
            observation['workspace_color_render'],
            pad,
            self.config.tile_h,
            self.config.tile_w,
            background=102,
        )
        table_pad = torch.LongTensor(table_pad).to(device)
        table_tiles = default_tile_transform(table_tiles).to(device)
        table_t = torch.LongTensor(table_tyx[...,0]).to(device)
        table_yx = torch.LongTensor(
            table_tyx[...,1] *
            self.config.table_tiles_w +
            table_tyx[...,2],
        ).to(device)
        
        # processs hand tiles
        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_tiled_seqs(
            observation['handspace_color_render'],
            pad,
            self.config.tile_h,
            self.config.tile_w,
            background=102,
        )
        hand_pad = torch.LongTensor(hand_pad).to(device)
        hand_tiles = default_tile_transform(hand_tiles).to(device)
        hand_t = torch.LongTensor(hand_tyx[...,0]).to(device)
        hand_yx = torch.LongTensor(
            hand_tyx[...,1] *
            self.config.hand_tiles_w +
            hand_tyx[...,2] +
            self.config.table_tiles,
        ).to(device)
        
        # cat table and hand ties
        tile_x, tile_pad = cat_padded_seqs(
            table_tiles, hand_tiles, table_pad, hand_pad)
        tile_t, _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
        tile_yx, _ = cat_padded_seqs(table_yx, hand_yx, table_pad, hand_pad)
        
        # process tokens
        token_x = torch.LongTensor(observation['phase_switch']).to(device)
        s = numpy.max(pad)
        b, = pad.shape
        
        token_t = torch.arange(s).view(s, 1).expand(s, b).contiguous().to(
            device)
        token_pad = torch.LongTensor(pad).to(device)
        
        # step
        decode_t = torch.LongTensor(observation['step']).to(device)
        decode_pad = token_pad
        
        return (
            tile_x, tile_t, tile_yx, tile_pad,
            token_x, token_t, token_pad,
            decode_t, decode_pad,
        )
    
    def loss(self, x, pad, y, log=None, clock=None):
        pass
    
    def tensor_to_actions(self, x, env, mode='sample'):
        pass
    
    def visualize_episodes(self, epoch, episodes, visualization_directory):
        pass
    
    def eval_episodes(self, episodes, log, clock):
        pass

#class BlocksTransformerInterface:
#    def __init__(self, model, config):
#        self.config = config
#        self.model = model
#    
#    def observation_to_tensors(self, observation, pad):
#        # get the device
#        device = next(self.model.parameters()).device
#        
#        # process table tiles
#        table_tiles, table_tyx, table_pad = batch_deduplicate_from_masks(
#            observation['table_render'],
#            observation['table_tile_mask'],
#            observation['step'],
#            pad,
#        )
#        
#        table_pad = torch.LongTensor(table_pad).to(device)
#        table_tiles = default_tile_transform(table_tiles).to(device)
#        table_t = torch.LongTensor(table_tyx[...,0]).to(device)
#        table_yx = torch.LongTensor(
#            table_tyx[...,1] *
#            self.config.table_tiles_w +
#            table_tyx[...,2],
#        ).to(device)
#        
#        # processs hand tiles
#        hand_tiles, hand_tyx, hand_pad = batch_deduplicate_from_masks(
#            observation['hand_render'],
#            observation['hand_tile_mask'],
#            observation['step'],
#            pad,
#        )
#        
#        hand_pad = torch.LongTensor(hand_pad).to(device)
#        hand_tiles = default_tile_transform(hand_tiles).to(device)
#        hand_t = torch.LongTensor(hand_tyx[...,0]).to(device)
#        hand_yx = torch.LongTensor(
#            hand_tyx[...,1] *
#            self.config.hand_tiles_w +
#            hand_tyx[...,2] +
#            self.config.table_tiles,
#        ).to(device)
#        
#        # cat table and hand ties
#        tile_x, tile_pad = cat_padded_seqs(
#            table_tiles, hand_tiles, table_pad, hand_pad)
#        tile_t, _ = cat_padded_seqs(table_t, hand_t, table_pad, hand_pad)
#        tile_yx, _ = cat_padded_seqs(table_yx, hand_yx, table_pad, hand_pad)
#        
#        # process token x/t/pad
#        token_x = torch.LongTensor(observation['phase']).to(device)
#        token_t = torch.LongTensor(observation['step']).to(device)
#        token_pad = torch.LongTensor(pad).to(device)
#        
#        # process decode t/pad
#        decode_t = token_t
#        decode_pad = token_pad
#        
#        return (
#            tile_x, tile_t, tile_yx, tile_pad,
#            token_x, token_t, token_pad,
#            decode_t, decode_pad,
#        )
#    
#    def loss(self, x, pad, y, log=None, clock=None):
#        # split out the components of x
#        #x_table, x_hand, x_mode_shape_color = x
#        #num_shapes = len(self.config.block_shapes)
#        #num_colors = len(self.config.block_colors)
#        #s, b = x_mode_shape_color.shape[:2]
#        #x_mode = x_mode_shape_color[..., 0:6].view(s, b, 6)
#        #x_shape = x_mode_shape_color[..., 6:6+num_shapes].view(s, b, num_shapes)
#        #x_color = x_mode_shape_color[
#        #    ..., 6+num_shapes:6+num_shapes+num_colors].view(s, b, num_colors)
#        x_table, x_hand, x_mode, x_shape, x_color = x
#        
#        pad = torch.LongTensor(pad).to(x_mode.device)
#        pad_mask = make_padding_mask(pad, (s,b))
#        
#        # mode supervision
#        y_mode = torch.LongTensor(y['mode']).to(x_mode.device)
#        mode_loss = torch.nn.functional.cross_entropy(
#            x_mode.view(-1,6), y_mode.view(-1), reduction='none')
#        mode_loss = mode_loss.view(s,b) * ~pad_mask
#        mode_loss = mode_loss.mean() * self.config.mode_loss_weight
#        
#        # table supervision
#        h, w = x_table.shape[2:4]
#        table_entries = ((y_mode == 0) | (y_mode == 1)).view(-1)
#        x_table = x_table.view(s*b, h*w)[table_entries]
#        y_table_y = torch.LongTensor(y['table_cursor'][:,:,0])
#        y_table_x = torch.LongTensor(y['table_cursor'][:,:,1])
#        y_table = (y_table_y * w + y_table_x).view(-1)[table_entries]
#        y_table = y_table.to(x_table.device)
#        table_loss = torch.nn.functional.cross_entropy(x_table, y_table)
#        table_loss = table_loss * self.config.table_loss_weight
#        
#        # hand supervision
#        h, w = x_hand.shape[2:4]
#        hand_entries = (y_mode == 1).view(-1)
#        x_hand = x_hand.view(s*b, h*w)[hand_entries]
#        y_hand_y = torch.LongTensor(y['hand_cursor'][:,:,0])
#        y_hand_x = torch.LongTensor(y['hand_cursor'][:,:,1])
#        y_hand = (y_hand_y * w + y_hand_x).view(-1)[hand_entries]
#        y_hand = y_hand.to(x_hand.device)
#        hand_loss = torch.nn.functional.cross_entropy(x_hand, y_hand)
#        hand_loss = hand_loss * self.config.hand_loss_weight
#        
#        # shape supervision
#        shape_entries = (y_mode == 2).view(-1)
#        x_shape = x_shape.view(s*b, num_shapes)[shape_entries]
#        y_shape = torch.LongTensor(y['shape']).view(-1)[shape_entries]
#        y_shape = y_shape.to(x_shape.device)
#        shape_loss = torch.nn.functional.cross_entropy(x_shape, y_shape)
#        shape_loss = shape_loss * self.config.shape_loss_weight
#        
#        # color supervision
#        color_entries = (y_mode == 2).view(-1)
#        x_color = x_color.view(s*b, num_colors)[color_entries]
#        y_color = torch.LongTensor(y['color']).view(-1)[color_entries]
#        y_color = y_color.to(x_color.device)
#        color_loss = torch.nn.functional.cross_entropy(x_color, y_color)
#        color_loss = color_loss * self.config.color_loss_weight
#        
#        loss = mode_loss + table_loss + hand_loss + shape_loss + color_loss
#        
#        if log is not None:
#            log.add_scalar('train/mode_loss', mode_loss, clock[0])
#            log.add_scalar('train/table_loss', table_loss, clock[0])
#            log.add_scalar('train/hand_loss', hand_loss, clock[0])
#            log.add_scalar('train/shape_loss', shape_loss, clock[0])
#            log.add_scalar('train/color_loss', color_loss, clock[0])
#            log.add_scalar('train/total_loss', loss, clock[0])
#        
#        return loss
#    
#    def forward_rollout(self, terminal, *x):
#        use_memory = torch.BoolTensor(~terminal).to(x[0].device)
#        return self.model(*x, use_memory=use_memory)
#    
#    def tensor_to_actions(self, x, env, mode='sample'):
#        #x_table, x_hand, x_mode_shape_color = x
#        #num_shapes = len(self.config.block_shapes)
#        #num_colors = len(self.config.block_colors)
#        #s, b = x_mode_shape_color.shape[:2]
#        #assert s == 1
#        #x_mode = x_mode_shape_color[..., 0:6].view(b, 6)
#        #x_shape = x_mode_shape_color[..., 6:6+num_shapes].view(b, num_shapes)
#        #x_color = x_mode_shape_color[
#        #    ..., 6+num_shapes:6+num_shapes+num_colors].view(b, num_colors)
#        x_table, x_hand, x_mode, x_shape, x_color = x
#        
#        mode_action = categorical_or_max(x_mode, mode=mode).cpu().numpy()
#        shape_action = categorical_or_max(x_shape, mode=mode).cpu().numpy()
#        color_action = categorical_or_max(x_color, mode=mode).cpu().numpy()
#        
#        s, b, h, w, c = x_table.shape
#        x_table = x_table.view(b, 1, h, w)
#        table_y, table_x = categorical_or_max_2d(x_table, mode=mode)
#        table_y = table_y.cpu().numpy()
#        table_x = table_x.cpu().numpy()
#        
#        s, b, h, w, c = x_hand.shape
#        x_hand = x_hand.view(b, 1, h, w)
#        hand_y, hand_x = categorical_or_max_2d(x_hand, mode=mode)
#        hand_y = hand_y.cpu().numpy()
#        hand_x = hand_x.cpu().numpy()
#        
#        actions = []
#        for i in range(b):
#            action = BlocksEnv.no_op_action()
#            action['mode'] = mode_action[i]
#            action['shape'] = shape_action[i]
#            action['color'] = color_action[i]
#            action['table_cursor'] = numpy.array([table_y[i], table_x[i]])
#            action['hand_cursor'] = numpy.array([hand_y[i], hand_x[i]])
#            actions.append(action)
#        
#        return actions
#    
#    def visualize_episodes(self, epoch, episodes, visualization_directory):
#        num_seqs = min(
#            self.config.visualization_seqs, episodes.num_seqs())
#        for seq_id in tqdm.tqdm(range(num_seqs)):
#            seq_path = os.path.join(
#                visualization_directory, 'seq_%06i'%seq_id)
#            if not os.path.exists(seq_path):
#                os.makedirs(seq_path)
#            
#            seq = episodes.get_seq(seq_id)
#            seq_len = len_hierarchy(seq)
#            table_frames = seq['observation']['table_render']
#            hand_frames = seq['observation']['hand_render']
#            for frame_id in range(seq_len):
#                table_frame = table_frames[frame_id]
#                hand_frame = hand_frames[frame_id]
#                th, tw = table_frame.shape[:2]
#                hh, hw = hand_frame.shape[:2]
#                w = tw + hw
#                joined_image = numpy.zeros((th, w, 3), dtype=numpy.uint8)
#                joined_image[:,:tw] = table_frame
#                joined_image[th-hh:,tw:] = hand_frame
#                
#                frame_action = index_hierarchy(seq['action'], frame_id)
#                frame_mode = int(frame_action['mode'])
#                frame_shape_id = int(frame_action['shape'])
#                frame_color_id = int(frame_action['color'])
#                ty, tx = frame_action['table_cursor']
#                ty = int(ty)
#                tx = int(tx)
#                hy, hx = frame_action['hand_cursor']
#                hy = int(hy)
#                hx = int(hx)
#                
#                joined_image[ty*4:(ty+1)*4, tx*4:(tx+1)*4] = (0,0,0)
#                yy = th - hh
#                joined_image[
#                    yy+hy*4:yy+(hy+1)*4, tw+(hx)*4:tw+(hx+1)*4] = (0,0,0)
#                
#                mode_string = 'Mode: %s'%([
#                    'disassemble',
#                    'place',
#                    'pick-up',
#                    'make',
#                    'end',
#                    'no-op'][frame_mode])
#                shape_string = 'Shape: %s'%str(
#                    self.config.block_shapes[frame_shape_id])
#                color_string = 'Color: %s'%str(
#                    self.config.block_colors[frame_shape_id])
#                lines = (mode_string, shape_string, color_string)
#                joined_image = write_text(joined_image, '\n'.join(lines))
#                
#                frame_path = os.path.join(
#                    seq_path,
#                    'frame_%04i_%06i_%04i.png'%(epoch, seq_id, frame_id),
#                )
#                save_image(joined_image, frame_path)
