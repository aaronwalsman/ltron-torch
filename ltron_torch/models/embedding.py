import numpy

import torch
from torch.nn import Module, Linear, Embedding, Dropout, LayerNorm

class TileEmbedding(Module):
    def __init__(self, tile_h, tile_w, tile_c, channels, dropout):
        super().__init__()
        self.tile_linear = Linear(tile_h * tile_w * tile_c, channels)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        s, b, h, w, c = x.shape
        x = x.view(s, b, h*w*c)
        x = self.tile_linear(x)
        x = self.dropout(x)
        return x

class OldTokenEmbedding(Module):
    def __init__(self, vocabulary, channels, dropout):
        super().__init__()
        self.embedding = Embedding(vocabulary, channels)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        return x

class TokenEmbedding(Module):
    def __init__(self,
        observation_space,
        channels,
        dropout,
        token_embedding=None,
        token_norm=None,
        temporal_embedding=None,
        temporal_norm=None,
        max_time_steps=None
    ):
        super().__init__()
        
        self.token_embedding = token_embedding or Embedding(
            observation_space.total, channels)
        self.token_norm = token_norm or LayerNorm(channels)
        
        self.temporal_embedding = temporal_embedding or Embedding(
            max_time_steps, channels)
        self.temporal_norm = temporal_norm or LayerNorm(channels)
        
        self.dropout = Dropout(dropout)
    
    def observation_to_tensors(self, observation, t, pad, device):
        x = torch.LongTensor(observation).to(device)
        t = torch.LongTensor(t).to(device)
        pad = torch.LongTensor(pad).to(device)
        return {'x':x}, t, pad
    
    def forward(self, x, t, pad):
        x = self.token_embedding(x)
        x = self.token_norm(x)
        
        t = self.temporal_embedding(t)
        t = self.temporal_norm(t)
        
        x = self.dropout(x+t)
        return x


class AssemblyEmbedding(Module):
    def __init__(self,
        observation_space,
        channels,
        dropout,
        shape_embedding=None,
        shape_norm = None,
        color_embedding=None,
        color_norm = None,
        pose_embedding=None,
        pose_norm = None,
        instance_id_embedding=None,
        instance_id_norm=None,
        temporal_embedding=None,
        temporal_norm=None,
        max_time_steps=None,
    ):
        
        super().__init__()
        
        self.num_shapes = numpy.max(observation_space['shape'].high+1)
        self.num_colors = numpy.max(observation_space['shape'].high+1)
        
        # shape embedding/norm
        self.shape_embedding = shape_embedding or Embedding(
            self.num_shapes, channels)
        self.shape_norm = shape_norm or LayerNorm(channels)
        
        # color embedding/norm
        self.color_embedding = color_embedding or Embedding(
            self.num_colors, channels)
        self.color_norm = color_norm or LayerNorm(channels)
        
        # pose embedding/norm
        self.pose_embedding = pose_embedding or PoseEmbedding(
            channels, dropout=0.)
        self.pose_norm = pose_norm or LayerNorm(channels)
        
        # index_embedding/norm
        self.instance_id_embedding = instance_id_embedding or Embedding(
            observation_space.max_instances+1, channels)
        self.instance_id_norm = instance_id_norm or LayerNorm(channels)
        
        # temporal embedding/norm
        self.temporal_embedding = temporal_embedding or Embedding(
            max_time_steps, channels)
        self.temporal_norm = temporal_norm or LayerNorm(channels)
        
        # dropout
        self.dropout = Dropout(dropout)
    
    def observation_to_tensors(self, observation, t, pad, device):
        shape = observation['shape']
        color = observation['color']
        pose = observation['pose']
        
        s, b = shape.shape[:2]
        
        assembly_length = shape.shape[-1]
        nonzero_instances = shape != 0
        assembly_pad = nonzero_instances.sum(axis=(0,-1))
        max_instances = assembly_pad.max()
        
        s_coord, b_coord, i_coord = numpy.where(shape)
        j_coord = numpy.concatenate([numpy.arange(cc) for cc in assembly_pad])
        
        # shape
        compressed_shape = shape[s_coord, b_coord, i_coord]
        padded_shape = numpy.zeros((b, max_instances), dtype=numpy.long)
        padded_shape[b_coord, j_coord] = compressed_shape
        padded_shape = torch.LongTensor(padded_shape.transpose(1,0)).to(device)
        
        # color
        compressed_color = color[s_coord, b_coord, i_coord]
        padded_color = numpy.zeros((b, max_instances), dtype=numpy.long)
        padded_color[b_coord, j_coord] = compressed_color
        padded_color = torch.LongTensor(padded_color.transpose(1,0)).to(device)
        
        # pose
        compressed_pose = pose[s_coord, b_coord, i_coord]
        padded_pose = numpy.zeros((b, max_instances, 4, 4))
        padded_pose[b_coord, j_coord] = compressed_pose
        padded_pose = torch.FloatTensor(
            padded_pose.transpose(1,0,2,3)).to(device)
        
        # instance id
        padded_instance_id = numpy.zeros((b, max_instances))
        padded_instance_id[b_coord, j_coord] = i_coord
        padded_instance_id = torch.LongTensor(
            padded_instance_id.transpose(1,0)).to(device)
        
        # x
        x = {
            'shape':padded_shape,
            'color':padded_color,
            'pose':padded_pose,
            'instance_id':padded_instance_id,
        }
        
        # t
        assembly_t = t.reshape(s, b, 1).repeat(assembly_length, axis=-1)
        compressed_t = assembly_t[s_coord, b_coord, i_coord]
        padded_t = numpy.zeros((b, max_instances))
        padded_t[b_coord, j_coord] = compressed_t
        padded_t = torch.LongTensor(padded_t.transpose(1,0)).to(device)
        
        # pad
        assembly_pad = torch.LongTensor(assembly_pad).to(device)
        
        return x, padded_t, assembly_pad
    
    def forward(self, shape, color, pose, instance_id, t, pad):
        shape = self.shape_embedding(shape)
        shape = self.shape_norm(shape)
        
        color = self.color_embedding(color)
        color = self.color_norm(color)
        
        pose = self.pose_embedding(pose)
        pose = self.pose_norm(pose)
        
        instance_id = self.instance_id_embedding(instance_id)
        instance_id = self.instance_id_norm(instance_id)
        
        time_step = self.temporal_embedding(t)
        time_step = self.temporal_norm(time_step)
        
        return self.dropout(shape + color + pose + instance_id + time_step)

class PoseEmbedding(Module):
    def __init__(self, channels, dropout, translation_scale_factor=0.005):
        super().__init__()
        self.translation_scale_factor = translation_scale_factor
        self.linear = Linear(12, channels)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        # remove the 0,0,0,1 row if present
        *coords, row, cols = x.shape
        x = x[...,:3,:]
        x[...,-1] = x[...,-1] * self.translation_scale_factor
        x = x.reshape(*coords, 12)
        x = self.linear(x)
        x = self.dropout(x)
        return x
