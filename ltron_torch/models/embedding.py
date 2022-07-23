import numpy

import torch
from torch.nn import Module, ModuleList, Linear, Embedding, LayerNorm

from ltron_torch.models.padding import compress_tensors

# raw data type embeddings
class TemporalEmbedding(Module):
    def __init__(self, observation_space, channels):
        super().__init__()
        self.embedding = Embedding(observation_space.max_steps, channels)
        self.norm = LayerNorm(channels)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        return x

class TileEmbedding(Module):
    def __init__(self, tile_h, tile_w, tile_c, channels):
        super().__init__()
        self.tile_linear = Linear(tile_h * tile_w * tile_c, channels)
        self.tile_norm = LayerNorm(channels)
    
    def forward(self, x):
        s, b, h, w, c = x.shape
        x = x.view(s, b, h*w*c)
        x = self.tile_linear(x)
        x = self.tile_norm(x)
        return x

class PoseEmbedding(Module):
    def __init__(self, channels, translation_scale_factor=0.005):
        super().__init__()
        self.translation_scale_factor = translation_scale_factor
        self.linear = Linear(12, channels)
        self.norm = LayerNorm(channels)
    
    def forward(self, x):
        # remove the 0,0,0,1 row if present
        *coords, row, cols = x.shape
        x = x[...,:3,:]
        x[...,-1] = x[...,-1] * self.translation_scale_factor
        x = x.reshape(*coords, 12)
        x = self.linear(x)
        x = self.norm(x)
        return x

# observation space embeddings

def build_shared_masked_tiled_image_embeddings(
    spaces,
    channels,
    temporal_embedding,
):
    tile_shapes = set()
    for name, space in spaces.items():
        tile_shapes.add((space.tile_height, space.tile_width, space.channels))
    assert len(tile_shapes) == 1
    tile_shape = next(iter(tile_shapes))
    
    # get tile shape
    tile_embedding = TileEmbedding(*tile_shape, channels)
    
    embeddings = {}
    for name, space in spaces.items():
        embeddings[name] = MaskedTiledImageEmbedding(
            space,
            channels,
            temporal_embedding,
            tile_embedding=tile_embedding,
        )

class MaskedTiledImageEmbedding(Module):
    def __init__(self,
        space,
        channels,
        temporal_embedding,
        tile_embedding=None,
    ):
        # temporal embedding
        self.temporal_embedding = temporal_embedding
        
        # tile linear
        self.tile_embedding = tile_embedding or TileEmbedding(
            space.tile_height, space.tile_width, space.channels, channels)
        
        # spatial position encoding
        positions = space.mask_height * space.mask_width
        self.position_encoding = LearnedPositionalEncoding(channels, positions)
        self.spatial_norm = LayerNorm(channels)
        self.sum_norm = LayerNorm(channels)
    
    def observation_to_tensors(self, observation):
        pass
    
    def forward(x, p, t, pad):
        raise Exception('THIS NEEDS SCRUTINY FIRST, IS EVERYTHING BELOW GOOD?')
        x = self.tile_embedding(x)
        x = x + self.temporal_embedding(t)
        x = x + self.spatial_norm(self.position_encoding(p))
        x = self.sum_norm(x)
        
        return x, t, pad

class DiscreteEmbedding(Module):
    def __init__(self, num_embeddings, channels, temporal_embedding):
        super().__init__()
        self.temporal_embedding = temporal_embedding
        self.discrete_embedding = Embedding(num_embeddings, channels)
        self.norm = LayerNorm(channels)
        self.sum_norm = LayerNorm(channels)
    
    def observation_to_tensors(self, observation, t, pad, device):
        x = {'x':torch.LongTensor(observation).to(device)}
        t = torch.LongTensor(t).to(device)
        pad = torch.LongTensor(pad).to(device)
        
        return x, t, pad
    
    def forward(self, x, t, pad):
        out_x = self.temporal_embedding(t)
        out_x = out_x + self.norm(self.discrete_embedding(x))
        out_x = self.sum_norm(out_x)
        
        return out_x, t, pad

class MultiDiscreteEmbedding(Module):
    def __init__(self,
        observation_space,
        channels,
        temporal_embedding,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.temporal_embedding = temporal_embedding
        discrete_embeddings = []
        for n in observation_space.nvec:
            discrete_embeddings.append(Embedding(n, channels))
        self.discrete_embeddings = ModuleList(discrete_embeddings)
        self.norm = LayerNorm(channels)
        self.sum_norm = LayerNorm(channels)
        
    def observation_to_tensors(self, observation, t, pad, device):
        x = {'x':torch.LongTensor(observation).to(device)}
        t = torch.LongTensor(t).to(device)
        pad = torch.LongTensor(pad).to(device)
        return x, t, pad
    
    def forward(self, x, t, pad):
        out_x = self.temporal_embedding(t)
        for i, discrete_embedding in enumerate(self.discrete_embeddings):
            out_x = out_x + self.norm(discrete_embedding(x[...,i]))
        
        out_x = self.sum_norm(out_x)
        return out_x, t, pad

'''
class TileEmbedding(Module):
    def __init__(self, tile_h, tile_w, tile_c, channels):
        super().__init__()
        self.tile_linear = Linear(tile_h * tile_w * tile_c, channels)
    
    def forward(self, x):
        s, b, h, w, c = x.shape
        x = x.view(s, b, h*w*c)
        x = self.tile_linear(x)
        return x

class OldTokenEmbedding(Module):
    def __init__(self, vocabulary, channels):
        super().__init__()
        self.embedding = Embedding(vocabulary, channels)
    
    def forward(self, x):
        x = self.embedding(x)
        return x

class TokenEmbedding(Module):
    def __init__(self,
        observation_space,
        channels,
        temporal_embedding,
        token_embedding=None,
        token_norm=None,
        #temporal_embedding=None,
        #temporal_norm=None,
        #max_time_steps=None
    ):
        super().__init__()
        
        self.temporal_embedding = temporal_embedding
        
        self.token_embedding = token_embedding or Embedding(
            observation_space.total, channels)
        self.token_norm = token_norm or LayerNorm(channels)
        
        #self.temporal_embedding = temporal_embedding or Embedding(
        #    max_time_steps, channels)
        #self.temporal_norm = temporal_norm or LayerNorm(channels)
        
    def observation_to_tensors(self, observation, t, pad, device):
        x = torch.LongTensor(observation).to(device)
        t = torch.LongTensor(t).to(device)
        pad = torch.LongTensor(pad).to(device)
        return {'x':x}, t, pad
    
    def forward(self, x, t, pad):
        x = self.token_embedding(x)
        x = self.token_norm(x)
        
        #t = self.temporal_embedding(t)
        #t = self.temporal_norm(t)
        t = self.temporal_embedding(t)
        
        x = x+t
        return x
'''

'''
class MultiAssemblyEmebdding(Module):
    
    #Wraps multiple AssemblyEmbeddings so that they all share the same
    #shape, color, and pose embeddings
    
    def __init__(self,
        observation_spaces,
        channels,
        temporal_embedding,
    ):
        # figure out how many different shapes and colors each assebly provides
        # and make sure they all match
        num_shapes = set()
        num_colors = set()
        for name, observation_space in observation_spaces.items():
            num_shapes.add(numpy.max(observation_space['shape'].high+1))
            num_colors.add(numpy.max(observation_space['color'].high+1))
        assert len(num_shapes) == 1 and len(num_colors) == 1
        self.num_shapes = next(iter(num_shapes))
        self.num_colors = next(iter(num_colors))
        
        # shape embedding/norm
        self.shape_embedding = Embedding(self.num_shapes, channels)
        self.shape_norm = LayerNorm(channels)
        
        # color embedding/norm
        self.color_embedding = Embedding(self.num_colors, channels)
        self.color_norm = ColorNorm(channels)
        
        # pose embedding/norm
        self.pose_embedding = PoseEmbedding(channels)
        self.pose_norm = pose_norm or LayerNorm(channels)
        
        # index embedding/norm
        assembly_embeddings = {}
        for name, observation_space in observation_spaces.items():
            assembly_embeddings[name] = AssemblyEmbedding(
                observation_space,
                channels,
                temporal_embedding,
                shape_embedding=self.shape_embedding,
                shape_norm=self.shape_norm,
                color_embedding=self.color_embedding,
                color_norm=self.color_norm,
                pose_embedding=self.pose_embedding,
                pose_norm=self.pose_norm,
            )
        self.assembly_embeddings = ModuleDict(assembly_embeddings)
    
    def observation_to_tensors(self, observation):
        out = {}
        for name, assembly_embedding in self.assembly_embeddings.items():
            out[name] = assembly_embedding.observation_to_tensors(
                observation[name]
            )
        
        return out
    
    def forward(self, **assembly_x):
        out = {}
        for name, x in assembly_x.items():
            out[name] = self.assembly_embeddings[name](**x)
        
        return out
'''

def build_shared_assembly_embeddings(
    assembly_spaces,
    channels,
    temporal_embedding,
):
    # figure out how many different shapes and colors each assebly provides
    # and make sure they all match
    num_shapes = set()
    num_colors = set()
    for name, assembly_space in assembly_spaces.items():
        num_shapes.add(numpy.max(assembly_space['shape'].high+1))
        num_colors.add(numpy.max(assembly_space['color'].high+1))
    assert len(num_shapes) == 1 and len(num_colors) == 1
    num_shapes = next(iter(num_shapes))
    num_colors = next(iter(num_colors))
    
    # shape embedding/norm
    shape_embedding = Embedding(num_shapes, channels)
    shape_norm = LayerNorm(channels)
    
    # color embedding/norm
    color_embedding = Embedding(num_colors, channels)
    color_norm = LayerNorm(channels)
    
    # pose embedding/norm
    pose_embedding = PoseEmbedding(channels)
    pose_norm = LayerNorm(channels)
    
    # index embedding/norm
    return {
        name : AssemblyEmbedding(
            assembly_space,
            channels,
            temporal_embedding,
            shape_embedding=shape_embedding,
            shape_norm=shape_norm,
            color_embedding=color_embedding,
            color_norm=color_norm,
            pose_embedding=pose_embedding,
            pose_norm=pose_norm,
        )
        for name, assembly_space in assembly_spaces.items()
    }

class AssemblyEmbedding(Module):
    def __init__(self,
        observation_space,
        channels,
        temporal_embedding,
        shape_embedding=None,
        shape_norm = None,
        color_embedding=None,
        color_norm = None,
        pose_embedding=None,
        pose_norm = None,
        instance_id_embedding=None,
        instance_id_norm=None,
        sum_norm=None,
        #temporal_embedding=None,
        #temporal_norm=None,
        #max_time_steps=None,
    ):
        
        super().__init__()
        
        self.num_shapes = numpy.max(observation_space['shape'].high+1)
        self.num_colors = numpy.max(observation_space['color'].high+1)
        
        # temporal embedding
        self.temporal_embedding = temporal_embedding
        
        # shape embedding/norm
        self.shape_embedding = shape_embedding or Embedding(
            self.num_shapes, channels)
        self.shape_norm = shape_norm or LayerNorm(channels)
        
        # color embedding/norm
        self.color_embedding = color_embedding or Embedding(
            self.num_colors, channels)
        self.color_norm = color_norm or LayerNorm(channels)
        
        # pose embedding/norm
        self.pose_embedding = pose_embedding or PoseEmbedding(channels)
        self.pose_norm = pose_norm or LayerNorm(channels)
        
        # index_embedding/norm
        self.instance_id_embedding = instance_id_embedding or Embedding(
            observation_space.max_instances+1, channels)
        self.instance_id_norm = instance_id_norm or LayerNorm(channels)
        
        # sum_norm
        self.sum_norm = sum_norm or LayerNorm(channels)
        
        # temporal embedding/norm
        #self.temporal_embedding = temporal_embedding or Embedding(
        #    max_time_steps, channels)
        #self.temporal_norm = temporal_norm or LayerNorm(channels)
    
    def observation_to_tensors(self, observation, t, pad, device):
        shape = torch.LongTensor(observation['shape']).to(device)
        color = torch.LongTensor(observation['color']).to(device)
        pose = torch.FloatTensor(observation['pose']).to(device)
        
        s, b, n = shape.shape
        
        assembly_length = shape.shape[-1]
        nonzero_instances = shape != 0
        
        s_coord, b_coord, i_coord = torch.where(shape)
        instance_id = torch.arange(n, device=device).view(1,1,n).expand(s,b,n)
        t = torch.arange(s, device=device).view(s, 1, 1).expand(s,b,n)
        (padded_shape,
         padded_color,
         padded_pose,
         padded_instance_id,
         padded_t), assembly_pad = compress_tensors(
            (shape, color, pose, instance_id, t), s_coord, b_coord, i_coord)
        
        # x
        x = {
            'shape':padded_shape,
            'color':padded_color,
            'pose':padded_pose,
            'instance_id':padded_instance_id,
        }
        
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
        
        x = self.sum_norm(shape + color + pose + instance_id + time_step)
        
        return x, t, pad
