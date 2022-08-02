import numpy

import torch
from torch.nn import (
    Module, ModuleList, ModuleDict, Linear, Embedding, LayerNorm)

from ltron_torch.models.padding import compress_tensors

# raw data type embeddings
class NormalizedEmbedding(Module):
    def __init__(self, num_embeddings, channels):
        super().__init__()
        self.embedding = Embedding(num_embeddings, channels)
        self.norm = LayerNorm(channels)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.norm(x)
        return x

class TemporalEmbedding(NormalizedEmbedding):
    def __init__(self, observation_space, channels):
        super().__init__(observation_space.max_steps, channels)

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

def MultiScreenPixelEmbedding(Module):
    def __init__(
        self,
        space,
        channels,
        temporal_embedding,
        fine_shape,
    ):
        super().__init__()
        
        self.space = space
        
        # temporal embedding
        self.temporal_embedding = temporal_embedding
        
        # fine embedding
        self.fine_span = NameSpan(**{'screen':fine_shape, 'NONE':1})
        self.fine_embedding = NormalizedEmbedding(self.fine_span.n, channels)
        
        # coarse embedding
        self.coarse_span = NameSpan()
        for name in space.layout.keys():
            if name == 'NO_OP' or name == 'DESELECT':
                self.coarse_span.add_names(name, 1)
            else:
                screen_shape = space.layout.get_shape(name)
                assert all(s%f == 0 for s, f in zip(screen_shape, fine_shape))
                coarse_shape = [s//f for s, f in zip(screen_shape, fine_shape)]
                self.coarse_span.add_names(name, coarse_shape)
        self.coarse_embedding = NormalizedEmbedding(
            self.coarse_span.n, channels)
        
        self.sum_norm = LayerNorm(channels)
    
    def observation_to_tensors(observations, t, pad, device):
        coarse_x = []
        fine_x = []
        
        # doing this in a loop is not ideal, but it's fast
        for i in observations:
            nyxc = self.space.unravel(i)
            n = nyxc[0]
            if n == 'NO_OP' or n == 'DESELECT':
                coarse_x.append(self.coarse_span.ravel(n, 0))
                #fine_x.append(self.fine_span.ravel(
    
    def forward(self, coarse_x, fine_x, t, pad):
        out_x = self.temporal_embedding(t)
        out_x = out_x + self.coarse_embedding(coarse_x)
        out_x = out_x + self.fine_embedding(fine_x)
        out_x = self.sum_norm(out_x)
        
        return out_x

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
        super().__init__()
        
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

class DiscreteTemporalEmbedding(Module):
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

class MultiDiscreteTemporalEmbedding(Module):
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
