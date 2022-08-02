import torch
from torch.nn import Module, ModuleDict, Dropout

from gym.spaces import Discrete

from ltron.config import Config
from ltron.gym.spaces import (
    MaskedTiledImageSpace,
    AssemblySpace,
    PhaseSpace,
    TimeStepSpace,
    MultiScreenPixelSpace,
    MultiScreenInstanceSnapSpace,
    SymbolicSnapSpace,
)

from ltron_torch.models.embedding import (
    TemporalEmbedding,
    build_shared_assembly_embeddings,
    build_shared_masked_tiled_image_embeddings,
    DiscreteTemporalEmbedding,
    MultiDiscreteTemporalEmbedding,
)

class AutoEmbeddingConfig(Config):
    channels = 768
    embedding_dropout = 0.1

class AutoEmbedding(Module):
    def __init__(self,
        config,
        observation_space,
        readout_layout,
    ):
        # Module super
        super().__init__()
        
        # store observation space and readout layout
        self.observation_space = observation_space
        self.readout_layout = readout_layout
        
        # find token generating elements of the observation space
        tile_shapes = set()
        embeddings = {}
        
        self.time_step_name = None
        self.time_step_space = None
        
        # build the temporal embedding
        for name, space in observation_space.items():
            if isinstance(space, TimeStepSpace):
                assert self.time_step_space is None
                self.time_step_name = name
                self.time_step_space = space
        
        assert self.time_step_space is not None
        self.temporal_embedding = TemporalEmbedding(
            self.time_step_space, config.channels)
        
        # build the readout embedding
        self.readout_embedding = DiscreteTemporalEmbedding(
            readout_layout.total,
            config.channels,
            self.temporal_embedding,
        )
        
        # build the tile embeddings
        tile_subspaces = {
            name:subspace for name, subspace in observation_space.items()
            if isinstance(subspace, MaskedTiledImageSpace)
        }
        if len(tile_subspaces):
            tile_embeddings = build_shared_masked_tiled_image_embeddings(
                tile_subspaces,
                config.channels,
                self.temporal_embedding,
            )
            embeddings.update(tile_embeddings)
        
        # build assembly embeddings
        assembly_subspaces = {
            name:subspace for name, subspace in observation_space.items()
            if isinstance(subspace, AssemblySpace)
        }
        if len(assembly_subspaces):
            assembly_embeddings = build_shared_assembly_embeddings(
                assembly_subspaces,
                config.channels,
                self.temporal_embedding,
            )
            embeddings.update(assembly_embeddings)
        
        # build other embeddings
        for name, space in observation_space.items():
            if isinstance(space, MultiScreenInstanceSnapSpace):
                embeddings[name] = MultiDiscreteTemporalEmbedding(
                    space,
                    config.channels,
                    self.temporal_embedding,
                )
            
            elif isinstance(space, TimeStepSpace):
                # do not generate a second embedding for the time step
                pass
            
            elif isinstance(space, Discrete):
                embeddings[name] = DiscreteTemporalEmbedding(
                    space.n,
                    config.channels,
                    self.temporal_embedding,
                )
        
        # build the final dropout layer
        self.dropout = Dropout(config.embedding_dropout)
        
        # auto
        self.embeddings = ModuleDict(embeddings)
    
    def observation_to_tensors(self, batch, seq_pad):
        device = next(iter(self.parameters())).device
        observation = batch['observation']
        
        # move the time_step to torch/device
        time_step = torch.LongTensor(
            observation[self.time_step_name]).to(device)
        s, b = time_step.shape[:2]
        
        # generate the tensors for each embedding
        auto_x = {}
        auto_t = {}
        auto_pad = {}
        for name, embedding in self.embeddings.items():
            name_obs = observation[name]
            a_x, a_t, a_pad = embedding.observation_to_tensors(
                name_obs,
                observation[self.time_step_name],
                seq_pad,
                device,
            )
            auto_x[name] = a_x
            auto_t[name] = a_t
            auto_pad[name] = a_pad
        
        # make the readout tokens
        readout_x = []
        readout_t = []
        readout_pad = []
        for name in self.readout_layout.keys():
            if name == 'PAD':
                continue
            index = self.readout_layout.ravel(name, 0)
            readout_x.append(torch.full_like(time_step, index))
            readout_t.append(time_step)
            readout_pad.append(torch.full((b,), s).cuda())
        
        # concatenate the readout tokens
        readout_x = torch.cat(readout_x, dim=0)
        readout_t = torch.cat(readout_t, dim=0)
        readout_pad = sum(readout_pad)
        assert 'readout' not in auto_x
        auto_x['readout'] = {'x':readout_x}
        auto_t['readout'] = readout_t
        auto_pad['readout'] = readout_pad
        
        return {'x':auto_x, 't':auto_t, 'pad':auto_pad}
    
    def forward(self, x, t, pad):
        out_x = {}
        out_t = {}
        out_pad = {}
        
        # observation-based embeddings
        for name, embedding in self.embeddings.items():
            try:
                x_n, t_n, pad_n = embedding(**x[name], t=t[name], pad=pad[name])
                x_n = self.dropout(x_n)
                out_x[name] = x_n
                out_t[name] = t_n
                out_pad[name] = pad_n
            except:
                print('Forwward failed while embedding: %s'%name)
                raise
        
        # readout tokens
        name = 'readout'
        out_x[name], out_t[name], out_pad[name] = self.readout_embedding(
            **x[name], t=t[name], pad=pad[name])
        
        # return
        return out_x, t, pad
