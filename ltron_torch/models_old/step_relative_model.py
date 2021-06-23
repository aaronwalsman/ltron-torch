import time

import torch

from ltron.gym.rollout_storage import parallel_deepmap

from ltron_torch.models.spatial import NerfSpatialEmbedding2D

class StepRelativeModel(torch.nn.Module):
    
    def __init__(self,
            backbone,
            dense_heads,
            single_heads,
            add_spatial_embedding=False,
            decoder_channels=256):
        super(SeqModel, self).__init__()
        self.backbone = backbone
        self.add_spatial_embedding = add_spatial_embedding
        if self.add_spatial_embedding:
            self.spatial_embedding_layer = NerfSpatialEmbedding2D(
                    num_layers=1, channels=decoder_channels)
        self.dense_heads = torch.nn.ModuleDict(dense_heads)
        self.single_heads = torch.nn.ModuleDict(single_heads)
    
    def forward(self, x, seq_mask=None, padding_mask=None):
        s, b, c, h, w = x.shape
        x = x.view(x*b, c, h, w)
        x, x_single = self.backbone(x)
        if self.add_spatial_embedding:
            x = self.spatial_embedding_layer(x)
        head_features = {
            head_name : head_model(x)
            for head_name, head_model in self.dense_heads.items()
        }
        def fix_shape(a):
            return a.view(s, b, *a.shape[1:])
        head_features = parallel_deepmap(fix_shape, head_features)
        head_features.update({
            head_name : head_model(x_single)
            for head_name, head_model in self.single_heads.items()
        })
        
        return head_features
