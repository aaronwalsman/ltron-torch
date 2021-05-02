import time

import torch

from ltron_torch.models.spatial import NerfSpatialEmbedding2D

class GraphStepModel(torch.nn.Module):
    '''
    '''
    def __init__(self,
            backbone,
            dense_heads,
            single_heads,
            add_spatial_embedding=False,
            decoder_channels=256):
        super(GraphStepModel, self).__init__()
        self.backbone = backbone
        self.add_spatial_embedding = add_spatial_embedding
        if self.add_spatial_embedding:
            self.spatial_embedding_layer = NerfSpatialEmbedding2D(
                    num_layers=1, channels=decoder_channels)
        self.dense_heads = torch.nn.ModuleDict(dense_heads)
        self.single_heads = torch.nn.ModuleDict(single_heads)
    
    def forward(self, x):
        x, xn = self.backbone(x)
        if len(self.single_heads):
            xs = torch.nn.functional.adaptive_avg_pool2d(xn[0], (1,1))
            xs = torch.flatten(xs, 1)
        if self.add_spatial_embedding:
            x = self.spatial_embedding_layer(x)
        head_features = {head_name : head_model(x)
                for head_name, head_model in self.dense_heads.items()}
        head_features.update({head_name : head_model(xs)
                for head_name, head_model in self.single_heads.items()})
        
        return head_features
