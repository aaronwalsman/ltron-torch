import time

import torch

from ltron.gym.rollout_storage import parallel_deepmap

from ltron_torch.models.spatial import NerfSpatialEmbedding2D

class SeqModel(torch.nn.Module):
    
    def __init__(self,
        backbone,
        dense_heads,
        single_heads,
        global_heads,
        add_spatial_embedding=False,
        decoder_channels=256,
    ):
        super(SeqModel, self).__init__()
        self.backbone = backbone
        self.add_spatial_embedding = add_spatial_embedding
        if self.add_spatial_embedding:
            self.spatial_embedding_layer = NerfSpatialEmbedding2D(
                    num_layers=1, channels=decoder_channels)
        self.dense_heads = torch.nn.ModuleDict(dense_heads)
        self.single_heads = torch.nn.ModuleDict(single_heads)
        self.global_heads = torch.nn.ModuleDict(global_heads)
    
    def forward(self, x, seq_mask=None, padding_mask=None):
        x, x_single, x_global = self.backbone(
            x,
            seq_mask=seq_mask,
            padding_mask=padding_mask)
        s, b, c, h, w = x.shape
        x = x.view(s*b, c, h, w)
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
        head_features.update({
            head_name : head_model(x_global)
            for head_name, head_model in self.global_heads.items()
        })
        
        return head_features

class SeqCrossProductPoseModel(torch.nn.Module):
    
    def __init__(self, seq_model, feature_head_name, confidence_head_name):
        super(SeqCrossProductPoseModel, self).__init__()
        self.seq_model = seq_model
        self.feature_head_name = feature_head_name
        self.confidence_head_name = confidence_head_name
    
    def forward(self,
        x,
        seq_mask=None,
        padding_mask=None,
        confidence_mode='max'
    ):
        head_features = self.seq_model(x, padding_mask, seq_mask)
        dense_features = head_features[self.feature_head_name]
        confidence = head_features[self.confidence_head_name]
        s, b, c, h, w = confidence.shape
        flat_confidence = confidence.view(s, b, h*w)
        if confidence_mode == 'max':
            confidence_indices = torch.argmax(flat_confidence, dim=-1)
        elif confidence_mode == 'sample':
            pass
        
        import pdb
        pdb.set_trace()
