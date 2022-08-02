import torch
from torch.nn import Linear

from ltron_torch.models.coarse_to_fine_decoder import CoarseToFineDecoder

class BrickDecoder(CoarseToFineDecoder):
    def __init__(self,
        num_shapes,
        num_colors,
        channels=768,
        **kwargs,
    ):
        super().__init__(
            (num_shapes, num_colors),
            channels=channels,
            **kwargs
        )
        
    def forward(self, x, **kwargs):
        x = super().forward(x, **kwargs)
        b = x.shape[0]
        x = x.view(b, -1)
        
        return x
