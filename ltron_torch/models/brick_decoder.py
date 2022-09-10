import torch
from torch.nn import Linear, Module, Sequential 

from ltron_torch.models.coarse_to_fine_decoder import CoarseToFineDecoder
from ltron_torch.models.mlp import linear_stack
from ltron_torch.models.heads import LinearMultiheadDecoder

class SimpleBrickDecoder(Module):
    def __init__(self,
        num_shapes,
        num_colors,
        channels=768,
        **kwargs,
    ):
        super().__init__()
        self.decoder = Sequential(
            linear_stack(
                2,
                channels,
                nonlinearity='gelu',
                final_nonlinearity=True,
                hidden_dropout=0.,
                out_dropout=0.,
            ),
            LinearMultiheadDecoder(
                channels,
                {
                    'shape':num_shapes,
                    'color':num_colors,
                }
            ),
        )

    def forward(self, x):
        x = self.decoder(x)
        #x = torch.cat((x['shape'], x['color']), dim=-1)
        b, c = x['shape'].shape
        x = x['shape'].view(b, c, 1) * x['color'].view(b, 1, c)
        x = x.reshape(b, c*c)
        
        return x

class ExplicitBrickDecoder(Module):
    def __init__(self,
        num_shapes,
        num_colors,
        channels=768,
        **kwargs,
    ):
        super().__init__()
        self.decoder = Sequential(
            linear_stack(
                2,
                channels,
                nonlinearity='gelu',
                final_nonlinearity=True,
                hidden_dropout=0.,
                out_dropout=0.,
            ),
            LinearMultiheadDecoder(
                channels,
                {
                    'shape_color':num_shapes * num_colors,
                }
            ),
        )

    def forward(self, x):
        x = self.decoder(x)['shape_color']
        
        return x

class BrickDecoder(CoarseToFineDecoder):
    def __init__(self,
        num_shapes,
        num_colors,
        channels=768,
        include_pose=False,
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
