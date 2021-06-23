import torch
import torchvision.models as tv_models

import ltron_torch.models.resnet as resnet
from ltron_torch.models.simple_fcn import SimpleDecoder

class TransformerModel(torch.nn.Module):
    def __init__(self,
        pretrained=True,
        decoder_channels=256,
        transformer_layers=2,
        transformer_heads=6,
        transformer_dropout=0.5,
    ):
        super(SimpleFCN, self).__init__()
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dropout = transformer_dropout
        backbone = tv_models.resnet50(pretrained=pretrained)
        backbone = resnet.ResnetBackbone(backbone, fcn=True)
        self.encoder = backbone
        self.decoder = SimpleDecoder(
            encoder_channels=(2048, 1024, 512, 256),
            decoder_channels=decoder_channels
        )
        
        self.frame_positional_encoding = PositionalEncoding(...)
        self.seq_positional_encoding = PositionalEncoding(...)
        
        transformer_layer = TransformerEncoderLayer(
            256, self.transformer_heads, 256, self.transformer_dropout)
        self.transformer_encoder = TransformerEncoder(
            transformer_layer, self.transformer_layers)
        
        self.init_weights()
    
    def forward(self, x):
        xn = self.encoder(xn)
        x = self.decoder(*xn)
        
        return x, xn
    
    def generate_subsequent_mask(self, steps):
        pass
    
    def init_weights(self):
        init_range = 0.1
        self.transformer_encoder.weight.data.uniform_(-init_range, init_range)
