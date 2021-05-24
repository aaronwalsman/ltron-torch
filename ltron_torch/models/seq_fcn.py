import torch
import torchvision.models as tv_models

import ltron_torch.models.resnet as resnet
from ltron_torch.models.simple_fcn import SimpleDecoder
from ltron_torch.models.simple_backbone import SimpleBackbone
from ltron_torch.models.spatial_sequence_transformer import (
    SpatialSequenceTransformer,
    SequenceTransformer,
)

class SeqGlobal(torch.nn.Module):
    def __init__(self,
        pretrained=True,
        global_channels=256,
        single_channels=256,
        decoder_channels=256,
        backbone='resnet50',
    ):
        super(SeqGlobal, self).__init__()
        if backbone == 'simple':
            backbone = SimpleBackbone()
            encoder_channels = (512, 256, 128, 64)
        elif backbone == 'resnet18':
            backbone = tv_models.resnet18(pretrained=pretrained)
            encoder_channels = (512, 256, 128, 64)
            backbone = resnet.ResnetBackbone(backbone, fcn=True)
        elif backbone == 'resnet34':
            backbone = tv_models.resnet50(pretrained=pretrained)
            #encoder_channels = (512, 256, 128, 64)
            encoder_channels = (2048, 1024, 512, 256)
            backbone = resnet.ResnetBackbone(backbone, fcn=True)
        elif backbone == 'resnet50':
            backbone = tv_models.resnet50(pretrained=pretrained)
            encoder_channels = (2048, 1024, 512, 256)
            backbone = resnet.ResnetBackbone(backbone, fcn=True)
        self.encoder = backbone
        self.decoder = SimpleDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )
        
        self.global_q = torch.nn.Parameter(torch.zeros(1,global_channels,1,1))
        self.global_k = torch.nn.Conv2d(
            encoder_channels[0], global_channels, kernel_size=1)
        self.global_v = torch.nn.Conv2d(
            encoder_channels[0], global_channels, kernel_size=1)
        
        self.single_q = torch.nn.Parameter(torch.zeros(1,single_channels,1,1))
        self.single_k = torch.nn.Conv2d(
            encoder_channels[0], single_channels, kernel_size=1)
        self.single_v = torch.nn.Conv2d(
            encoder_channels[0], single_channels, kernel_size=1)
        
        self.global_linear = torch.nn.Linear(
            global_channels, encoder_channels[0])
        self.single_linear = torch.nn.Linear(
            single_channels, encoder_channels[0])
    
    def forward(self, x, seq_mask, padding_mask):
        s, b, c, h, w = x.shape
        x = x.view(s*b, c, h, w)
        xn = self.encoder(x)
        x0 = xn[0]
        
        global_k = self.global_k(x0)
        sb, c, h, w = global_k.shape
        global_a = torch.sum(global_k * self.global_q, dim=1)
        global_a = global_a.view(s, b, h, w)
        global_a = global_a.permute(1,0,2,3).reshape(b,1,s*h*w)
        global_a = torch.softmax(global_a, dim=-1)
        x_global = self.global_v(x0)
        x_global = x_global.view(s, b, c, h, w)
        x_global = x_global.permute(1,2,0,3,4).reshape(b,c,s*h*w)
        x_global = x_global * global_a
        x_global = torch.sum(x_global, dim=-1)
        
        single_k = self.single_k(x0)
        sb, c, h, w = single_k.shape
        single_a = torch.sum(single_k * self.single_q, dim=1).view(sb, 1, h*w)
        single_a = torch.softmax(single_a, dim=-1)
        x_single = self.single_v(x0).view(sb, c, h*w)
        x_single = x_single * single_a
        x_single = torch.sum(x_single, dim=-1).view(s, b, c)
        
        x_global_expand = self.global_linear(x_global).view(1, b, -1, 1, 1)
        x_single_expand = self.single_linear(x_single).view(s*b, -1, 1, 1)
        
        sb, c, h, w = x0.shape
        x0 = x0 + x_single_expand
        x0 = (x0.view(s, b, c, h, w) + x_global_expand).view(s*b, c, h, w)
        
        xn = (x0, *xn[1:])
        x = self.decoder(*xn)
        sb, c, h, w = x.shape
        x = x.view(s, b, c, h, w)
        
        return x, x_single, x_global

class SeqFCN(torch.nn.Module):
    def __init__(self,
        pretrained=True,
        transformer_channels=256,
        decoder_channels=256,
        compute_single=True,
        residual_transformer=False,
    ):
        super(SeqFCN, self).__init__()
        self.compute_single = compute_single
        self.residual_transformer=residual_transformer
        
        #backbone = tv_models.resnet50(pretrained=pretrained)
        #encoder_channels = (2048, 1024, 512, 256)
        #backbone = tv_models.resnet34(pretrained=pretrained)
        encoder_channels = (512, 256, 128, 64)
        backbone = resnet.ResnetBackbone(backbone, fcn=True)
        #backbone = SimpleBackbone()
        self.encoder = backbone
        #self.seq_transformer = SpatialSequenceTransformer(
            #channels=2048,
        #    channels=512,
        #    hidden_channels=decoder_channels,
        #)
        
        self.transformer_linear = torch.nn.Linear(
            encoder_channels[0], transformer_channels)
        
        self.seq_transformer = SequenceTransformer(
            channels=transformer_channels,
            hidden_channels=transformer_channels,
        )
        
        self.single_linear = torch.nn.Linear(
            transformer_channels, encoder_channels[0])
        self.global_linear = torch.nn.Linear(
            transformer_channels, encoder_channels[0])
        
        self.decoder = SimpleDecoder(
            #encoder_channels=(2048, 1024, 512, 256),
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )
    
    def forward(self, x, seq_mask=None, padding_mask=None):
        s, b, c, h, w = x.shape
        x = x.view(s*b, c, h, w)
        xn = self.encoder(x)
        x0 = xn[0]
        sb, c, h, w = x0.shape
        x_single = torch.nn.functional.adaptive_avg_pool2d(x0, 1)
        x_single = x_single.view(s, b, c)
        x_single = self.transformer_linear(x_single)
        x_single, x_global = self.seq_transformer(x_single)
        x_single_expand = self.single_linear(x_single).view(s*b, c, 1, 1)
        x_global_expand = self.global_linear(x_global).view(1, b, c, 1, 1)
        
        x0 = x0 + x_single_expand
        x0 = (x0.view(s, b, c, h, w) + x_global_expand).view(s*b, c, h, w)
        # go from here, add x_single, x_global to x[0]
        # upsample as normal
        # predict global transform
        # multiply global transform (these two steps should probably be bumped
        #   out a level in the abstraction layer, and will no longer be
        #   "generic"... oh well.
        
        '''
        x0 = xn[0].view(s, b, c, h, w)
        xt = self.seq_transformer(
            x0,
            seq_mask=seq_mask,
            padding_mask=padding_mask,
        )
        xt = xt.reshape(s*b, c, h, w)
        '''
        #if self.residual_transformer:
        #    x0 = x0 + xt
        #else:
        #    x0 = xt
        
        #if self.compute_single:
        #    x_single = torch.nn.functional.adaptive_avg_pool2d(x0, (1,1))
        #    x_single = torch.flatten(x_single, 1)
        #    x_single = x_single.view(s, b, -1)
        #else:
        #    x_single = None
        
        xn = (x0, *xn[1:])
        x = self.decoder(*xn)
        sb, c, h, w = x.shape
        x = x.view(s, b, c, h, w)
        
        return x, x_single, x_global
