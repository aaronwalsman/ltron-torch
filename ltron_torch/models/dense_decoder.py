import torch.nn as nn
import torch.nn.functional as F

from ltron_torch.models.film import FILMLayer

'''
DPT
'''

class ResidualConv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
    
    def forward(self, x):
        out = F.relu(x)
        out = self.conv1(out)
        out = F.relu(out)
        out = self.conv2(out)
        return out + x

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res1 = ResidualConv(channels)
        self.res2 = ResidualConv(channels)
    
    def forward(self, xf, xr):
        #output = xs[0]
        #if len(xs) == 2:
        #    output = output + self.res1(xs[1])
        #output = self.res2(output)
        x = self.res1(xr)
        x = x + xf
        x = self.res2(x)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)
        
        return x

def ReassemblyBlock1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4),
    )

def ReassemblyBlock2(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
    )

def ReassemblyBlock3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)

def ReassemblyBlock4(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
        nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=2, padding=1),
    )

class LayerNorm2d(nn.Module):
    '''
    Bro this suuuuuuucks
    '''
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(*args, **kwargs)
    
    def forward(self, x):
        x = x.permute(0,2,3,1)
        x = self.ln(x)
        x = x.permute(0,3,1,2)
        return x

class Head(nn.Module):
    def __init__(self, config, upscale=True):
        super().__init__()
        self.upscale=upscale
        #c4 = config.channels//4
        #c8 = config.channels//8
        #c16 = config.channels // 16
        c = config.dpt_channels
        c2 = config.dpt_channels // 2
        c4 = config.dpt_channels // 4
        #self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, c2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c2, c4, kernel_size=3, padding=1)
        #self.norm2 = LayerNorm2d(c4)
    
    def forward(self, x):
        #x = self.norm1(x)
        x = self.conv1(x)
        if self.upscale:
            x = F.interpolate(
                x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        #x = self.norm2(x)
        #x = F.relu(x)
        
        return x

class DenseDecoder(nn.Module):
    def __init__(self, config, upscale=True, include_head=True):
        super().__init__()
        
        '''
        self.conv1 = nn.Conv2d(
            in1, out1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(
            in2, out2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            in3, out3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(
            in4, out4, kernel_size=3, stride=1, padding=1, bias=False)
        '''
        
        self.reassembly1 = ReassemblyBlock1(
            config.channels, 256) #config.channels//4)
        self.reassembly2 = ReassemblyBlock2(
            config.channels, 256) #config.channels//2)
        self.reassembly3 = ReassemblyBlock3(
            config.channels, 256) #config.channels)
        self.reassembly4 = ReassemblyBlock4(
            config.channels, 256) #config.channels)
        
        self.fusion1 = FusionBlock(256) #config.channels//4)
        self.fusion2 = FusionBlock(256) #config.channels//2)
        self.fusion3 = FusionBlock(256) #config.channels)
        self.fusion4 = FusionBlock(256) #config.channels)
        
        #c1 = config.channels // 16
        #c2 = config.channels // 8
        #c3 = config.channels // 4
        #c4 = config.channels // 2
        
        if include_head:
            self.head = Head(config, upscale)
        else:
            self.head = nn.Identity()
    
    def forward(self, x1, x2, x3, x4):
        r1 = self.reassembly1(x1)
        r2 = self.reassembly2(x2)
        r3 = self.reassembly3(x3)
        r4 = self.reassembly4(x4)
        
        f4 = self.fusion4(0., r4)
        f3 = self.fusion3(f4, r3)
        f2 = self.fusion2(f3, r2)
        f1 = self.fusion1(f2, r1)
        
        x = self.head(f1)
        
        return x

class FILMDenseDecoder(nn.Module):
    def __init__(self, config, upscale=True, include_head=True):
        super().__init__()
        
        self.film1 = FILMLayer(config.channels, 256, dim=1)
        self.film2 = FILMLayer(config.channels, 256, dim=1)
        self.film3 = FILMLayer(config.channels, 256, dim=1)
        self.film4 = FILMLayer(config.channels, 256, dim=1)
        
        self.reassembly1 = ReassemblyBlock1(
            config.channels, 256)
        self.reassembly2 = ReassemblyBlock2(
            config.channels, 256)
        self.reassembly3 = ReassemblyBlock3(
            config.channels, 256)
        self.reassembly4 = ReassemblyBlock4(
            config.channels, 256)
        
        self.fusion1 = FusionBlock(256)
        self.fusion2 = FusionBlock(256)
        self.fusion3 = FusionBlock(256)
        self.fusion4 = FusionBlock(256)
        
        if include_head:
            self.head = Head(config, upscale)
        else:
            self.head = nn.Identity()
    
    def forward(self, x, x1, x2, x3, x4):
        r1 = self.reassembly1(x1)
        r2 = self.reassembly2(x2)
        r3 = self.reassembly3(x3)
        r4 = self.reassembly4(x4)
        
        f4 = self.fusion4(0., r4)
        f4 = self.film4(x, f4)
        f3 = self.fusion3(f4, r3)
        f3 = self.film3(x, f3)
        f2 = self.fusion2(f3, r2)
        f2 = self.film2(x, f2)
        f1 = self.fusion1(f2, r1)
        f1 = self.film1(x, f1)
        
        x = self.head(f1)
        
        return x
