from conspiracy import Config

import torch.nn as nn
import torch.nn.functional as F

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
        out = F.relu(x)
        out = self.conv2(out)
        return out + x

#class FirstFusionBlock(nn.Module):
#    def __init__(self, channels):
#        super().__init__()
#        self.res = ResidualConv(channels)
#    
#    def forward(self, r):
#        x = self.res(r)
#        x = F.interpolate(
#            x, scale_factor=2, mode='bilinear', align_corners=True)
#        return x

class FusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.res1 = ResidualConv(channels)
        self.res2 = ResidualConv(channels)
    
    def forward(self, f, r):
        x = f + self.res1(r)
        x = self.res2(x)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

def ReassembleBlock(in_channels, out_channels, scale_factor):
    seq = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1))
    if scale_factor < 0.99:
        seq.append(nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=round(1./scale_factor),
            stride=round(1./scale_factor),
            #padding=1,
        ))
    elif scale_factor > 1.01:
        seq.append(nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=int(scale_factor),
            stride=int(scale_factor),
        ))
    return seq

class UpsampleHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.dpt_channels
        c2 = config.dpt_channels // 2
        c4 = config.dpt_channels // 4
        self.conv1 = nn.Conv2d(c, c2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c2, c4, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        
        return x

class DPTConfig(Config):
    channels = 768
    dpt_channels = 256
    dpt_scale_factor = 16
    dpt_blocks = (2,5,8,11)
    include_dpt_upsample_head = True

class DPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # This will upsample 2**(dpt_blocks+1) times
        num_blocks = len(config.dpt_blocks)
        
        self.reassemble_blocks = nn.ModuleList()
        self.fusion_blocks = nn.ModuleList()
        for i in range(num_blocks):
            exponent = num_blocks-i
            if config.include_upsample_head:
                exponent += 1
            block_scale_factor = config.dpt_scale_factor / (2**exponent)
            
            reassemble = ReassembleBlock(
                config.channels, config.dpt_channels, block_scale_factor)
            fusion = FusionBlock(config.dpt_channels)
            
            self.reassemble_blocks.append(reassemble)
            self.fusion_blocks.append(fusion)
        
        if config.include_upsample_head:
            self.head = UpsampleHead(config)
        else:
            self.head = nn.Identity()
    
    def forward(self, *xs):
        f = 0.
        for i, xi in enumerate(reversed(xs)):
            r = self.reassemble_blocks[i](xi)
            f = self.fusion_blocks[i](f, r)
        
        #r1 = self.reassembly1(x1)
        #r2 = self.reassembly2(x2)
        #r3 = self.reassembly3(x3)
        #r4 = self.reassembly4(x4)
        #
        #f4 = self.fusion4(r4)
        #f3 = self.fusion3(f4, r3)
        #f2 = self.fusion2(f3, r2)
        #f1 = self.fusion1(f2, r1)
        
        x = self.head(f)
        
        return x
