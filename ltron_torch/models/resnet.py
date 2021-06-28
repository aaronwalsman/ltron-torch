import torch

import ltron_torch.models.spatial as spatial

class ResnetBackbone(torch.nn.Module):
    def __init__(self, resnet, fcn=False, pool=False):
        super(ResnetBackbone, self).__init__()
        self.resnet = resnet
        self.fcn = fcn
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        
        if self.fcn:
            return x4, x3, x2, x1
        else:
            return x4

def replace_fc(resnet, num_classes):
    fc = resnet.fc
    resnet.fc = torch.nn.Linear(
            fc.in_features, num_classes).to(fc.weight.device)

def replace_conv1(resnet, input_channels):
    conv1 = resnet.conv1
    resnet.conv1 = torch.nn.Conv2d(
            input_channels, conv1.out_channels,
            kernel_size=(7,7),
            stride=(2,2),
            padding=(3,3),
            bias=False).to(conv1.weight.device)

def make_spatial_attention_resnet(resnet, shape, do_spatial_embedding=True):
    device = resnet.fc.weight.device
    backbone = ResnetBackbone(resnet)
    in_channels = resnet.conv1.in_channels
    x = torch.zeros(
            1, in_channels, shape[0], shape[1]).to(resnet.conv1.weight.device)
    with torch.no_grad():
        x = backbone(x)
        _, channels, h, w = x.shape
    
    layers = []
    if do_spatial_embedding:
        layers.append(
                spatial.AddSpatialEmbedding((h,w), channels).to(device))
    layers.append(spatial.SpatialAttention2D(channels).to(device))
    resnet.avgpool = torch.nn.Sequential(*layers)
