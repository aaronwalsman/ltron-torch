import torch
import torchvision.models.resnet

#import ltron_torch.models.spatial as spatial

class ResnetBackbone(torch.nn.Module):
    def __init__(self, resnet, *output_layers):
        super(ResnetBackbone, self).__init__()
        self.resnet = resnet
        del(self.resnet.fc) # remove the fc layer to free up memory
        self.output_layers = output_layers
    
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        f = {}
        f['layer0'] = self.resnet.maxpool(x)
        
        f['layer1'] = self.resnet.layer1(f['layer0'])
        f['layer2'] = self.resnet.layer2(f['layer1'])
        f['layer3'] = self.resnet.layer3(f['layer2'])
        f['layer4'] = self.resnet.layer4(f['layer3'])
        
        if 'pool' in self.output_layers:
            f['pool'] = self.resnet.avgpool(f['layer4'])
            f['pool'] = torch.flatten(f['pool'], 1)
        
        if not len(self.output_layers):
            return f
        else:
            return tuple(f[output_layer] for output_layer in self.output_layers)

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

def named_backbone(name, *output_layers, pretrained=False):
    resnet = getattr(torchvision.models.resnet, name)(pretrained=pretrained)
    return ResnetBackbone(resnet, *output_layers)

def named_encoder_channels(name):
    if '18' in name or '34' in name:
        return (512, 256, 128, 64)
    elif '50' in name or '101' in name or '152' in name:
        return (2048, 1024, 512, 256)
    else:
        raise NotImplementedError

'''
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
'''
