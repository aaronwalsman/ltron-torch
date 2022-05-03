import torch
from torch.nn import Module, Dropout, ReLU, GELU

def make_nonlinearity(nonlinearity):
    if nonlinearity == 'relu':
        nonlinearity = ReLU
    elif nonlinearity == 'gelu':
        nonlinearity = GELU
    return nonlinearity()

class ResidualBlock(Module):
    def __init__(self, mlp, Norm=None):
        super().__init__()
        self.mlp = mlp
    
    def forward(self, x):
        x = x + self.mlp(x)

def stack(
    Layer,
    num_layers,
    in_channels,
    hidden_channels=None,
    out_channels=None,
    first_norm=False,
    Norm=None,
    nonlinearity='relu',
    final_nonlinearity=False,
    hidden_dropout=None,
    out_dropout=None,
    **kwargs,
):
    hidden_channels = hidden_channels or in_channels
    out_channels = out_channels or in_channels
    
    try:
        hidden_channels = [int(hidden_channels)] * (num_layers-1)
    except TypeError:
        assert len(hidden_channels) == num_layers-1
    
    features = [in_channels] + hidden_channels + [out_channels]
    
    layers = []
    if first_norm:
        assert Norm is not None
        layers.append(Norm())
    
    for i, (in_f, out_f) in enumerate(zip(features[:-1], features[1:])):
        layers.append(Layer(in_f, out_f, **kwargs))
        
        if i != num_layers-1:
            if Norm is not None:
                layers.append(Norm())
            if hidden_dropout is not None:
                layers.append(Dropout(hidden_dropout))
            if nonlinearity is not None:
                layers.append(make_nonlinearity(nonlinearity))
        
        if out_dropout is not None:
            layers.append(Dropout(out_dropout))
        
        if final_nonlinearity:
            layers.append(make_nonlinearity(nonlinearity))
    
    return torch.nn.Sequential(*layers)

def linear_stack(*args, **kwargs):
    return stack(torch.nn.Linear, *args, **kwargs)

def conv2d_stack(*args, kernel_size=1, **kwargs):
    return stack(torch.nn.Conv2d, *args, kernel_size=kernel_size, **kwargs)

def cross_product_concat(xa, xb):
    sa, ba, ca = xa.shape
    sb, bb, cb = xb.shape
    assert ba == bb
    
    xa = xa.view(sa, 1, ba, ca).expand(sa, sb, ba, ca)
    xb = xb.view(1, sb, bb, cb).expand(sa, sb, bb, cb)
    x = torch.cat((xa, xb), dim=-1)
    
    return x

class Conv2dStack(torch.nn.Module):
    def __init__(
        self,
        num_layers,
        in_features,
        hidden_features,
        out_features,
    ):
        super(Conv2dStack, self).__init__()
        layer_features = (
            [in_features] + [hidden_features] * (num_layers-1) + [out_features])
        layers = []
        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(torch.nn.Conv2d(in_f, out_f, 1))
            layers.append(torch.nn.ReLU())
        
        # remove the final relu
        self.layers = torch.nn.Sequential(*layers[:-1])
    
    def forward(self, x):
        return self.layers(x)

class LinearStack(torch.nn.Module):
    def __init__(self, num_layers, in_features, hidden_features, out_features):
        super(LinearStack, self).__init__()
        layer_features = (
                [in_features] +
                [hidden_features] * (num_layers-1) +
                [out_features])
        layers = []
        for in_f, out_f in zip(layer_features[:-1], layer_features[1:]):
            layers.append(torch.nn.Linear(in_f, out_f))
            layers.append(torch.nn.ReLU())
        
        # remove the final relu
        self.layers = torch.nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.layers(x)

