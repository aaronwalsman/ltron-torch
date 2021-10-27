from torch.nn import Module, ModuleDict, Linear

class MultiheadDecoder(Module):
    def __init__(self, in_channels, heads):
        super(MultiheadDecoder, self).__init__()
        if isinstance(heads, str):
            heads = [h.split(':') for h in heads.split(',') if ':' in h]
            heads = {k.strip():int(v.strip()) for k,v in heads}
        self.heads = heads
        self.head_layers = ModuleDict({
            head_name : Linear(in_channels, head_channels)
            for head_name, head_channels in heads.items()
        })
    
    def forward(self, x):
        return {
            head_name : layer(x)
            for head_name, layer in self.head_layers.items()
        }
