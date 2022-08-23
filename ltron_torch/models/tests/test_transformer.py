import torch

from ltron_torch.models.transformer import TransformerConfig, Transformer

class TransformerTest(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, config.channels)
        self.transformer = Transformer(config)
    
    def forward(self, x, t, pad, use_memory=None):
        x = self.embedding(x)
        x = self.transformer(x, t, pad, use_memory=use_memory)
        
        return x

config = TransformerConfig(
    blocks = 4,
    channels = 512,
    heads = 8
)

model = TransformerTest(config).cuda().eval()

x = torch.LongTensor([
    [ 1, 2, 4, 6],
    [ 2, 4, 8,12],
    [ 3, 6,12,18],
    [ 4, 8,16,24],
    [ 5,10, 0, 0],
    [ 6,12, 0, 0],
    [ 7,14, 0, 0],
    [ 8,16, 0, 0],
]).cuda()

t = torch.LongTensor([
    [ 1, 1, 1, 1],
    [ 2, 2, 2, 2],
    [ 3, 3, 3, 3],
    [ 4, 4, 4, 4],
    [ 5, 5, 0, 0],
    [ 6, 6, 0, 0],
    [ 7, 7, 0, 0],
    [ 8, 8, 0, 0],
]).cuda()

pad = torch.LongTensor([8,8,4,4]).cuda()

y = model(x, t, pad)[-1]

ys = []
for i in range(24):
    j = i % 8
    k = i % 4
    xx = x[[j,j,k,k],[0,1,2,3]].view(1,4)
    tt = t[[j,j,k,k],[0,1,2,3]].view(1,4)
    print(xx)
    print(tt)
    pp = torch.ones(4, dtype=torch.long).cuda()
    use_memory = torch.BoolTensor([j!=0,j!=0,k!=0,k!=0]).cuda()
    print(use_memory)
    print('----')
    yy = model(xx, tt, pp, use_memory=use_memory)
    ys.append(yy[-1])

yy = torch.cat(ys, axis=0)

dfirst = y - yy[:8]
match_first = torch.all(torch.abs(dfirst) < 0.0001, axis=-1)
print(match_first)

dlast = y - yy[-8:]
match_last = torch.all(torch.abs(dlast) < 0.0001, axis=-1)
print(match_last)

import pdb
pdb.set_trace()
