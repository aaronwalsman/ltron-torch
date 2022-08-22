import numpy
import torch

logits = torch.FloatTensor([5, 1, 0.2])
p = torch.softmax(logits, dim=-1).numpy()

nt = torch.LongTensor([0, 0, 0])
nn = numpy.zeros(3, dtype=int)

for _ in range(10000):
    z = -torch.log(-torch.log(torch.rand_like(logits)))
    #import pdb
    #pdb.set_trace()
    i = torch.topk(logits+z, 2).indices
    nt[i] += 1
    
    i = numpy.random.choice([0,1,2], size=2, p=p, replace=False)
    nn[i] += 1

print('gumbel:')
print(nt.numpy())
print('choice:')
print(nn)
