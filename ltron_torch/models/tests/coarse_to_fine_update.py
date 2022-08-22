import random
import math

import numpy

import torch
from torch.nn import Module, Parameter, Embedding
from torch.nn.functional import log_softmax, cross_entropy
from torch.optim import Adam

from splendor.image import save_image

from ltron.name_span import NameSpan

from ltron_torch.models.cursor_decoder import (
    #CoarseToFineCursorDecoderConfig,
    CoarseToFineVisualCursorDecoder,
)

def test():
    
    channels = 512
    
    span = NameSpan(main = [16,16])
    decoder = CoarseToFineCursorDecoder(
        span,
        (16,16),
        channels=channels,
        default_k=4,
    )
    
    targets = [(32, 45), (210, 112), (0, 119), (127,127)]
    
    class WrapperModule(Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding(len(targets), channels)
            self.decoder = decoder
        
        def forward(self, x, **kwargs):
            x = self.embedding(x)
            return self.decoder(x, **kwargs)
    
    wrapper = WrapperModule().cuda()
    wrapper.train()
    
    optimizer = Adam(wrapper.parameters(), lr=3e-4)
    
    bs = 64
    running_loss = 0.
    j = 0
    
    while True:
        xs = []
        ys = []
        for i in range(bs):
            y = torch.zeros(1, 256, 256, 1).cuda()
            i = random.randint(0, len(targets)-1)
            target = targets[i]
            yy = max(0, min(255, round(random.gauss(target[0], 3))))
            xx = max(0, min(255, round(random.gauss(target[1], 3))))
            y[0,yy,xx,0] = 1.
            xs.append(i)
            ys.append(y)
        x = torch.LongTensor(xs).view(bs).cuda()
        y = torch.cat(tuple(ys), dim=1).view(bs,256,256)
        
        x = wrapper(x)[:,1:]
        
        #x = x.view(bs, 16, 16, 16, 16).permute(0,1,3,2,4).reshape(bs, 256, 256)
        loss = -torch.sum(log_softmax(x, dim=-1) * y.view(bs,-1))/bs
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss = running_loss * 0.95 + float(loss) * 0.05
        
        print(j, running_loss)
        
        #if j % 100 == 0:
        x_im = torch.softmax(x.view(bs,-1)[0], dim=-1).detach().view(256,256)
        x_max = torch.max(x_im)
        x_im = x_im / x_max
        x_im = x_im.cpu().numpy() * 255
        x_im = x_im.astype(numpy.uint8)
        save_image(x_im, 'x_%08i.png'%j)
        j += 1

if __name__ == '__main__':
    test()
