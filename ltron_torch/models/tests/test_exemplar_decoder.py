import torch
from torch.nn import Module, Embedding, Linear, Parameter
from torch.nn.functional import relu, mse_loss
from torch.optim import Adam

class ExemplarDecoder(Module):
    def __init__(self, exemplars):
        super().__init__()
        self.layer_0 = Linear(2, 1024)
        self.layer_1 = Linear(1024, 1024)
        self.layer_2 = Linear(1024, 1024)
        self.layer_3 = Linear(1024, 1024)
        self.layer_4 = Linear(1024, exemplars)
        
        self.exemplars = Parameter(torch.rand((2,exemplars)) * 2 - 1.)
    
    def forward(self, x):
        x = self.layer_0(x)
        x = relu(x)
        x = self.layer_1(x)
        x = relu(x)
        x = self.layer_2(x)
        x = relu(x)
        x = self.layer_3(x)
        x = relu(x)
        x = self.layer_4(x)
        t = 1. / x.shape[-1] ** 0.5
        x = torch.softmax(x*t, dim=-1)
        
        x = torch.einsum('bn,mn->bm', x, self.exemplars)
        return x

def train():
    model = ExemplarDecoder(1024).cuda()
    optimizer = Adam(model.parameters(), lr=3e-4)
    
    running_loss = 0.
    running_x_min = 0.
    running_y_min = 0.
    running_x_max = 0.
    running_y_max = 0.
    
    i = 0
    while True:
        try:
            x = (torch.rand((64,2)) * 2 - 1.).cuda()
            y = x.clone()
            x = model(x)
            
            loss = mse_loss(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            running_loss = running_loss * 0.99 + float(loss) * 0.01
            
            x_min = torch.min(model.exemplars[0])
            x_max = torch.max(model.exemplars[0])
            y_min = torch.min(model.exemplars[1])
            y_max = torch.max(model.exemplars[1])
            running_x_min = running_x_min * 0.99 + float(x_min) * 0.01
            running_x_max = running_x_max * 0.99 + float(x_max) * 0.01
            running_y_min = running_y_min * 0.99 + float(y_min) * 0.01
            running_y_max = running_y_max * 0.99 + float(y_max) * 0.01
            
            i += 1
            
            print('Steps: %i, Loss: %.04f, BBOX:(%.04f, %.04f, %.04f, %.04f)'%(
                i,
                running_loss,
                running_x_min,
                running_y_min,
                running_x_max,
                running_y_max,
            ))
        except KeyboardInterrupt:
            break
    
    print('DONE')

if __name__ == '__main__':
    train()
