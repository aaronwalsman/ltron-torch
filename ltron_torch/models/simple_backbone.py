import torch
from torch.nn.functional import relu

class SimpleBackbone(torch.nn.Module):
    def __init__(self):
        super(SimpleBackbone, self).__init__()
        
        print('makin simple BB')
        
        # 256
        self.conv0 = torch.nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3)
        # 128
        self.conv1 = torch.nn.Conv2d(
            64, 64, kernel_size=3, stride=1, padding=1)
        self.mp1 = torch.nn.MaxPool2d(2, 2)
        # 64
        self.conv2 = torch.nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1)
        self.mp2 = torch.nn.MaxPool2d(2, 2)
        # 32
        self.conv3 = torch.nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1)
        self.mp3 = torch.nn.MaxPool2d(2, 2)
        # 16
        self.conv4 = torch.nn.Conv2d(
            256, 512, kernel_size=3, stride=1, padding=1)
        self.mp4 = torch.nn.MaxPool2d(2, 2)
        # 8
    
    def forward(self, x):
        x0 = self.conv0(x)
        x0 = relu(x0)
        
        x1 = self.conv1(x0)
        x1 = relu(x1, inplace=True)
        x1 = self.mp1(x1)
        
        x2 = self.conv2(x1)
        x2 = relu(x2, inplace=True)
        x2 = self.mp2(x2)
        
        x3 = self.conv3(x2)
        x3 = relu(x3, inplace=True)
        x3 = self.mp3(x3)
        
        x4 = self.conv4(x3)
        x4 = relu(x4, inplace=True)
        x4 = self.mp4(x4)
        
        return x4, x3, x2, x1
