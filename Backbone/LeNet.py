from torch import nn
from torch.nn import functional as F
import torch
  
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 48 * 48, 1024)
        self.fc2 = nn.Linear(1024, 40)
        
         
         
    def forward(self, x):
        x = F.relu(self.conv1(x))
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# import numpy as np
# net = LeNet()
# device = torch.device("cuda:0")
# print(torch.cuda.is_available())
# net = net.to(device)
# inx = np.random.randint(0,500,size=(8,1,200,200))
# print(inx.shape)
# inx = torch.tensor(inx).float().cuda()
# y = net(inx)
# print(y.shape)