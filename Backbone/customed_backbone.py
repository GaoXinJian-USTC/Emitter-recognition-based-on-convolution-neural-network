import torch
import torch.nn as nn
import torch.nn.functional as F


class customed_backbone(nn.Module):
    def __init__(self,num_classes = 40):
        super(customed_backbone,self).__init__()
        # self.input = input
        self.conv1 = nn.Conv2d(1,8,7)
        self.conv2 = nn.Conv2d(kernel_size=7,in_channels=8,out_channels=8)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(kernel_size=7,in_channels=8,out_channels=16)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(44*44*16,1024)
        self.fc2 = nn.Linear(1024,40)
        # self.softmax = nn.Softmax(40)
        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv2(self.conv1(x))))
        x = self.maxpool2(F.relu(self.conv3(x)))
        x = x.view(-1,44*44*16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    





# model = MobileNetV3(type='small')
# print(model)

# input = torch.randn(1, 1, 224, 224)
# out = model(input)
# print(out.shape)
# import numpy as np
# net = Simple_backbone_()
# device = torch.device("cuda:0")
# print(torch.cuda.is_available())
# net = net.to(device)
# inx = np.random.randint(0,200,size=(8,1,200,200))
# print(inx.shape)
# inx = torch.tensor(inx).float().cuda()
# y = net(inx)
# print(y.shape)
