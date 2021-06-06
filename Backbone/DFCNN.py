import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCNN(nn.Module):
    def __init__(self,num_classes = 40):
        super(DFCNN,self).__init__()
        # self.input = input
        self.conv1 = nn.Conv2d(1,16,7)
        self.conv2 = nn.Conv2d(kernel_size=7,in_channels=16,out_channels=16)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(kernel_size=7,in_channels=16,out_channels=32)
        self.conv4 = nn.Conv2d(kernel_size=7,in_channels=32,out_channels=32)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(41*41*32,2048)
        self.fc2 = nn.Linear(2048,40)
        # self.softmax = nn.Softmax(40)
        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv2(self.conv1(x))))
        x = self.maxpool2(F.relu(self.conv4(self.conv3(x))))
        # print(x.shape)
        x = x.view(-1,41*41*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
