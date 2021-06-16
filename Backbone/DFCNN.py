import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCNN(nn.Module):
    def __init__(self,num_classes = 10):
        super(DFCNN,self).__init__()
        # self.input = input
        self.conv1 = nn.Conv2d(1,16,7)
        self.conv2 = nn.Conv2d(kernel_size=3,in_channels=16,out_channels=16)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(kernel_size=3,in_channels=16,out_channels=16)
        self.conv4 = nn.Conv2d(kernel_size=3,in_channels=16,out_channels=16)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.conv5 = nn.Conv2d(kernel_size=3,in_channels=16,out_channels=32)
        self.conv6 = nn.Conv2d(kernel_size=3,in_channels=32,out_channels=32)
        self.conv7 = nn.Conv2d(kernel_size=3,in_channels=32,out_channels=32)
        self.maxpool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*57*20,2048)
        self.fc2 = nn.Linear(2048,num_classes)
        # self.softmax = nn.Softmax(40)
        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv2(self.conv1(x))))
        x = self.maxpool2(F.relu(self.conv4(self.conv3(x))))
        x = self.maxpool3(F.relu(self.conv7(self.conv6(self.conv5(x)))))
        # print(x.shape)
        self.linear_features = x.shape[-1]*x.shape[-2]
        x = x.view(-1,self.linear_features*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DFCNN_ori(nn.Module):
    def __init__(self,num_classes = 40):
        super(DFCNN_ori,self).__init__()
        # self.input = input
        self.conv1 = nn.Conv2d(1,16,7)
        self.conv2 = nn.Conv2d(kernel_size=7,in_channels=16,out_channels=16)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(kernel_size=7,in_channels=16,out_channels=32)
        self.conv4 = nn.Conv2d(kernel_size=7,in_channels=32,out_channels=32)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*166*41,2048)
        self.fc2 = nn.Linear(2048,num_classes)
        # self.softmax = nn.Softmax(40)
        
    def forward(self, x):
        x = self.maxpool1(F.relu(self.conv2(self.conv1(x))))
        x = self.maxpool2(F.relu(self.conv4(self.conv3(x))))
        # print(x.shape)
        self.linear_features = x.shape[-1]*x.shape[-2]
        x = x.view(-1,self.linear_features*32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import numpy as np
# net = DFCNN_ori()
# device = torch.device("cuda:0")
# print(torch.cuda.is_available())
# net = net.to(device)
# inx = np.random.randint(0,200,size=(1,700,200))
# print(inx.shape)
# inx = torch.tensor(inx).float().cuda()
# y = net(inx)
# print(y.shape)