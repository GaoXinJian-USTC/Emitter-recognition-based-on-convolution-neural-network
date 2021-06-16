import numpy as np
from numpy.lib import split
import torch
from tqdm import tqdm
from torch.utils import data
import cv2

class npyset(data.Dataset):
    
    def __init__(self,split,target,paths):
        self.paths = paths
        self.target = target
        self.split = split

    def __getitem__(self, index):
        path = self.paths[index]
        IQimage =  np.load(path)
        # IQimage = IQimage.reshape((700,200,1))
        # IQimage = cv2.resize(IQimage,(500,200))
        # IQimage = IQimage.reshape((1,500,200))
        img_tensor = torch.tensor(IQimage,dtype=torch.float32)
        target = self.target[index]
        # img_tensor = img_tensor.permute(1,0,2) //改变通道顺序
        target = torch.tensor(target)
        if self.split == "mobilenet":
            img_tensor = torch.nn.functional.pad(img_tensor, (12, 12, 12, 12))
        return img_tensor,target,path
        

    def __len__(self):
        return len(self.paths)

# import glob
# data_root = "/data/gaoxinjian/datafolder/Sample/Radar/3-IQ-norm"
# train_path = glob.glob(data_root+"/*/train_*.npy")
# path = train_path[0]
# IQimage =  np.load(path)
# print(IQimage.shape)
# IQimage = IQimage.reshape((600,200,1))
# IQimage = cv2.resize(IQimage,(300,200))
# IQimage = IQimage.reshape((1,300,200))
# print(IQimage.shape)
