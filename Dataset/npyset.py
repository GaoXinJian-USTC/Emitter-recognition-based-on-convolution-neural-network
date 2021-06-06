import numpy as np
from numpy.lib import split
import torch
from tqdm import tqdm
from torch.utils import data


class npyset(data.Dataset):
    
    def __init__(self,split,target,paths):
        self.paths = paths
        self.target = target
        self.split = split

    def __getitem__(self, index):
        path = self.paths[index]
        IQimage =  np.load(path)
        img_tensor = torch.tensor(IQimage,dtype=torch.float32)
        target = self.target[index]
        # img_tensor = img_tensor.permute(1,0,2) //改变通道顺序
        target = torch.tensor(target)
        if self.split == "mobilenet":
            img_tensor = torch.nn.functional.pad(img_tensor, (12, 12, 12, 12))
        return img_tensor,target
        

    def __len__(self):

        return len(self.paths)