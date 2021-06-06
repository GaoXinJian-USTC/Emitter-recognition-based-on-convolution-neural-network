import numpy as np
import csv
import os
import glob
import json
import cv2
from tqdm import tqdm
file_root = r"/home/gaoxinjian/datafolder/Sample/IFF/3-IQ-norm"
data_dic = {}

for i in tqdm(range(40)):
    data_path = os.path.join(file_root,str(i))
    files = glob.glob(data_path+"/*.csv")
    files = sorted(files,key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for nf,file in enumerate(files):
        I_list_train = []
        Q_list_train = []
        I_list_val = []
        Q_list_val = []
        if nf <800:
            if file.endswith("csv"):
                m_i = np.zeros((100,200))
                m_q = np.zeros((100,200))
                with open(file,"r+") as csv_f:
                    for n,line in enumerate(csv_f):
                        if n <100:
                            data_i = round(float(line.split(",")[0]),4)
                            data_q = round(float(line.split(",")[1]),4)
                            data_i = int(data_i*199)
                            data_q = int(data_q*199)
                            m_i[n][data_i] = data_i
                            m_q[n][data_q] = data_q
                I_list_train.append(m_i)
                Q_list_train.append(m_q)
                sum_l = np.concatenate((I_list_train,Q_list_train),axis=1)
                file_name = file.split("/")[-1]
                npy_path = file.replace(file_name,"train_{}.npy".format(file_name.replace(".csv","")))
                np.save(npy_path,sum_l)
        elif nf <990:
            if file.endswith("csv"):
                m_i = np.zeros((100,200))
                m_q = np.zeros((100,200))
                with open(file,"r+") as csv_f:
                    for n,line in enumerate(csv_f):
                        if n <100:
                            data_i = round(float(line.split(",")[0]),4)
                            data_q = round(float(line.split(",")[1]),4)
                            data_i = int(data_i*199)
                            data_q = int(data_q*199)
                            m_i[n][data_i] = data_i
                            m_q[n][data_q] = data_q
                I_list_val.append(m_i)
                Q_list_val.append(m_q)
                sum_l = np.concatenate((I_list_val,Q_list_val),axis=1)
                file_name = file.split("/")[-1]
                npy_path = file.replace(file_name,"test_{}.npy".format(file_name.replace(".csv","")))
                np.save(npy_path,sum_l)
