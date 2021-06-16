
import numpy as np
import csv
import os
import glob
import json
from config import file_root_test
from tqdm import tqdm

file_root = file_root_test
def step1():
    data_dic = {}
    fl = open("errorfile.txt","w+") 
    files = glob.glob(file_root+"/*.csv")
    files = sorted(files,key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for file in tqdm(files):
        data_i = []
        data_q = []
        if file.endswith("csv"):
            with open(file,"r+") as csv_f:
                reader = csv.reader(csv_f)
                for line in reader:
                    data_i.append(float(line[0]))
                    data_q.append(float(line[1]))     
        if len(data_i) and len(data_q): 
            data_dic['{}'.format(file.split(".")[0].split("/")[-1])] = [(min(data_i),max(data_i)),(min(data_q),max(data_q))]
        else:
            fl.writelines(file.replace("./","")+"\n")
    fl.close()
    jstr = json.dumps(data_dic)       
    with open(file_root+"/max_min.json","w+") as f:
        f.writelines(jstr)

def step2():
    with open(file_root+"/max_min.json","r+") as f:
        json_ = f.readlines()
        data_dic = json.loads(json_[0])

    files = glob.glob(file_root+"/*.csv")
    files = sorted(files,key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for file in tqdm(files):
        key = '{}'.format(file.split(".")[0].split("/")[-1])
        try:
            min_i,max_i,min_q,max_q = round(data_dic[key][0][0],4),round(data_dic[key][0][1],4),round(data_dic[key][1][0],4),round(data_dic[key][1][1],4)
        except:
            print("error")
            continue
        if file.endswith("csv"):
            with open(file,"r+") as csv_f:
                new_file = file.replace("3-IQ","3-IQ-norm")
                name = new_file.split("/")[-1]
                file_path = new_file.replace("/{}".format(name),"")
                # print(file_path)
                if not os.path.exists(file_path):
                        os.makedirs(file_path)
                with open( new_file,"w+",newline='') as new_f:
                        reader = csv.reader(csv_f)
                        for line in reader:
                            data_i = (round(float(line[0]),4) - min_i) / (max_i - min_i)
                            data_q = (round(float(line[1]),4) - min_q) / (max_q - min_q)
                            if data_i <0 or data_q < 0:
                                print(file,round(float(line[0]),4),round(float(line[1]),4,min_i,min_q))
                            writer = csv.writer(new_f)
                            writer.writerow([data_i,data_q])

def step3():
    file_root_ = file_root+"-norm"
    data_dic = {}
    files = glob.glob(file_root_+"/*.csv")
    files = sorted(files,key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for nf,file in enumerate(files):
        I_list_train = []
        Q_list_train = []
        if file.endswith("csv"):
            m_i = np.zeros((150,300))
            m_q = np.zeros((150,300))
            with open(file,"r+") as csv_f:
                for n,line in enumerate(csv_f):
                    if n <150:
                        data_i = round(float(line.split(",")[0]),4)
                        data_q = round(float(line.split(",")[1]),4)
                        data_i = int(data_i*299)
                        data_q = int(data_q*299)
                        m_i[n][data_i] = data_i
                        m_q[n][data_q] = data_q
            I_list_train.append(m_i)
            Q_list_train.append(m_q)
            sum_l = np.concatenate((I_list_train,Q_list_train),axis=1)
            file_name = file.split("/")[-1]
            npy_path = file.replace(file_name,"train_{}.npy".format(file_name.replace(".csv","")))
            np.save(npy_path,sum_l)
       
print("step 1 start !")
step1()
print("step 2 start !")
step2()
print("step 3 start !")
step3()
