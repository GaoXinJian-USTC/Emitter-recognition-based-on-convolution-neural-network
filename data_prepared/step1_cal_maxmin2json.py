import numpy as np
import csv
import os
import glob
import json
from config import file_root_train
data_dic = {}
fl = open("errorfile.txt","w+")
for i in range(40):
    data_path = os.path.join(file_root_train,str(i))
    files = glob.glob(data_path+"/*.csv")
    files = sorted(files,key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for file in files:
        data_i = []
        data_q = []
        if file.endswith("csv"):
           with open(file,"r+") as csv_f:
               reader = csv.reader(csv_f)
               for line in reader:
                   data_i.append(float(line[0]))
                   data_q.append(float(line[1]))     
        if len(data_i) and len(data_q): 
            data_dic['{}_{}'.format(i,file.split(".")[0].split("/")[-1])] = [(min(data_i),max(data_i)),(min(data_q),max(data_q))]
        else:
            fl.writelines(file.replace("./","")+"\n")

fl.close()
        
jstr = json.dumps(data_dic)       

with open(file_root_train+"/max_min.json","w+") as f:
    f.writelines(jstr)
