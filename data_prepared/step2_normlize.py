import os
import json
import glob
import csv
from tqdm import tqdm

with open(r"/home/gaoxinjian/datafolder/Sample/IFF/3-IQ/max_min.json","r+") as f:
    json_ = f.readlines()
    data_dic = json.loads(json_[0])
   

file_root = r"/home/gaoxinjian/datafolder/Sample/IFF/3-IQ"
for i in range(40):
    data_path = os.path.join(file_root,str(i))
    files = glob.glob(data_path+"/*.csv")
    files = sorted(files,key=lambda x:int(x.split("/")[-1].split(".")[0]))
    for file in tqdm(files):
        key = '{}_{}'.format(i,file.split(".")[0].split("/")[-1])
        try:
            min_i,max_i,min_q,max_q = round(data_dic[key][0][0],4),round(data_dic[key][0][1],4),round(data_dic[key][1][0],4),round(data_dic[key][1][1],4)
        except:
            continue
        if file.endswith("csv"):
           with open(file,"r+") as csv_f:
               new_file = file.replace("3-IQ","3-IQ-norm")
               name = new_file.split("/")[-1]
               file_path = new_file.replace("/{}".format(name),"")
               print(file_path)
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