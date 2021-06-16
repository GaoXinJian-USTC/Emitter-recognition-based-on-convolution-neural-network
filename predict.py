from Backbone import MobileNetV3,customed_backbone,DFCNN_iff,LeNet
import os
import torch
import glob
from torch.autograd import Variable
from Dataset.npydata_radar import npyset
from torch import nn
import argparse
import csv

def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')
    parser.add_argument('--model_path', type=str, default="./out_put_model/model_acc_0.7276.pt")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_root', type=str, default="/data/gaoxinjian/datafolder/kemu3/test/IFF/3-IQ-norm", help='path to the dataset')
    # parser.add_argument('--data_root', type=str, default="/data/gaoxinjian/datafolder/Sample/IFF/3-IQ-norm", help='path to the dataset')
    parser.add_argument('--use_gpu', action="store_true", default=True,help="use cuda")
    parser.add_argument('--model', type=str, default="DFCNN")
    parser.add_argument("--res_dir",type=str,default="./9-3-2_.xls")
    args = parser.parse_args()
    return args

def predict(model,dataloader,use_gpu,res_dir):
    with open(res_dir,"w+") as f_w:
        writer = csv.writer(f_w)
        for iter, data in enumerate(dataloader['test'], 1):
            # print(data)
            xx, yy, p = data
            if use_gpu:
                xx, yy = xx.cuda(), yy.cuda()
            else:
                xx, yy = Variable(xx), Variable(yy)
            y_pred = model(xx)
            x1, pred = torch.max(y_pred.data, 1)
            print("{} -> {}".format(p,pred))
            writer.writerow([int(pred)])

def valid(model,dataloader,use_gpu,res_dir):
    corrects = 0
    with open(res_dir,"w+") as f_w:
        writer = csv.writer(f_w)
        for iter, data in enumerate(dataloader['test'], 1):
            # print(data)
            xx, yy, p = data
            if use_gpu:
                xx, yy = xx.cuda(), yy.cuda()
            else:
                xx, yy = Variable(xx), Variable(yy)
            y_pred = model(xx)
            x1, pred = torch.max(y_pred.data, 1)
            corrects += torch.sum(pred == yy.data)
            # print("{} -> {}  label: {}".format(p,int(pred),int(yy)))
            # if int(pred)==int(yy):
            #     corrects +=1
            # else:
            #     print("{} -> {}  label: {}".format(p,int(pred),int(yy)))
    print("Corrects :{} ".format(corrects))
            # writer.writerow([int(pred)])

if __name__ == '__main__':
    args = parse_args()
    test_path = glob.glob(args.data_root+"/*.npy")
    test_path = sorted(test_path,key=lambda x:int(x.split("/")[-1].split(".")[0].split("_")[-1]))
    # test_target = [int(l.split("/")[-2]) for l in test_path]
    test_target = [88 for l in test_path]
    test_set = npyset(split="test",target=test_target,paths=test_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    dataloader = {x: torch.utils.data.DataLoader(dataset=eval("{}_set".format(x)), batch_size=args.batch_size, shuffle=False) for x in
                ['test']}
    if args.use_gpu:
        if args.model == "MobileNetV3":
            net = MobileNetV3.MobileNetV3(num_classes=40,type="small")
            model = nn.DataParallel(net)
            model = model.cuda()
            print("**** Training with backbone mobilenet v3 ****")
        elif args.model == "DFCNN":
            model = DFCNN_iff.DFCNN(40)
            net = torch.load(args.model_path)
            model = torch.nn.DataParallel(model).cuda()
            model.load_state_dict(net)
            print("**** Testing with DFCNN backbone ****")
        elif args.model == "LeNet":
            net = LeNet.LeNet()
            model = nn.DataParallel(net)
            model = model.cuda()
            print("**** Training with LeNet backbone ****")
    else:
        print("no gpu can use!")
        exit(-1)
        
    predict(model,dataloader,args.use_gpu,args.res_dir)
    # valid(model,dataloader,args.use_gpu,args.res_dir)