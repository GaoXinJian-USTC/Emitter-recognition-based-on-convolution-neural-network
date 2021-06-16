from Backbone import MobileNetV3,customed_backbone,DFCNN_iff,LeNet
import os
import torch
import glob
from torch.autograd import Variable
from Dataset.npyset import npyset
import argparse
from torch import nn

def parse_args():
    parser = argparse.ArgumentParser(description='Train RODNet.')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--opt', type=str, default="sgd")
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--data_root', type=str, default="/data/gaoxinjian/datafolder/Sample/IFF/3-IQ-norm", help='path to the dataset')
    parser.add_argument('--use_gpu', action="store_true", default=True,help="use cuda")
    parser.add_argument('--model', type=str, default="DFCNN")
    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learn_rate * (0.5 ** (epoch // 60))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def val(phrase,use_gpu,optimizer,batch_size,loss_f,model,e):
    running_corrects=0.0
    running_loss=0
    for iter, data in enumerate(dataloader[phrase], 1):
            xx, yy = data
            if use_gpu:
                xx, yy = xx.cuda(), yy.cuda()
            else:
                xx, yy = Variable(xx), Variable(yy)
            y_pred = model(xx)
            x1, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, yy)
            if phrase == 'train':
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(pred == yy.data)
            if iter % 40 == 0 and phrase == 'train':
                print("Iter:{}  Train Loss:{:.4f}  lr: {:.4f}  Train ACC:{:.4f}".format(
                    iter, running_loss / (float)(iter),optimizer.param_groups[0]['lr'],running_corrects / (float)(batch_size * iter)))
    epoch_loss = running_loss * batch_size / (float)(len(eval("{}_path".format(phrase))))
    epoch_acc = running_corrects / (float)(len(eval("{}_path".format(phrase))))
    print("Epoch {} {} loss:{:.4f}  lr: {:.4f}  Acc:{:.4f}".format(e,phrase, epoch_loss, optimizer.param_groups[0]['lr'],epoch_acc))
    return epoch_acc


def train(model,dataloader,learn_rate,epoch,use_gpu,batch_size):
    
    loss_f = torch.nn.CrossEntropyLoss()
    if args.opt == "sgd":
        print("-"*20+"Using optimizer SGD"+"-"*20)
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    elif args.opt == "adam":
        print("-"*20+"Using optimizer Adam"+"-"*20)
        optimizer = torch.optim.Adam(model.parameters(),lr = learn_rate)
    max_val_acc = 0
    t=1
    for e in range(epoch):
        print("Epoch {} / {}".format(e, epoch - 1))
        adjust_learning_rate(optimizer,e+1)
        print("-"*25+"Training"+"-"*25)
        phrase = "train"
        model.train(True)
        running_loss = 0.0
        running_corrects = 0
        for iter, data in enumerate(dataloader["train"], 1):
            xx, yy = data
            if use_gpu:
                xx, yy = xx.cuda(), yy.cuda()
            else:
                xx, yy = Variable(xx), Variable(yy)
            y_pred = model(xx)
            x1, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, yy)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(pred == yy.data)
            if iter % 40 == 0 and phrase == 'train':
                print("Iter:{}  Train Loss:{:.4f}  lr: {:.4f}  Train ACC:{:.4f}".format(
                    iter, running_loss / (float)(iter),optimizer.param_groups[0]['lr'],running_corrects / (float)(batch_size * iter)))
        epoch_loss = running_loss * batch_size / (float)(len(eval("{}_path".format(phrase))))
        epoch_acc = running_corrects / (float)(len(eval("{}_path".format(phrase))))
        print("Epoch {} {} loss:{:.4f}  lr: {:.4f}  Acc:{:.4f}".format(e,phrase, epoch_loss, optimizer.param_groups[0]['lr'],epoch_acc))
        if t % 5 == 0 and t>=5:
            print("-"*25+"Validing"+"-"*25)
            phrase = "test"
            model.train(False)
            epoch_acc = val(phrase,use_gpu,optimizer,batch_size,loss_f,model,e)
            if epoch_acc > max_val_acc and phrase=='test':
                max_val_acc = epoch_acc
                file = glob.glob("*.pt")
                if file:
                    os.remove(file[0])
                torch.save(model.state_dict(),"model_acc_{:.4f}.pt".format(epoch_acc))
                print("Model is saved !")
        t+=1


if __name__ == '__main__':
    args = parse_args()
    train_path = glob.glob(args.data_root+"/*/train_*.npy")
    test_path = glob.glob(args.data_root+"/*/test_*.npy")
    train_target = [int(l.split("/")[-2]) for l in train_path]
    test_target = [int(l.split("/")[-2]) for l in test_path]
    train_set = npyset(split = "train",target=train_target,paths=train_path,)
    test_set = npyset(split="test",target=test_target,paths=test_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    dataloader = {x: torch.utils.data.DataLoader(dataset=eval("{}_set".format(x)), batch_size=args.batch_size, shuffle=True,num_workers=4 ) for x in
                ['train', 'test']}

    print("Total samples: Train set {} , Test set {} ".format(len(train_path),len(test_path)) )
    
    if args.use_gpu:
        if args.model == "MobileNetV3":
            net = MobileNetV3.MobileNetV3(num_classes=40,type="small")
            model = nn.DataParallel(net)
            model = model.cuda()
            print("**** Training with backbone mobilenet v3 ****")
        elif args.model == "DFCNN":
            net = DFCNN_iff.DFCNN(num_classes=40)
            model = nn.DataParallel(net)
            model = model.cuda()
            print("**** Training with DFCNN backbone ****")
        elif args.model == "LeNet":
            net = LeNet.LeNet()
            model = nn.DataParallel(net)
            model = model.cuda()
            print("**** Training with LeNet backbone ****")
        # elif args.model == "Resnet50":     
        #     # net = torch.(num_classes=40)
        #     # model = nn.DataParallel(net)
        #     # model = model.cuda()
        #     print("**** Training with DFCNN backbone ****")
    else:
        print("no gpu can use!")
        exit(-1)
        
    train(model,dataloader,args.learn_rate,args.epoch,args.use_gpu,args.batch_size)
