import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np
from heteroclients import Client_init
from dataset import dataSet_MLP, dataSet_CNN
from utils import *
from datetime import datetime
import os
from models import *
from args import set_args
import ipdb
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


def local_train(args, model, loader, epochs_resp, device, model_name, path):
    print("{} local training begin!...".format(model_name))
    model.train()
    ce = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 40], gamma=args.gamma)
    
    for epoch in range(epochs_resp):
        for i, (xi, yi) in enumerate(loader):
            if xi.shape[0] < 2:
                break
            xi = xi.to(torch.float).to(device)
            yi = yi.to(device)

            optimizer.zero_grad()
            *_, out = model(xi)
            loss = ce(out, yi)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 输出训练集上预测精度
        model.eval()  
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs = inputs.to(torch.float).to(device)
                labels = labels.to(device)
                outputs = model(inputs)[-1]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            train_acc = 100 * correct / total

        # 输出测试集上预测精度
        test_acc = test(model, test_loader, model_name)
        print ('Epoch [{}/{}], Loss: [{:.6f}], Train_acc: [{:.6f} %], Test_acc: [{:.6f} %]'
                .format(epoch+1, epochs_resp, loss.item(), train_acc, test_acc))
        f = open(os.path.join(path, '{}_local_log.txt'.format(model_name)), "a")
        f.write('Iteration [{}/{}] \n'.format(epoch+1, epochs_resp))
        f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
        f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
        f.close()


def test(model, loader, model_name):
    model.eval()  
    if model_name != 'global_model':
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, labels in loader:
                inputs = inputs.to(torch.float).to(device)
                labels = labels.to(device)
                outputs = model(inputs)[-1]
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            acc = 100 * correct / total
        return acc



args = set_args()
model_total = args.model
seed = args.seed
lr = args.lr
dataset_name = args.dataset
train_ratio = args.train_ratio
local_log_name = args.local_log_name
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
batch_size = args.batch_size

client = Client_init(args, dataset_name, model_total, train_ratio)
class_num = client.class_number             # 记录故障总类别数
label_client = client.label_client          # 记录每个类别在每个边端各有多少样本

time_log = datetime.now().strftime('%m-%d %H:%M')
path = os.path.join(local_log_name, time_log)
if not os.path.exists(path):
    os.makedirs(path)
for model_name in model_total:
    f = open(os.path.join(path, '{}_local_log.txt'.format(model_name)), "a")
    f.write('This is a log of results on local.py----Model:{model_name} \n'
                'time: {time}\n'.format(model_name=model_name, time=time_log))
    for k,v in sorted(vars(args).items()):
        f.write(k+'='+str(v)+'\n')
    f.close()

CNN_base = {'channel1':8, 'channel2':16, 'kernel1':2, 'kernel2':2, 'output':15}
MLP_base = {'input':52, 'hidden1':40, 'hidden2':20, 'output':15}

set_seed(seed)
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False


for i in range(len(model_total)):
    uname ='u_{0:03d}'.format(i)
    model_name = model_total[i]

    if 'CNN' in model_name and dataset_name == 'TE':
        model = CNN(MLP_base['input'],
                    CNN_base['channel1'] + (int(model_name[-1])-1)*4, CNN_base['kernel1'], \
                    CNN_base['channel2'] + (int(model_name[-1])-1)*4, CNN_base['kernel2'], \
                    CNN_base['output']).to(device)
    if 'MLP' in model_name and dataset_name == 'TE':
        model = MLP(MLP_base['input'], \
                    MLP_base['hidden1'] + (int(model_name[-1])-1), \
                    MLP_base['hidden2'] + (int(model_name[-1])-1), \
                    MLP_base['output']).to(device)
    if dataset_name == 'Ku':
        z_dim = 64
        hid_dim1 = 256
        ganout_dim = 430
        output_dim = 9
        y_dim = output_dim
    if dataset_name == 'TE':
        z_dim = args.z_dim
        hid_dim1 = args.hid_dim1
        ganout_dim = 52
        output_dim = 15
        y_dim = output_dim

    traindata = np.array(client.users[uname]['train']['x'])
    trainlabel = np.array(client.users[uname]['train']['y'])
    testdata = np.array(client.users[uname]['test']['x'])
    testlabel = np.array(client.users[uname]['test']['y'])
    labels = client.users[uname]['labels']
    traindata, testdata = z_score(traindata, testdata)
    
    if 'CNN' in model_name:
        trainset = dataSet_CNN(traindata, trainlabel)
        testset = dataSet_CNN(testdata, testlabel)
    if 'MLP' in model_name:
        trainset = dataSet_MLP(traindata, trainlabel)
        testset = dataSet_MLP(testdata, testlabel)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)

    epochs_resp = args.epochs * 50
    
    local_train(args, model, train_loader, epochs_resp, device, model_name, path)

