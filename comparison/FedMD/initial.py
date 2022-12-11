import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np
from heteroclients import Client_init
from dataset import dataSet_MLP, dataSet_CNN
from utils import *
from datetime import datetime
import os
import torch.nn.functional as F
from models import *
from args import set_args
import ipdb
from cloud import cloud


args = set_args()
seed = args.seed
batch_size = args.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
train_ratio = args.train_ratio
iterations = args.iterations
lr = args.lr
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset
model_total = args.model
log_name = args.log_name

CNN_base = {'channel1':8, 'channel2':16, 'kernel1':2, 'kernel2':2, 'output':15}
MLP_base = {'input':52, 'hidden1':40, 'hidden2':20, 'output':15}

set_seed(seed)
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False

init = True
public_num = args.public_num    # 表示公共数据集每一类样本的数量，乘以类别数即为公共数据集的整体样本量
client = Client_init(args, dataset_name, train_ratio, public_num, init)
# 完成私有数据分配和公共数据加载
# 私有数据及其标签都是列表，但公有数据及其标签都是array
# client.public['data'] (7500, 52)  client.public['label'] (7500,)
class_num = client.class_number             # 记录故障总类别数
label_client = client.label_client          # 记录每个类别在每个边端各有多少样本


def train_public(args, model, loader, validloader, device, optimizer, model_name):
    print("{} train on public data begining!...".format(model_name))
    model.train()
    ce = nn.CrossEntropyLoss()

    loss_epochs = []
    train_accs = []
    test_accs = []
    for epoch in range(args.public_epochs):
        loss_epoch = 0
        for i, (xi, yi) in enumerate(loader):
            if xi.shape[0] < 2:
                break
            xi = xi.to(torch.float).to(device)
            yi = yi.to(device)

            optimizer.zero_grad()
            *_, out = model(xi)
            loss = ce(out, yi)
            loss.backward()
            loss_epoch += loss.item()
            optimizer.step()
        loss_epoch = loss_epoch / len(loader)
        loss_epochs.append(loss_epoch)

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
            train_accs.append(train_acc)
            print ('| Epoch [{}/{}] | Loss: [{:.4f}] | Accuracy [{:4f}%]'
                .format(epoch+1, args.public_epochs, loss_epoch, train_acc))
        
        # 输出测试集上预测精度
        test_acc = test(model, validloader, model_name)
        test_accs.append(test_acc)
        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('Epoch [{}/{}] \n'.format(epoch+1, args.epochs))
        f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
        f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
        f.close()

    return train_accs, test_accs, loss_epochs


def train_private(args, model, loader, validloader, device, optimizer, model_name):
    print("{} train on private data begining!...".format(model_name))
    model.train()
    ce = nn.CrossEntropyLoss()

    loss_epochs = []
    train_accs = []
    test_accs = []
    for epoch in range(args.epochs):
        loss_epoch = 0
        for i, (xi, yi) in enumerate(loader):
            if xi.shape[0] < 2:
                break
            xi = xi.to(torch.float).to(device)
            yi = yi.to(device)

            optimizer.zero_grad()
            *_, out = model(xi)
            loss = ce(out, yi)
            loss.backward()
            loss_epoch += loss.item()
            optimizer.step()
        loss_epoch = loss_epoch / len(loader)
        loss_epochs.append(loss_epoch)

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
            train_accs.append(train_acc)
            print ('| Epoch [{}/{}] | Loss: [{:.4f}] | Accuracy [{:4f}%]'
                .format(epoch+1, args.epochs, loss_epoch, train_acc))
        
        # 输出测试集上预测精度
        test_acc = test(model, validloader, model_name)
        test_accs.append(test_acc)
        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('Epoch [{}/{}] \n'.format(epoch+1, args.epochs))
        f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
        f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
        f.close()
    torch.save(model, './models/{}.pth'.format(model_name))

    return train_accs, test_accs, loss_epochs


def test(model, loader, model_name):
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
        acc = 100 * correct / total
        print('Test Accuracy of {} Model: {} %'.format(model_name, acc))
    return acc


######################################################################################
# prepare for log record
######################################################################################
#region
time_log = datetime.now().strftime('%m-%d %H:%M')
path = os.path.join(log_name, time_log)
if not os.path.exists(path):
    os.makedirs(path)
for model_name in model_total:
    f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
    f.write('This is a log of results on cnn_mlp_fuse.py----Model:{model_name} \n'
                'time: {time}\n'.format(model_name=model_name, time=time_log))
    for k,v in sorted(vars(args).items()):
        f.write(k+'='+str(v)+'\n')
    f.close()
#endregion

publicdata = client.public['data']
publiclabel = client.public['label']
public_validdata = client.public_valid['data']
public_validlabel = client.public_valid['label']
publicdata, public_validdata = z_score(publicdata, public_validdata)
print('public data shape:', publicdata.shape)
print('public validate data shape:', public_validdata.shape)
##################################################################################
# Initialize client models
# First train on public dataset and then private dataset
# Finally save all client models
# This initialization can be done only once
##################################################################################
public_logits = torch.zeros((publicdata.shape[0], class_num)).to(device)
for i in range(len(model_total)):
    uname ='u_{0:03d}'.format(i)
    model_name = model_total[i]

    # model inilization
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

    # train on public dataset
    # prepare public data ---- load data and pre-process (normalize)
    if 'CNN' in model_name:
        publicset = dataSet_CNN(publicdata, publiclabel)
        publicset_valid = dataSet_CNN(public_validdata, public_validlabel)
    if 'MLP' in model_name:
        publicset = dataSet_MLP(publicdata, publiclabel)
        publicset_valid = dataSet_MLP(public_validdata, public_validlabel)
    public_loader = DataLoader(publicset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=False)
    public_validloader = DataLoader(publicset_valid, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    # public train and test
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accs, test_accs, loss_epochs = train_public(args, model, public_loader, public_validloader, device, optimizer, model_name)
    
    plt.figure()
    x = np.arange(len(loss_epochs))
    plt.plot(x, loss_epochs)
    plt.savefig('./fig/Epoch{}_trainloss_publicdata.png'.format(args.public_epochs))
    plt.figure()
    plt.plot(x, train_accs, c='r', marker='o')
    plt.plot(x, test_accs, c='b', marker='+')
    plt.savefig('./fig/Epoch{}_trainvalidacc_publicdata.png'.format(args.public_epochs))

    # train on private dataset
    # prepare private data ---- load data and pre-process (normalize)
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

    # public train and test
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accs, test_accs, loss_epochs = train_private(args, model, train_loader, test_loader, device, optimizer, model_name)
    ipdb.set_trace()

    # f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
    # f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
    # f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
    # f.close()

    # distill logits on public data
    for _, (input, label) in enumerate(public_loader):
        inputs = inputs.to(torch.float).to(device)
        labels = labels.to(device)
        outputs = model(inputs)[-1]
        













