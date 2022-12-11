####################################################################
# FedMD
####################################################################
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
device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
dataset_name = args.dataset
model_total = args.model
log_name = args.log_name

CNN_base = {'channel1':8, 'channel2':16, 'kernel1':2, 'kernel2':2, 'output':15}
MLP_base = {'input':52, 'hidden1':40, 'hidden2':20, 'output':15}

set_seed(seed)
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False

init = False
public_num = args.public_num    # 表示公共数据集每一类样本的数量，乘以类别数即为公共数据集的整体样本量
client = Client_init(args, dataset_name, train_ratio, public_num, init)
# 完成私有数据分配和公共数据加载
# 私有数据及其标签都是列表，但公有数据及其标签都是array
# client.public['data'] (7500, 52)  client.public['label'] (7500,)
class_num = client.class_number             # 记录故障总类别数
label_client = client.label_client          # 记录每个类别在每个边端各有多少样本


def train(args, model, loader, device, optimizer, model_name, init_flag):
    print("{} train begining!...".format(model_name))
    model.train()
    total_step = len(loader)
    ce = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
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

            if i%100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: [{:.4f}]'
                .format(epoch+1, args.epochs, i+1, total_step, loss.item()))

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
        print('Train Accuracy of {} Model: {} %'.format(model_name, 100 * correct / total))
    # torch.save(model.state_dict(), './model/model.pth')
    return model, 100 * correct / total

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
            print('Test Accuracy of {} Model: {} %'.format(model_name, acc))
        return acc

    if model_name == 'global_model':
        with torch.no_grad():
            acc = 0
            for ld in loader:
                correct = 0
                total = 0
                predicted_set, labels_set = [], []
                for inputs, labels in ld:
                    if len(inputs.shape) == 3:
                        inputs = inputs.squeeze(dim=1)
                    inputs = inputs.to(torch.float).to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)[-1]
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    predicted_set.append(predicted)
                    labels_set.append(labels)
                acc += 100 * correct / total
            acc = acc / len(loader)
            print('Test Accuracy of {} Model: {} %'.format(model_name, acc))
        return acc, predicted_set, labels_set


# prepare for log record
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

testloaders = []
# cnn and mlp collaboration
for iter in range(iterations):
    print("Iter [{}/{}]".format(iter+1, iterations))
    init_flag = False
    models = {}

    # 是否进行学习率衰减
    if args.decay == 1:
        if iter >= 0 and iter < 9:
            lr = args.lr
        elif iter >= 9 and iter < 19:
            lr = args.lr * args.gamma
        elif iter >= 19:
            lr = args.lr * args.gamma**2
        print("Local Learning Rate:{}".format(lr))
    else:
        lr = args.lr
    
    # clients
    for i in range(len(model_total)):
        uname ='u_{0:03d}'.format(i)
        model_name = model_total[i]

        if iter == 0:
            init_flag = True

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
            cloud_gen = GEN(z_dim, y_dim, hid_dim1, ganout_dim).to(device)
        else:
            model = model_last[model_name]

        # prepare data ---- load data and pre-process (normalize)
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
        testloaders.append(test_loader)

        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('Iteration [{}/{}] \n'.format(iter+1, iterations))
        f.close()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model_trained, train_acc = train(args, model, train_loader, device, optimizer, model_name, init_flag)

        test_acc = test(model_trained, test_loader, model_name)

        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
        f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
        f.close()

        models[model_name] = model_trained
    model_last = models

    # cloud
    cloud_gen, models = cloud(args, models, attention_last, prototype_last, cloud_gen, z_dim, iter, class_num, label_client, device)
    model_last = models

    i = 0
    for name, model in models.items():
        test_acc = test(model, testloaders[i], name)
        print ('| Model [{}] | Test Accuracy: [{:.4f}] % |'.format(name, test_acc))
        f = open(os.path.join(path, '{}_log.txt'.format(name)), "a")
        f.write('test accuracy on {} model: {}\n'.format(name, test_acc))
        f.close()
        i = i + 1
    
    # plot distribution of cloud generated samples
    plot_generate_sample(args, cloud_gen, testdata, testlabel, iter, device, time_log)