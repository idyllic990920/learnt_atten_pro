#################################################################################
# This code is designed for new aggregation in Cloud - learnable attention idea #
#################################################################################
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
from collections import Counter
from adversarial import adversarial
from fedoptimizer import FedZKTl2Optimizer
import warnings
warnings.filterwarnings("ignore")


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

client = Client_init(args, dataset_name, model_total, train_ratio)
class_num = client.class_number             # 记录故障总类别数
label_client = client.label_client          # 记录每个类别在每个边端各有多少样本
write_to_excel(label_client)


def train(args, model, loader, device, model_name, init_flag):
    print("{} train begining!...".format(model_name))
    model.train()
    total_step = len(loader)
    ce = nn.CrossEntropyLoss()
    
    # if init_flag == True:
    #     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # else:
    #     optimizer = FedZKTl2Optimizer(model.parameters(), lr=args.lr, beta=args.beta)

    optimizer = FedZKTl2Optimizer(model.parameters(), lr=args.lr, beta=args.beta)
    
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

def attention_and_prototype(model, loader, labels, device): 
    # extract atten and proto
    attens, protos = {}, {}
    for item in labels:
        attens[item] = []
        protos[item] = []
    attention_mat, prototype_mat = {}, {}

    model.eval()
    for i, (xi, yi) in enumerate(loader):
        xi = xi.to(torch.float).to(device)

        a, p, *_ = model(xi)

        yi = yi.numpy()
        for i in range(len(yi)):
            attens[yi[i]].append(a[i, :].data.unsqueeze(dim=0).cpu().numpy())
            protos[yi[i]].append(p[i, :].data.unsqueeze(dim=0).cpu().numpy())

    # attention
    attention_mat['data'] = {}
    for key,value in attens.items():
        value = np.vstack(value)
        atten = np.mean(value, axis=0)
        # attention normalization
        if np.linalg.norm(atten.reshape(52)) != 0:
            attention_mat['data'][key] = atten.reshape(52) / np.linalg.norm(atten.reshape(52))
        else:
            attention_mat['data'][key] = atten.reshape(52)
    attention_mat['label'] = labels

    # prototype
    prototype_mat['data'] = {}
    for key,value in protos.items():
        value = np.vstack(value)
        proto = np.mean(value, axis=0)
        # prototype without normalization
        prototype_mat['data'][key] = proto.reshape(52)
    prototype_mat['label'] = labels

    return attention_mat, prototype_mat, 


# prepare for log record
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


testloaders = []
# cnn and mlp collaboration
for iter in range(iterations):
    print("Iter [{}/{}]".format(iter+1, iterations))

    init_flag = False
    prototype_matrix, attention_matrix = [], []
    models = {}

    # clients
    for i in range(len(model_total)):
        uname ='u_{0:03d}'.format(i)
        model_name = model_total[i]

        if iter == 0:
            init_flag = True
            prototype_last = []
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

        # 最好的办法还是在这里生成数据，然后加入真实数据集构成训练集
        if iter != 0:
            z_noise = torch.randn(traindata.shape[0], args.z_dim).to(device)
            y = torch.arange(y_dim).repeat(traindata.shape[0]//y_dim+1)[:traindata.shape[0]].to(device)
            y_hot = F.one_hot(y)
            fake = cloud_gen(z_noise, y_hot).cpu().detach().numpy()
            y = y.cpu().numpy()

            traindata = np.vstack((traindata, fake))
            trainlabel = np.concatenate((trainlabel, y))

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

        model_trained, train_acc = train(args, model, train_loader, device, model_name, init_flag)

        test_acc = test(model_trained, test_loader, model_name)

        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
        f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
        f.close()

        # Create attention matrix and prototype matrix of each client
        if iter == 0:
            attention_matrix.append(attention_and_prototype(model_trained, train_loader, labels, device)[0])
            prototype_matrix.append(attention_and_prototype(model_trained, train_loader, labels, device)[1])
        else:
            attention_matrix.append(attention_and_prototype(model_trained, train_loader, set(np.arange(y_dim)), device)[0])
            prototype_matrix.append(attention_and_prototype(model_trained, train_loader, set(np.arange(y_dim)), device)[1])

        models[model_name] = model_trained
    attention_last = attention_matrix
    prototype_last = prototype_matrix
    model_last = models

    # cloud
    # cloud_gen, global_atten_dict = adversarial(args, model_last, attention_last, prototype_last, \
    #                                            cloud_gen, iter, class_num, label_client, device)
    cloud_gen = adversarial(args, model_last, attention_last, prototype_last, \
                                               cloud_gen, iter, class_num, label_client, device)

    # plot distribution of cloud generated samples
    plot_generate_sample(args, cloud_gen, testdata, testlabel, iter, device, time_log)
    # # 全局atten层参数加载到各个边端的atten层
    # for para in model_last.values():
    #     para.atten.load_state_dict(global_atten_dict)


# ipdb.set_trace()
# plot_average_weight(aver_weights, './img/weight_change.png')


# # cnn and mlp trained respectively 
# # just for comparison, not included in collaboration procedure
# for model_name in model_total:
#     if model_name == 'CNN':
#         model = CNN(channel1, kernel1, channel2, kernel2, num_class, input_dim).to(device)
#     if model_name == 'MLP':
#         model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim).to(device)
    
#     if model_name == 'CNN':
#         trainset = dataSet_CNN(client.traindata[model_name], client.trainlabel[model_name])
#         testset = dataSet_CNN(client.testdata[model_name], client.testlabel[model_name])
#     if model_name == 'MLP':
#         trainset = dataSet_MLP(client.traindata[model_name], client.trainlabel[model_name])
#         testset = dataSet_MLP(client.testdata[model_name], client.testlabel[model_name])
#     train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
#     test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     epochs_resp = epochs * iterations
#     model_trained, train_acc = train(model, train_loader, optimizer, epochs_resp, device, model_name, class_num, True, prototype_last)

#     test_acc = test(model_trained, test_loader, model_name)

#     f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
#     f.write("**********************models trained respectively**********************\n")
#     f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
#     f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
#     f.close()




