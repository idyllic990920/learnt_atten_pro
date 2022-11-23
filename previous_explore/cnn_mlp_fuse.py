import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np
from Client import Client_init
from dataset import dataSet_MLP, dataSet_CNN
from utils import set_seed, plot_proto_cor
import matplotlib.pyplot as plt
import argparse
from datetime import datetime
import os
from models import CNN, MLP


parser = argparse.ArgumentParser(description='Grad Attention Fuse')
parser.add_argument('--dataset', default='TE', type=str)
parser.add_argument('--log_name', default='./log/fuse', type=str)
parser.add_argument('--model', default=['MLP', 'CNN'], type=list)
parser.add_argument('--lr', default='1e-3', type=float, help='Learning rate')
parser.add_argument('--epochs', default='10', type=int)
parser.add_argument('--iterations', default='20', type=int)
parser.add_argument('--cu_num', default='0,1', type=str)
parser.add_argument('--seed', default='1', type=int)
parser.add_argument('--batch_size', default='32', type=int)
args = parser.parse_args()
print(args)

seed = args.seed
batch_size = args.batch_size
os.environ['CUDA_VISIBLE_DEVICES'] = args.cu_num
lr = args.lr
epochs = args.epochs
iterations = args.iterations
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
dataset = args.dataset
model_total = args.model
log_name = args.log_name
if dataset == 'Ku':
    input_dim = 430
else:
    input_dim = 52

set_seed(seed)
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False

client = Client_init(dataset, model_total)
class_num = client.class_number

def train(model, loader, optimizer, epochs, device, model_name, class_num, init_flag, prototype_last):
    model.train()
    total_step = len(loader)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(loader):
            xi = xi.to(torch.float).to(device)
            yi = yi.squeeze().to(device)
            xi.requires_grad = True

            optimizer.zero_grad()
            out = model(xi)
            loss = ce(out, yi)

            if init_flag != True:
                onehot_y = torch.nn.functional.one_hot(yi, class_num).to(device)
                dycdx = torch.autograd.grad((out * onehot_y).sum(), xi, create_graph=True, retain_graph=True)[0]
                mean_proto = torch.tensor((prototype_last[0] + prototype_last[1]) / 2).to(device)
                if model_name == 'CNN':
                    mean_proto = mean_proto[yi].unsqueeze(dim=1)
                if model_name == 'MLP':
                    mean_proto = mean_proto[yi]
                loss_pro = mse(dycdx, mean_proto)
                loss = loss + 0.5 * loss_pro

            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: [{:.4f}]'
                .format(epoch+1, epochs, i+1, total_step, loss.item()))
        # print("Training left time: {}".format((time.time() - start_time) * (epochs-epoch)))
    
    # 输出训练集上预测精度
    model.eval()  
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs = inputs.to(torch.float).to(device)
            labels = labels.squeeze().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Train Accuracy of {} Model: {} %'.format(model_name, 100 * correct / total))
    # torch.save(model.state_dict(), './model/model.pth')
    return model, 100 * correct / total

def test(model, loader, model_name):
    model.eval()  
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in loader:
            inputs = inputs.to(torch.float).to(device)
            labels = labels.squeeze().to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Test Accuracy of {} Model: {} %'.format(model_name, 100 * correct / total))
    return 100 * correct / total

def prototype(model, loader, class_num, device, dataset, model_name, epochs, plot=False):
    dydx = {}
    for item in range(class_num):
        dydx[item] = []

    model.eval()
    for i, (xi, yi) in enumerate(loader):
        # print(xi.requires_grad)
        xi = xi.to(torch.float).to(device)
        yi = yi.squeeze()

        xi.requires_grad = True

        out = model(xi)

        onehot_y = torch.nn.functional.one_hot(yi, class_num).to(device)

        dycdx = torch.autograd.grad((out * onehot_y).sum(), xi, create_graph=True, retain_graph=True)[0]

        yi = yi.numpy()
        for i in range(len(yi)):
            dydx[yi[i]].append(dycdx[i, :].data.unsqueeze(dim=0).cpu().numpy())

    prototypes = []
    for _,value in dydx.items():
        value = np.vstack(value)
        proto = np.mean(value, axis=0)
        proto[proto <= 0] = 0
        prototypes.append(proto)

    prototype_mat = np.vstack(prototypes)

    if plot == True:
        plot_proto_cor(prototype_mat, dataset, model_name, epochs)
    return prototype_mat


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

# cnn and mlp collaboration
for iter in range(iterations):
    print("####################### Iter [{}/{}] #######################".format(iter+1, iterations))

    init_flag = False
    prototype_matrix = []
    models = {}
    for model_name in model_total:
        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('Iteration [{}/{}] \n'.format(iter+1, iterations))
        f.close()

        if iter == 0:
            init_flag = True
            prototype_last = []

            if model_name == 'CNN' and dataset == 'Ku':
                channel1 = 8
                channel2 = 16
                kernel1 = 3
                kernel2 = 2
                num_class = 9
            if model_name == 'CNN' and dataset == 'TE':
                channel1 = 512
                channel2 = 1024
                kernel1 = 2
                kernel2 = 2
                num_class = 15
            if model_name == 'MLP' and dataset == 'Ku':
                input_dim = 430
                hidden_dim1 = 200
                hidden_dim2 = 50
                output_dim = 9
            if model_name == 'MLP' and dataset == 'TE':
                input_dim = 52
                hidden_dim1 = 40
                hidden_dim2 = 20
                output_dim = 15

            if model_name == 'CNN':
                model = CNN(channel1, kernel1, channel2, kernel2, num_class).to(device)
            if model_name == 'MLP':
                model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim).to(device)
        else:
            model = model_last[model_name]

        # prepare data
        if model_name == 'CNN':
            trainset = dataSet_CNN(client.traindata[model_name], client.trainlabel[model_name])
            testset = dataSet_CNN(client.testdata[model_name], client.testlabel[model_name])
        if model_name == 'MLP':
            trainset = dataSet_MLP(client.traindata[model_name], client.trainlabel[model_name])
            testset = dataSet_MLP(client.testdata[model_name], client.testlabel[model_name])
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        model_trained, train_acc = train(model, train_loader, optimizer, epochs, device, model_name, class_num, init_flag, prototype_last)

        test_acc = test(model_trained, test_loader, model_name)

        f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
        f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
        f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
        f.close()

        # Create prototype matrix of each model
        prototype_matrix.append(prototype(model, train_loader, class_num, device, \
                                          args.dataset, model_name, args.epochs, plot=False))
        models[model_name] = model_trained
    prototype_last = prototype_matrix
    model_last = models
    mean_proto = (prototype_last[0] + prototype_last[1]) / 2
    plot_proto_cor(mean_proto, dataset, 'Fuse_global', epochs, iter, iterations)

# cnn and mlp trained respectively
for model_name in model_total:
    if model_name == 'CNN':
        model = CNN(channel1, kernel1, channel2, kernel2, num_class).to(device)
    if model_name == 'MLP':
        model = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim).to(device)
    
    if model_name == 'CNN':
        trainset = dataSet_CNN(client.traindata[model_name], client.trainlabel[model_name])
        testset = dataSet_CNN(client.testdata[model_name], client.testlabel[model_name])
    if model_name == 'MLP':
        trainset = dataSet_MLP(client.traindata[model_name], client.trainlabel[model_name])
        testset = dataSet_MLP(client.testdata[model_name], client.testlabel[model_name])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    epochs_resp = epochs * iterations
    model_trained, train_acc = train(model, train_loader, optimizer, epochs_resp, device, model_name, class_num, True, prototype_last)

    test_acc = test(model_trained, test_loader, model_name)

    f = open(os.path.join(path, '{}_log.txt'.format(model_name)), "a")
    f.write("**********************models trained respectively**********************\n")
    f.write('train accuracy on {} model: {}   '.format(model_name, train_acc))
    f.write('test accuracy on {} model: {}\n'.format(model_name, test_acc))
    f.close()

    prototype(model_trained, train_loader, class_num, device, \
                                          args.dataset, '{}_respective'.format(model_name), epochs_resp, plot=True)




