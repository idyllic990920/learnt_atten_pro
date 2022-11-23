import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np
from Client import Client_init
from dataset import dataSet_MLP
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
import random
# from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import argparse
from utils import set_seed


parser = argparse.ArgumentParser(description='mlp grad test')
# training settings
parser.add_argument('--dataset', default='TE', type=str)
parser.add_argument('--model_name', default='MLP_', type=str)
parser.add_argument('--lr', default='1e-3', type=float, help='Learning rate')
parser.add_argument('--epochs', default='10', type=int)
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
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dataset = args.dataset
model_name = args.model_name

if args.dataset == 'Ku':
    input_dim = 430
    hidden_dim1 = 200
    hidden_dim2 = 50
    output_dim = 9
else:
    input_dim = 52
    hidden_dim1 = 40
    hidden_dim2 = 20
    output_dim = 15


set_seed(seed)
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False

# train model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            # nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim2, output_dim)
        )
        

    def forward(self, x):
        return self.model(x)

def train(model, loader, optimizer, epochs, device, model_name):
    model.train()
    total_step = len(loader)
    ce = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(loader):
            xi = xi.to(torch.float).to(device)
            yi = yi.squeeze().to(device)

            optimizer.zero_grad()
            out = model(xi)
            loss = ce(out, yi)
            loss.backward()
            optimizer.step()    

            if i%100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: [{:.4f}]'
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))
    
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
    torch.save(model.state_dict(), './model/model.pth')


client = Client_init(dataset)
trainset = dataSet_MLP(client.traindata, client.trainlabel)
testset = dataSet_MLP(client.testdata, client.testlabel)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

mlp = MLP(input_dim=input_dim, hidden_dim1=hidden_dim1, hidden_dim2=hidden_dim2, output_dim=output_dim).to(device)

optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

train(mlp, train_loader, optimizer, epochs, device, model_name)
# mlp.load_state_dict(torch.load('./model/model.pth'))

# Test the model
mlp.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(torch.float).to(device)
        labels = labels.squeeze().to(device)
        outputs = mlp(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of {} Model: {} %'.format(model_name, 100 * correct / total))

#region
# # SVM
# clf = SVC(kernel='linear')
# clf.fit(client.traindata, client.trainlabel)
# y_predict = clf.predict(client.testdata)
# accuracy = (y_predict == client.testlabel.squeeze()).sum() / client.testlabel.shape[0]
# print("Test Accuracy of SVM:", accuracy*100, "%")

# # RF
# clf = RandomForestClassifier()
# clf.fit(client.traindata, client.trainlabel)
# y_predict = clf.predict(client.testdata)
# accuracy = (y_predict == client.testlabel.squeeze()).sum() / client.testlabel.shape[0]
# print("Test Accuracy of RF:", accuracy*100, "%")
#endregion

dydx = {}
for item in range(client.class_number):
    dydx[item] = []

mlp.eval()
for i, (xi, yi) in enumerate(train_loader):
    # print(xi.requires_grad)
    xi = xi.to(torch.float).to(device)
    yi = yi.squeeze()

    xi.requires_grad = True

    out = mlp(xi)

    onehot_y = torch.nn.functional.one_hot(yi, client.class_number).to(device)

    dycdx = torch.autograd.grad((out * onehot_y).sum(), xi, create_graph=True, retain_graph=True)[0]

    yi = yi.numpy()
    for i in range(len(yi)):
        dydx[yi[i]].append(dycdx[i, :].data.unsqueeze(dim=0).cpu().numpy())

prototypes = []
for key,value in dydx.items():
    value = np.vstack(value)
    proto = np.mean(value, axis=0)
    proto[proto <= 0] = 0
    prototypes.append(proto)

prototype_mat = np.vstack(prototypes)

cor_mat = np.corrcoef(prototype_mat)

img = plt.matshow(cor_mat)
for i in range(cor_mat.shape[0]):
    for j in range(cor_mat.shape[1]):
        plt.text(x=j, y=i, s=np.round(cor_mat[i,j], 2), fontsize=5)
plt.colorbar(img, ticks=[cor_mat.min(), 0.5, 1])
plt.xticks(np.arange(prototype_mat.shape[0]))
plt.yticks(np.arange(prototype_mat.shape[0]))
plt.title('Attention Relationship of Faults in {}'.format(dataset))
plt.savefig("./img/{}/cor_mat_{}.png".format(args.model_name, epochs))
print(np.sort(cor_mat)[-1][-2])

