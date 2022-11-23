import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import numpy as np
from Client import Client_init
from dataset import dataSet_MLP, dataSet_CNN
from utils import set_seed
import matplotlib.pyplot as plt
import argparse
import os
# from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


parser = argparse.ArgumentParser(description='cnn grad test')
# training settings
parser.add_argument('--dataset', default='TE', type=str)
parser.add_argument('--model_name', default='CNN_', type=str)
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

if dataset == 'Ku':
    channel1 = 8
    channel2 = 16
    kernel1 = 3
    kernel2 = 2
    num_class = 9
else:
    channel1 = 256
    channel2 = 512
    channel3 = 1024
    kernel1 = 2
    kernel2 = 2
    kernel3 = 2
    num_class = 15

set_seed(seed)
torch.backends.cudnn.deterministic = True      
torch.backends.cudnn.benchmark = False

# train model
class CNN(nn.Module):
    def __init__(self, channel1, kernel1, channel2, kernel2, channel3, kernel3, num_class):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channel1, kernel_size=kernel1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            # nn.BatchNorm1d(channel1),
            nn.Conv1d(channel1, channel2, kernel_size=kernel2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            # nn.Conv1d(channel2, channel3, kernel_size=kernel3, stride=1),
            # nn.ReLU(),
            # nn.MaxPool1d(kernel_size=2, stride=1),
            # nn.BatchNorm1d(channel2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(channel2*48, num_class)
        )    

    def forward(self, x):
        x = self.features(x) 
        x = torch.flatten(x, 1) 
        out = self.classifier(x)
        return out

def train(model, loader, optimizer, epochs, device, model_name):
    model.train()
    total_step = len(loader)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(loader):
            xi = xi.to(torch.float).to(device)
            yi = yi.to(device)

            optimizer.zero_grad()
            out = model(xi)

            yi = yi.squeeze()

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
trainset = dataSet_CNN(client.traindata, client.trainlabel)
testset = dataSet_CNN(client.testdata, client.testlabel)
train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

cnn = CNN(channel1, kernel1, channel2, kernel2, channel3, kernel3, num_class).to(device)

optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)

train(cnn, train_loader, optimizer, epochs, device, model_name)
# mlp.load_state_dict(torch.load('./model/model.pth'))

# Test the model
cnn.eval()  
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.to(torch.float).to(device)
        labels = labels.squeeze().to(device)
        outputs = cnn(inputs)
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

cnn.eval()
for i, (xi, yi) in enumerate(train_loader):
    # print(xi.requires_grad)
    xi = xi.to(torch.float).to(device)
    yi = yi.squeeze()

    xi.requires_grad = True

    out = cnn(xi)

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
plt.savefig("./img/{}/cor_mat_{}.png".format(model_name, epochs))
print(np.sort(cor_mat)[-1][-2])

