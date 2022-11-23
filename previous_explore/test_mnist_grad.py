import torch
import torch.nn as nn
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import numpy as np

# train model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        fc1 = self.relu1(self.fc1(x))
        out = self.relu2(self.fc2(fc1))
        return out


def train(model, loader, optimizer, epochs, batch_size, device):
    model.train()
    total_step = len(loader)
    for epoch in range(epochs):
        for i, (xi, yi) in enumerate(loader):
            xi = xi.reshape(batch_size, 28*28)
            xi = xi.to(device)
            yi = yi.to(device)
            # xi = torch.autograd.Variable(xi, requires_grad=True)

            optimizer.zero_grad()
            loss = ce(model(xi), yi)
            loss.backward()
            optimizer.step()

            if i%100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: [{:.4f}]'
                   .format(epoch+1, epochs, i+1, total_step, loss.item()))
    torch.save(model.state_dict(), './model/model.pth')

lr = 0.001
input_dim = 28*28
hidden_dim = int(784 / 5)
output_dim = 10
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 5
batch_size = 32

data_train = MNIST('./data/', download=True, transform=transforms.ToTensor())
data_test = MNIST('./data/', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=4)

# mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
# optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

# ce = nn.CrossEntropyLoss()

# train(mlp, train_loader, optimizer, epochs, batch_size, device)

# # Test the model
# mlp.eval()  
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images=images.reshape(-1, 28*28)
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = mlp(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

# calculate attention (for the input)
# method 1----without autograd
# for i, (xi, yi) in enumerate(train_loader):
#     print(xi.requires_grad)
#     xi = xi.to(device)
#     yi = yi.to(device)

#     xi.requires_grad = True
#     img = xi.reshape(-1, 28*28)
#     out = mlp(img)

#     loss = ce(out, yi)

#     loss.backward()

#     print('input xi has grad:', xi.requires_grad)
#     print('input grad:', xi.grad)
#     print(xi)

mlp = MLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim).to(device)
mlp.load_state_dict(torch.load('./model/model.pth'))
ce = nn.CrossEntropyLoss(reduction='none')

for i, (xi, yi) in enumerate(train_loader):
    print(xi.requires_grad)
    xi = xi.to(device)
    # yi = yi.to(device)

    xi.requires_grad = True
    img = xi.reshape(-1, 28*28)
    out = mlp(img)

    onehot_y = torch.nn.functional.one_hot(yi, 10).to(device)

    dycdx = torch.autograd.grad((out * onehot_y).sum(), xi, create_graph=True, retain_graph=True)[0]

    # loss = ce(out, yi)
    # dlossdx = torch.autograd.grad(loss, xi, create_graph=True, retain_graph=True)
    
    print('input xi has grad:', xi.requires_grad)
    # print('dlossdx1 :', dlossdx)
    print('dycdx :', dycdx)

    print(xi)

    # loss.backward()
    # print('input grad:', xi.grad)

    # print(xi.grad == dlossdx[0])

    