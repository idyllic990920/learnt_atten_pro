from torch.utils.data import Dataset
import torch

class dataSet_MLP(Dataset):
    def __init__(self, traindata, train_label):
        self.input_data = traindata
        self.input_label = train_label

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.input_data[index])
        input_label = torch.tensor(self.input_label[index])

        return input_data, input_label

    def __len__(self):
        return self.input_data.shape[0]

class dataSet_CNN(Dataset):
    def __init__(self, traindata, train_label):
        self.input_data = traindata
        self.input_label = train_label

    def __getitem__(self, index):
        input_data = torch.from_numpy(self.input_data[index]).unsqueeze(dim=0)
        input_label = torch.tensor(self.input_label[index])

        return input_data, input_label

    def __len__(self):
        return self.input_data.shape[0]

class dataSet_cloud(Dataset):
    def __init__(self, noise, target):
        self.noise = noise
        self.target = target

    def __getitem__(self, index):
        input_data = self.noise[index, :]
        input_target = self.target[index]
        return input_data, input_target

    def __len__(self):
        return self.noise.shape[0]
