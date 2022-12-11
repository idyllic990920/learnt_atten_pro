import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MLP, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
        )
        self.atten = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = self.atten(x)
        a_new = torch.exp(a)
        x_new = x * a_new
        out = self.out(x_new)
        return a, x_new, out


class CNN(nn.Module):
    def __init__(self, input_dim, channel1, kernel1, channel2, kernel2, num_class):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, channel1, kernel_size=kernel1, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            # nn.BatchNorm1d(channel1),
            nn.Conv1d(channel1, channel2, kernel_size=kernel2, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            # nn.BatchNorm1d(channel2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(channel2*48, num_class),
        )

        self.atten = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh()
        )

    def forward(self, x):
        a = self.atten(x)
        a_new = torch.exp(a)
        x_new = x * a_new

        feature = self.features(x_new)
        feature = torch.flatten(feature, 1) 
        out = self.classifier(feature)
        return a, x_new, out


class GEN(nn.Module):
    def __init__(self, z_dim, y_dim, hid_dim1, ganout_dim):
        super(GEN, self).__init__()
        self.out = nn.Sequential(
            nn.Linear(z_dim + y_dim, hid_dim1),
            nn.BatchNorm1d(hid_dim1),
            nn.LeakyReLU(0.2),
            nn.Linear(hid_dim1, ganout_dim),
            nn.BatchNorm1d(ganout_dim, affine=False),
            # nn.LeakyReLU(0.2)
        )

    def forward(self, z, y):
        x = torch.cat((z, y), dim=1)
        out = self.out(x)
        return out


# class GLOBAL(nn.Module):
#     def __init__(self, input_dim, hidden_dim1, output_dim):
#         super(GLOBAL, self).__init__()
#         self.pre_out = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim1),
#             nn.BatchNorm1d(hidden_dim1),
#             nn.LeakyReLU(0.2),
#         )
#         self.out = nn.Sequential(
#             nn.Linear(hidden_dim1, output_dim),
#         )
#         self.atten = nn.Sequential(
#             nn.Linear(input_dim, input_dim),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         a = self.atten(x)
#         a_new = torch.exp(a)
#         x_new = x * a_new

#         pre_out = self.pre_out(x_new)
#         out = self.out(pre_out)
#         return a, x_new, pre_out, out




