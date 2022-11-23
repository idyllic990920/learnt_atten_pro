#### Author: Wang Jia Ye #####
# This py document is used to generate data for each client
# no matter existing overlapping categories among clients
from scipy.io import loadmat
from collections import Counter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class Client_init:
    """
    The class is used to read data for client.
    dataset_name denotes the name of used dataset, including 'Ku' and 'TE'.
    """
    def __init__(self, dataset_name, models):
        self.dataset_name = dataset_name
        print('#####################################')
        print("Loading {} data...".format(self.dataset_name))

        if self.dataset_name == 'Ku':
            # some quality of dataset Ku
            self.class_number = 9

            # read data
            self.fault_dict = np.load('./data/Semantic/raw_data.npy', allow_pickle=True).item()
        
        if self.dataset_name == 'TE':
            self.class_number = 15
            
            path = './data/TE_mat_data/'

            fault1 = loadmat(path + 'd01.mat')['data']
            fault2 = loadmat(path + 'd02.mat')['data']
            fault3 = loadmat(path + 'd03.mat')['data']
            fault4 = loadmat(path + 'd04.mat')['data']
            fault5 = loadmat(path + 'd05.mat')['data']
            fault6 = loadmat(path + 'd06.mat')['data']
            fault7 = loadmat(path + 'd07.mat')['data']
            fault8 = loadmat(path + 'd08.mat')['data']
            fault9 = loadmat(path + 'd09.mat')['data']
            fault10 = loadmat(path + 'd10.mat')['data']
            fault11 = loadmat(path + 'd11.mat')['data']
            fault12 = loadmat(path + 'd12.mat')['data']
            fault13 = loadmat(path + 'd13.mat')['data']
            fault14 = loadmat(path + 'd14.mat')['data']
            fault15 = loadmat(path + 'd15.mat')['data']

            self.fault_dict = {0:fault1.T, 1:fault2.T, 2:fault3.T, 3:fault4.T, 4:fault5.T,
                               5:fault6.T, 6:fault7.T, 7:fault8.T, 8:fault9.T, 9:fault10.T,
                               10:fault11.T, 11:fault12.T, 12:fault13.T, 13:fault14.T, 14:fault15.T}

        self.get_data()
        self.model_get_data(models)
        


    def get_data(self):
        trainlabel = []
        traindata = []
        testlabel = []
        testdata = []

        # 构建训练集、验证集、测试集
        for item in range(len(self.fault_dict)):

            cur_sample = self.fault_dict[item].shape[0]    # cur_sample就是每一类的样本数量
            train_sample = int(cur_sample * 0.7)           # 选取其中的70%作为训练集，剩下的15%作为测试集，15%作为验证集

            data = self.fault_dict[item]
            per = np.random.permutation(data.shape[0])		#打乱后的行号
            data = data[per, :]		#获取打乱后的训练数据

            traindata.append(data[:train_sample, :])
            trainlabel.append(np.repeat(item, train_sample).reshape(-1, 1))

            # 把上述训练每个类别剩下的样本加入到测试集中，并创建相应的类别标签
            testdata.append(data[train_sample:, :])
            testlabel.append(np.repeat(item, cur_sample-train_sample).reshape(-1, 1))

        self.trainlabel = np.row_stack(trainlabel)
        self.traindata = np.row_stack(traindata)
        testlabel = np.row_stack(testlabel)
        testdata = np.row_stack(testdata)

        mean = np.mean(self.traindata, axis=0)
        std = np.std(self.traindata, axis=0)
        self.traindata = (self.traindata - mean) / std
        testdata = (testdata - mean) / std

        # 之前的操作一直没有划分验证集，所以在这里将处理好的测试集数据划分一半作为验证集
        # 首先要测试集数据打乱，因为原本基本是按照标签顺序来排的，现在要划分一半的数据作为验证集
        per = np.random.permutation(testdata.shape[0])		#打乱后的行号
        self.testdata = testdata[per, :]		#获取打乱后的训练数据
        self.testlabel = testlabel[per]
        
        # self.validdata = self.testdata[:self.testdata.shape[0] // 2]
        # self.testdata = self.testdata[self.testdata.shape[0] // 2:]

        # self.validlabel = self.testlabel[:self.testlabel.shape[0] // 2]
        # self.testlabel = self.testlabel[self.testlabel.shape[0] // 2:]       

#region
# class Client(Client_init):
#     """
#     The class is inherited from Client_init class
#     dataset_name denotes the name of used dataset, including 'Ku' and 'TE'. (str)
#     client_id denotes the client index of which client. (int)
#     train_index denotes the seen catogries of all clients. (many lists)
#     """
#     def __init__(self, dataset_name, client_id, *train_index):
#         super().__init__(dataset_name)
        
#         # 统计每个训练已见类别在各个端中见到的次数
#         index = []
#         for id in train_index:
#             index += id
#         self.train_index_collection = Counter(index)   
        
#         # 构建每一个类别有哪些client见过的字典，比如{0:[1,2,3],1:[1,2]}表示0故障123端都见过，1故障12端见过。
#         client_dict = {}
#         for i in range(self.class_number):
#            client_dict[i] = []
#         for i in range(self.class_number):
#             for j in range(len(train_index)):
#                 if i in train_index[j]:
#                     client_dict[i].append(j+1)
#         self.client_dict = client_dict
#         self.train_index = train_index
#         self.get_data(self.client_id)


#     def get_data(self, client_id):
#         trainlabel = []
#         train_attribute = []
#         traindata = []

#         testlabel = []
#         test_attribute = []
#         testdata = []

#         # 构建训练集
#         for item in self.train_index[client_id-1]:
#             test_index = list(set(np.arange(self.class_number)) - set(self.train_index[client_id-1]))

#             # 构建训练样本，分两种情况—— 只有自己一个端见过的类 and 还有别的端见过的类
#             # 只有自己一个端见过的类
#             data, attribute, label = [], [], []
#             r_data, r_attribute, r_label = [], [], []
#             if self.train_index_collection[item] > 1:
#                 # 把该类全体样本均分给每一个端，cur_sample就是每一个端该类的样本数量，通过用户端id来判断到底拥有哪一份数据
#                 cur_sample = int(self.fault_dict[item].shape[0] / self.train_index_collection[item])
#                 index = self.client_dict[item].index(client_id)
#                 self.fault_dict[item] = self.fault_dict[item][index * cur_sample:(index+1) * cur_sample, :]
#             else:
#                 cur_sample = self.fault_dict[item].shape[0]    # cur_sample就是每一类的样本数量

#             train_sample = int(cur_sample * 0.7)      # 选取其中的70%作为训练集，剩下的20%作为广义零样本的测试
#             for i in range(train_sample - self.input_window):
#                 data.append(self.fault_dict[item][i:i + self.input_window, :])
#                 attribute.append(self.attribute_matrix[item, :])
#                 label.append(item)
#             traindata.append(np.row_stack(data).reshape(-1, self.input_window, self.fault_dict[item].shape[1]))
#             train_attribute.append(np.row_stack(attribute))
#             trainlabel.append(np.row_stack(label))

#             self.trainlabel = np.row_stack(trainlabel)
#             self.train_attribute = np.row_stack(train_attribute)
#             self.traindata = np.row_stack(traindata)

#             # 把上述训练已见类别剩下的样本加入到测试集中，并创建相应的属性和类别标签
#             for i in range(train_sample, cur_sample - self.input_window):
#                 r_data.append(self.fault_dict[item][i:i + self.input_window, :])
#                 r_attribute.append(self.attribute_matrix[item, :])
#                 r_label.append(item)
#             testdata.append(np.row_stack(r_data).reshape(-1, self.input_window, self.fault_dict[item].shape[1]))
#             test_attribute.append(np.row_stack(r_attribute))
#             testlabel.append(np.row_stack(r_label))

#         # 构建未见类别测试集
#         for item in test_index:
#             data, attribute, label = [], [], []
#             cur_sample = self.fault_dict[item].shape[0]
#             for i in range(cur_sample - self.input_window):
#                 data.append(self.fault_dict[item][i:i + self.input_window, :])
#                 attribute.append(self.attribute_matrix[item, :])
#                 label.append(item)
#             testdata.append(np.row_stack(data).reshape(-1, self.input_window, self.fault_dict[item].shape[1]))
#             test_attribute.append(np.row_stack(attribute))
#             testlabel.append(np.row_stack(label))

#         testlabel = np.row_stack(testlabel)
#         test_attribute = np.row_stack(test_attribute)
#         testdata = np.row_stack(testdata)

#         mean = np.mean(self.traindata.reshape((-1, self.fault_dict[item].shape[1])), axis=0)
#         std = np.std(self.traindata.reshape((-1, self.fault_dict[item].shape[1])), axis=0)
#         self.traindata = (self.traindata - mean) / std
#         testdata = (testdata - mean) / std

#         # 之前的操作一直没有划分验证集，所以在这里将处理好的测试集数据划分一半作为验证集
#         # 首先要测试集数据打乱，因为原本基本是按照标签顺序来排的，现在要划分一半的数据作为验证集
#         per = np.random.permutation(testdata.shape[0])		#打乱后的行号
#         self.testdata = testdata[per, :, :]		#获取打乱后的训练数据
#         self.testlabel = testlabel[per]
#         self.test_attribute = test_attribute[per, :]

#         self.validdata = self.testdata[:self.testdata.shape[0] // 2]
#         self.testdata = self.testdata[self.testdata.shape[0] // 2:]

#         self.validlabel = self.testlabel[:self.testlabel.shape[0] // 2]
#         self.testlabel = self.testlabel[self.testlabel.shape[0] // 2:]

#         self.valid_attribute = self.test_attribute[:self.test_attribute.shape[0] // 2]
#         self.test_attribute = self.test_attribute[self.test_attribute.shape[0] // 2:]

# # 用于调试的简单测试代码
# # Clients_1 = Client('Ku', 1, [0,1,2,3,4,5], [2,3,4,5,6,7], [3,4,5,6,7,8])
# # Clients_2 = Client('Ku', 2, [0,1,2,3,4,5], [2,3,4,5,6,7], [3,4,5,6,7,8])
# # clients_3 = Client('TE', 1, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14])
#endregion

    def model_get_data(self, models):
        traindata = {}
        trainlabel = {}
        testdata = {}
        testlabel = {}
        for model_name in models:
            traindata[model_name] = []
            trainlabel[model_name] = []
            testdata[model_name] = []
            testlabel[model_name] = []

        # prepare traindata and trainlabel
        cur_samples = Counter(self.trainlabel.squeeze())

        client_num = len(models)

        start = 0
        for i in range(self.class_number):
            number = int(cur_samples[i] / client_num)  # number of samples belonging to every client
            
            for j, model_name in enumerate(models):
                traindata[model_name].append(self.traindata[start+j*number:start+(j+1)*number, :])
                trainlabel[model_name].append(self.trainlabel[start+j*number:start+(j+1)*number])
            
            start += cur_samples[i]

        for key, value in traindata.items():
            v = np.row_stack(value)
            traindata[key] = v

        for key, value in trainlabel.items():
            v = np.row_stack(value)
            trainlabel[key] = v
        
        # prepare traindata and trainlabel
        number = int(self.testdata.shape[0] / client_num)
        for i, model_name in enumerate(models):
            testdata[model_name] = self.testdata[i*number:(i+1)*number, :]
            testlabel[model_name] = self.testlabel[i*number:(i+1)*number]

        self.traindata = traindata
        self.trainlabel = trainlabel
        self.testdata = testdata
        self.testlabel = testlabel

        print("Loading finished !")


