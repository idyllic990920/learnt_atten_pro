#### Author: Wang Jia Ye #####
# This py document is used to patition data for each client 
# (statistical heterogeneity by Dirichlet distribution)
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from tqdm import trange
from collections import Counter
import random
import os

class Client_init:
    """
    The class is used to read data for client.
    dataset_name denotes the name of used dataset, including 'Ku' and 'TE'.
    """
    def __init__(self, args, dataset_name, models, train_ratio):
        self.dataset_name = dataset_name
        print('#####################################')
        print("Loading {} data...".format(self.dataset_name))

        if self.dataset_name == 'Ku':
            # some quality of dataset Ku
            self.class_number = 9
            self.train_ratio = train_ratio
            # read data
            self.fault_dict = np.load('../data/Semantic/raw_data.npy', allow_pickle=True).item()
        
        if self.dataset_name == 'TE':
            self.class_number = 15
            self.train_ratio = train_ratio
            
            self.path = '../data/extend_TE/'
            fault1 = np.load(self.path + 'fault_1.npy')[:5]
            fault2 = np.load(self.path + 'fault_2.npy')[:5]
            fault3 = np.load(self.path + 'fault_3.npy')[:5]
            fault4 = np.load(self.path + 'fault_4.npy')[:5]
            fault5 = np.load(self.path + 'fault_5.npy')[:5]
            fault6 = np.load(self.path + 'fault_6.npy')[:5]
            fault7 = np.load(self.path + 'fault_7.npy')[:5]
            fault8 = np.load(self.path + 'fault_8.npy')[:5]
            fault9 = np.load(self.path + 'fault_9.npy')[:5]
            fault10 = np.load(self.path + 'fault_10.npy')[:5]
            fault11 = np.load(self.path + 'fault_11.npy')[:5]
            fault12 = np.load(self.path + 'fault_12.npy')[:5]
            fault13 = np.load(self.path + 'fault_13.npy')[:5]
            fault14 = np.load(self.path + 'fault_14.npy')[:5]
            fault15 = np.load(self.path + 'fault_15.npy')[:5]
            
            self.fault_dict = {0:fault1, 1:fault2, 2:fault3, 3:fault4, 4:fault5,
                               5:fault6, 6:fault7, 7:fault8, 8:fault9, 9:fault10,
                               10:fault11, 11:fault12, 12:fault13, 13:fault14, 14:fault15}

        if args.resplit:
            train_dict, test_dict, _ = self.train_test_partition()
            train_data, test_data = self.model_get_heterodata(args, train_dict, test_dict)
        else:
            print("------------ Load pre-splited data! ------------")
            train_data = np.load(os.path.join(self.path, "train/train.npy"), allow_pickle=True).item()
            test_data = np.load(os.path.join(self.path, "test/test.npy"), allow_pickle=True).item()

        self.users = {i:{} for i in train_data['users']}

        i = 0
        for name in train_data['users']:
            self.users[name]['train'] = train_data['user_data'][name]
            self.users[name]['test'] = test_data['user_data'][name]
            self.users[name]['labels'] = train_data['labels'][i]
            self.users[name]['count'] = Counter(train_data['user_data'][name]['y'])
            i += 1

        label_client = {i:{} for i in range(self.class_number)}
        for i in range(self.class_number):
            for j in range(len(self.users)):
                uname = 'u_00{}'.format(j)
                if i in self.users[uname]['labels']:
                    label_client[i][j] = self.users[uname]['count'][i]
        self.label_client = label_client

    def train_test_partition(self):
        # 首先把所有的数据中前三列删掉，以及把前20个样本删掉，并变成二维矩阵
        for key,value in self.fault_dict.items():
            value = np.delete(value, np.arange(20), 1)
            value = np.delete(value, np.arange(3), 2)
            self.fault_dict[key] = value.reshape(-1, 52)

        train_dict = {}
        test_dict = {}
        validate_dict = {}

        # 构建训练集、验证集、测试集
        for item in range(len(self.fault_dict)):
            cur_sample = self.fault_dict[item].shape[0]                 # cur_sample就是每一类的样本数量
            train_sample = int(cur_sample * self.train_ratio)           # 选取其中的70%作为训练集，剩下的15%作为测试集，15%作为验证集

            train_dict[item] = self.fault_dict[item][:train_sample, :]
            test_dict[item] = self.fault_dict[item][train_sample:, :]

            per = np.random.permutation(test_dict[item].shape[0])	    # 打乱后的行号
            test_dict[item] = test_dict[item][per, :]		            # 获取打乱后的测试数据
            validate_dict[item] = test_dict[item][test_dict[item].shape[0] // 2:, :]
            test_dict[item] = test_dict[item][:test_dict[item].shape[0] // 2, :]

        return train_dict, test_dict, validate_dict

    def model_get_heterodata(self, args, train_dict, test_dict):
        NUM_USERS = args.n_user
        print(f"Reading source dataset.")

        train_data, n_train_sample, test_data, n_test_sample, SRC_N_CLASS = self.get_dataset(train_dict, test_dict)
        SRC_CLASSES=list(np.arange(SRC_N_CLASS))
        random.shuffle(SRC_CLASSES)
        print("{} labels in total.".format(len(SRC_CLASSES)))
        labels, train_data = self.process_user_data(args, 'train', train_data, n_train_sample, \
                                                                     SRC_CLASSES, NUM_USERS)
        test_data = self.process_user_data(args, 'test', test_data, n_test_sample, SRC_CLASSES, NUM_USERS, labels=labels)
        print("Finish Allocating User Samples")
        return train_data, test_data

    def get_dataset(self, train_dict, test_dict):
        train_data = []
        SRC_N_CLASS = len(train_dict)
        n_train_sample = 0
        for i in range(len(train_dict)):
            train_data.append(train_dict[i])
            n_train_sample += train_dict[i].shape[0]
        print(f"TRAIN SET:\n  Total #samples: {n_train_sample}.")
        print("  # samples per class:\n", [len(v) for v in train_data])

        test_data = []
        n_test_sample = 0
        for i in range(len(test_dict)):
            test_data.append(test_dict[i])
            n_test_sample += test_dict[i].shape[0]
        print(f"TEST SET:\n  Total #samples: {n_test_sample}.")
        print("  # samples per class:\n", [len(v) for v in test_data])

        return train_data, n_train_sample, test_data, n_test_sample, SRC_N_CLASS

    def process_user_data(self, args, mode, data, n_sample, SRC_CLASSES, NUM_USERS, labels=None):
        if mode == 'train':
            X, y, labels, idx_batch, samples_per_user  = self.divide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.alpha)
        if mode == 'test':
            assert labels != None or args.unknown_test
            X, y = self.divide_test_data(NUM_USERS, SRC_CLASSES, data, labels, args.unknown_test)
        dataset = {'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(NUM_USERS):
            uname ='u_{0:03d}'.format(i)
            dataset['users'].append(uname)
            dataset['user_data'][uname] = {
                'x': X[i],
                'y': y[i]}
            dataset['labels'] = labels

        print("{} #sample by user:".format(mode.upper()), dataset['num_samples'])

        data_path=f'{self.path}{mode}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_path=os.path.join(data_path, "{}.".format(mode) + "npy")
        with open(data_path, 'wb') as outfile:
            print(f"Dumping train data => {data_path}")
            np.save(outfile, dataset)
        if mode == 'train':
            for u in range(NUM_USERS):
                print("{} samples in total".format(samples_per_user[u]))
                train_info = ''
                # train_idx_batch, train_samples_per_user
                n_samples_for_u = 0
                for l in sorted(list(labels[u])):
                    n_samples_for_l = len(idx_batch[u][l])
                    n_samples_for_u += n_samples_for_l
                    train_info += "c={},n={}| ".format(l, n_samples_for_l)
                print(train_info)
                print("{} Labels/ {} Number of training samples for user [{}]:".format(len(labels[u]), n_samples_for_u, u))
            return labels, dataset
        else: 
            return dataset

    def divide_train_data(self, data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, alpha=0.5):
        min_size = 0 # track minimal samples per user

        ###### Determine Sampling #######
        while min_size < min_sample:
            print("Try to find valid data separation")
            idx_batch=[{} for _ in range(NUM_USERS)]
            samples_per_user = [0 for _ in range(NUM_USERS)]
            max_samples_per_user = n_sample // NUM_USERS

            for l in SRC_CLASSES:
                # get indices for all that data of one class
                idx_l = [i for i in range(len(data[l]))]
                np.random.shuffle(idx_l)
                
                # samples_for_l = int( min(max_samples_per_user, len(data[l]) // NUM_USERS) )
                # idx_l = idx_l[:samples_for_l]
                print(l, len(data[l]), len(idx_l))

                # dirichlet sampling from this label
                proportions=np.random.dirichlet(np.repeat(alpha, NUM_USERS))
                # re-balance proportions
                proportions=np.array([p * (n_per_user < max_samples_per_user) for p, n_per_user in zip(proportions, samples_per_user)])
                proportions=proportions / proportions.sum()
                proportions=(np.cumsum(proportions) * len(idx_l)).astype(int)[:-1]
                # participate data of that label
                for u, new_idx in enumerate(np.split(idx_l, proportions)):
                    # add new idex to the user
                    idx_batch[u][l] = new_idx.tolist()
                    samples_per_user[u] += len(idx_batch[u][l])
            min_size=min(samples_per_user)

        ###### CREATE USER DATA SPLIT #######
        X = [[] for _ in range(NUM_USERS)]
        y = [[] for _ in range(NUM_USERS)]
        labels=[set() for _ in range(NUM_USERS)]
        print("Processing users---allocate data for users......")
        for u, user_idx_batch in enumerate(idx_batch):
            for l, indices in user_idx_batch.items():
                if len(indices) == 0: continue
                X[u] += data[l][indices].tolist()
                y[u] += (l * np.ones(len(indices))).tolist()
                labels[u].add(l)
            y[u] = list(map(int, y[u]))
        return X, y, labels, idx_batch, samples_per_user

    def divide_test_data(self, NUM_USERS, SRC_CLASSES, test_data, labels, unknown_test):
        # Create TEST data for each user.
        test_X = [[] for _ in range(NUM_USERS)]
        test_y = [[] for _ in range(NUM_USERS)]
        idx = {l: 0 for l in SRC_CLASSES}
        for user in trange(NUM_USERS):
            if unknown_test: # use all available labels
                user_sampled_labels = SRC_CLASSES
            else:
                user_sampled_labels = list(labels[user])
            for l in user_sampled_labels:
                num_samples = int(len(test_data[l]) / NUM_USERS )
                assert num_samples + idx[l] <= len(test_data[l])
                test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
                test_y[user] += (l * np.ones(num_samples)).tolist()
                assert len(test_X[user]) == len(test_y[user]), f"{len(test_X[user])} == {len(test_y[user])}"
                idx[l] += num_samples
            test_y[user] = list(map(int, test_y[user]))
        return test_X, test_y



