import numpy as np
import pandas as pd
import torch
import random
import os
import matplotlib.pyplot as plt
from dataset import dataSet_cloud
from sklearn.manifold import TSNE


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def z_score(traindata, testdata):
    mean = np.mean(traindata, axis=0)
    std = np.std(traindata, axis=0)
    traindata = (traindata - mean) / std
    testdata = (testdata - mean) / std
    return traindata, testdata

def write_to_excel(label_client):
    df = pd.DataFrame(label_client)
    df = df.reset_index().rename(columns={'index':'class_id'})
    df = df.sort_values(by="class_id", ascending=True) 
    df.fillna(value=0, inplace=True)
    df = df.astype('int')
    df.to_excel('client_data_partition.xlsx')

def generate_samples(total_num, feature_dim, batch_size, y, shuffle=True):
    """
    输入：要生成的样本总数、特征维度、每一批云端训练数据集的数量、cpu/gpu、语义库
    输出：打包好的trainloader
    """
    if total_num % y.shape[0] != 0:
        raise AssertionError
    z_noise = torch.randn(total_num, feature_dim) # 输入噪声
    # 属性和类别标签都是重复了total_num/8次，是顺序排列的， 即000...111...222......888
    y = y.detach().cpu().numpy()
    target = torch.tensor(y.repeat(total_num / y.shape[0], axis=0))
    # target = torch.tensor(np.arange(class_num).repeat(total_num / y.shape[0], axis=0))

    # 包装dataset和dataloader
    trainset = dataSet_cloud(z_noise, target)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True)
    return trainloader

def plot_proto_cor(prototype_mat, dataset, model_name, epochs, iter=None, iterations=None):

    cor_mat = np.corrcoef(prototype_mat)

    img = plt.matshow(cor_mat)
    for i in range(cor_mat.shape[0]):
        for j in range(cor_mat.shape[1]):
            plt.text(x=j, y=i, s=np.round(cor_mat[i,j], 2), fontsize=5)
    plt.colorbar(img, ticks=[cor_mat.min(), 0.5, 1])
    plt.xticks(np.arange(prototype_mat.shape[0]))
    plt.yticks(np.arange(prototype_mat.shape[0]))
    plt.title('Attention Relationship of Faults in {}'.format(dataset))

    if model_name == 'Fuse_global':
        path = "./img/{}/iter{}_epoch{}/iter_{}".format(model_name, iterations, epochs, iter+1)
    else:
        path = "./img/{}".format(model_name)
    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(os.path.join(path, "cor_mat_{}.png".format(epochs)))
    print("---------------The largest relationship: ", np.sort(cor_mat)[-1][-2], "-----------------")

def plot_average_weight(aver_weights, filename):
    """
    params: aver_weights --- 多轮迭代中每一轮多个老师网络对每一类样本分类准确率的均值的最大值
    params: filename --- 画的图保存的位置
    """
    aver_weights = np.array(aver_weights)
    x = np.arange(aver_weights.shape[0])
    plt.figure()
    plt.plot(x, aver_weights)
    plt.savefig(filename)

def plot_generate_sample(args, cloud_gen, testdata, testlabel, iter, device, time_log):
    """
    按类别画出生成样本和原始测试样本的TSNE图（直接对数据层面画，没有特征提取）
    """
    # prepare generated data
    y = torch.tensor(np.arange(15)).to(device)
    y = torch.nn.functional.one_hot(y)
    y_number = torch.argmax(y, dim=1).cpu().numpy()

    trainloader = generate_samples(args.plot_num, args.z_dim, args.plot_batch_size, y, False)

    with torch.no_grad():
        for z, label in trainloader:
            z, label = z.to(device), label.to(device)
            fake = cloud_gen(z, label)
        label = torch.argmax(label, dim=1)
            
    fake = fake.cpu().numpy()
    label = label.cpu().numpy()

    # prepare testdata
    tsne = TSNE(n_components=2)

    # data through tsne
    tt = np.vstack((fake, testdata))
    all_label = np.vstack((label.reshape((-1, 1)), testlabel.reshape((-1, 1))))
    test_tsne = tsne.fit_transform(tt)

    test_tsne = np.hstack((test_tsne, all_label))
    test_tsne_gene = pd.DataFrame({'x':test_tsne[:args.plot_num, 0], 
                                   'y':test_tsne[:args.plot_num, 1], 
                                   'label':test_tsne[:args.plot_num, 2]})

    test_tsne_orig = pd.DataFrame({'x':test_tsne[args.plot_num:, 0], 
                                  'y':test_tsne[args.plot_num:, 1], 
                                  'label':test_tsne[args.plot_num:, 2]})
    
    plt.figure()
    for i in range(len(y)):
        X_data = test_tsne_gene.loc[test_tsne_gene['label'] == y_number[i]]['x']
        Y_data = test_tsne_gene.loc[test_tsne_gene['label'] == y_number[i]]['y']
        plt.scatter(X_data, Y_data, marker='o', c=plt.get_cmap('tab20b')(np.linspace(0, 1, 15))[i], label='gene ' + str(y_number[i]), alpha=0.1)
        # plt.scatter(X_data, Y_data, marker='o', c='r', label='gene ' + str(y_number[i]), alpha=0.1)
        
        X_data = test_tsne_orig.loc[test_tsne_orig['label'] == y_number[i]]['x']
        Y_data = test_tsne_orig.loc[test_tsne_orig['label'] == y_number[i]]['y']
        plt.scatter(X_data, Y_data, marker='D', c=plt.get_cmap('tab20b')(np.linspace(0, 1, 15))[i], label='orig ' + str(y_number[i]), alpha=0.1)
        # plt.scatter(X_data, Y_data, marker='D', c='b', label='orig ' + str(y_number[i]), alpha=0.1)

    # plt.legend()
    if not os.path.exists("./img/generate/{}/iter{}".format(time_log, iter+1)):
        os.makedirs("./img/generate/{}/iter{}".format(time_log, iter+1))
    plt.savefig('./img/generate/{}/iter{}/test_orig_tsne_class{}'.format(time_log, iter+1, i+1))

#region
# def plot_cls_out(args, cloud_gen, global_model, plot_loader, iter, device, time_log):
#     """
#     画出生成样本和原始测试样本在分类器倒数第二层的输出特征TSNE图（分类器最后一层输出分类logit，这里画的是最后一层的输入）
#     """
#     y = torch.tensor(np.arange(15)).to(device)
#     y = torch.nn.functional.one_hot(y)
#     y_number = torch.argmax(y, dim=1).cpu().numpy()

#     trainloader = generate_samples(args.plot_num, args.z_dim, args.plot_batch_size, y, False)

#     with torch.no_grad():
#         # prepare generated data
#         for z, label in trainloader:
#             z, label = z.to(device), label.to(device)
#             fake = cloud_gen(z, label)
#             _, _, fake_pre_out, _ = global_model(fake)
#         label = torch.argmax(label, dim=1)

#         # prepare testdata
#         for testdata, testlabel in plot_loader:
#             testdata, testlabel = testdata.to(torch.float).to(device), testlabel.to(device)
#             _, _, test_pre_out, _ = global_model(testdata)

#     label = label.cpu().numpy()
#     fake = fake.cpu().numpy()
#     testlabel = testlabel.cpu().numpy()
#     fake_pre_out = fake_pre_out.cpu().numpy()
#     test_pre_out = test_pre_out.cpu().numpy()

#     # data through tsne
#     tsne = TSNE(n_components=2)
#     tt = np.vstack((fake_pre_out, test_pre_out))
#     all_label = np.vstack((label.reshape((-1, 1)), testlabel.reshape((-1, 1))))
#     test_tsne = tsne.fit_transform(tt)

#     test_tsne = np.hstack((test_tsne, all_label))
#     test_tsne_gene = pd.DataFrame({'x':test_tsne[:args.plot_num, 0], 
#                                    'y':test_tsne[:args.plot_num, 1], 
#                                    'label':test_tsne[:args.plot_num, 2]})

#     test_tsne_orig = pd.DataFrame({'x':test_tsne[args.plot_num:, 0], 
#                                   'y':test_tsne[args.plot_num:, 1], 
#                                   'label':test_tsne[args.plot_num:, 2]})

#     plt.figure()
#     for i in range(len(y)):
#         X_data = test_tsne_gene.loc[test_tsne_gene['label'] == y_number[i]]['x']
#         Y_data = test_tsne_gene.loc[test_tsne_gene['label'] == y_number[i]]['y']
#         plt.scatter(X_data, Y_data, c=plt.get_cmap('tab20b')(np.linspace(0, 1, 15))[i], label='gene ' + str(y_number[i]), alpha=0.35)
#     # for i in range(len(y)):
#         # X_data = test_tsne_orig.loc[test_tsne_orig['label'] == y_number[i]]['x']
#         # Y_data = test_tsne_orig.loc[test_tsne_orig['label'] == y_number[i]]['y']
#         # plt.scatter(X_data, Y_data, c=plt.get_cmap('tab20')(np.linspace(0, 1, 15))[i], label='orig ' + str(y_number[i]), alpha=0.35)
#     plt.legend()
#     if not os.path.exists("./img/generate/{}/iter{}".format(time_log, iter+1)):
#         os.makedirs("./img/generate/{}/iter{}".format(time_log, iter+1))
#     plt.savefig('./img/generate/{}/iter{}/gene_feature_class'.format(time_log, iter+1))

#     plt.figure()
#     for i in range(len(y)):
#         X_data = test_tsne_orig.loc[test_tsne_orig['label'] == y_number[i]]['x']
#         Y_data = test_tsne_orig.loc[test_tsne_orig['label'] == y_number[i]]['y']
#         plt.scatter(X_data, Y_data, c=plt.get_cmap('tab20')(np.linspace(0, 1, 15))[i], label='orig ' + str(y_number[i]), alpha=0.35)

#     plt.legend()
#     if not os.path.exists("./img/generate/{}/iter{}".format(time_log, iter+1)):
#         os.makedirs("./img/generate/{}/iter{}".format(time_log, iter+1))
#     plt.savefig('./img/generate/{}/iter{}/orig_feature_class'.format(time_log, iter+1))
#endregion



