import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *

def adversarial(args, models, attentions, prototypes, cloud_gen, global_model, iter, class_num, label_client, device):
    """
    这个函数里面使用的不是不同边端的加权策略，而是选择最好的边端向他看齐的策略。
    简称为边端选择策略。
    """
    if args.dataset == 'Ku':
        y = torch.tensor(np.arange(9)).to(device)
    if args.dataset == 'TE':
        y = torch.tensor(np.arange(15)).to(device)
    y = F.one_hot(y)

    trainloader = generate_samples(args.total_num, args.z_dim, args.batch_size_global, y)

    # 得到边端权重
    # weight = client_weight(trainloader, cloud_gen, models, device, class_num, label_client, args.weight_threshold)
    max_index = client_choose(label_client)

    # 边端变量关注向量&原型的加权融合
    mean_atten = torch.tensor(global_attention(attentions, class_num, max_index))
    mean_proto = torch.tensor(global_prototype(prototypes, class_num, max_index))

    for model in models.values():
        model.eval()
    cloud_gen.train()
    global_model.train()

    # 是否进行学习率衰减
    if args.decay == 1:
        global_lr = args.global_lr * (args.gamma ** iter)
        gen_lr = args.gen_lr * (args.gamma ** iter)
    else:
        global_lr = args.global_lr
        gen_lr = args.gen_lr

    optimizer_global = torch.optim.Adam(global_model.parameters(), lr=global_lr)
    optimizer_gen = torch.optim.Adam(cloud_gen.parameters(), lr=gen_lr)

    soft = nn.Softmax(dim=1)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    print("Adversarial Training Begin! ...")
    print("Train the generator first!")
    for epoch in range(args.global_epoch // 2):
        teachers_acc = []
        for _, (z, y) in enumerate(trainloader):
            z = z.to(device)
            atten_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_atten.numpy())).to(device)
            proto_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_proto.numpy())).to(device)
            y = y.to(device)
            y_number = torch.argmax(y, dim=1)

            optimizer_gen.zero_grad()
            fake = cloud_gen(z, y)

            # 边端模型logit的ensemble
            prob = {}
            atten, proto, logit = {}, {}, {}
            i = 0
            for name, model in models.items():
                if 'CNN' in name:
                    f = fake.unsqueeze(dim=1)
                else:
                    f = fake
                atten[i], proto[i], logit[i] = model(f)
                prob[i] = soft(logit[i])

                if 'CNN' in name:
                    atten[i] = atten[i].squeeze(dim=1)
                    proto[i] = proto[i].squeeze(dim=1)
                i = i + 1
            
            t_logit = torch.zeros((len(y_number), class_num)).to(device)
            t_prob = torch.zeros((len(y_number), class_num)).to(device)
            t_atten = torch.zeros((len(y_number), fake.shape[1])).to(device)
            t_proto = torch.zeros((len(y_number), fake.shape[1])).to(device)
            for i in range(len(y_number)):
                t_logit[i] = logit[int(max_index[y_number[i].cpu().numpy()])][i]
                t_prob[i] = prob[int(max_index[y_number[i].cpu().numpy()])][i]
                t_atten[i] = atten[int(max_index[y_number[i].cpu().numpy()])][i]
                t_proto[i] = proto[int(max_index[y_number[i].cpu().numpy()])][i]
            
            # 加权的老师网络预测结果的准确率
            teacher_acc = (torch.argmax(t_prob, dim=1) == y_number).sum().item() / (len(y_number))
            teachers_acc.append(teacher_acc)

            loss_G_atten = mse(t_atten, atten_target)
            loss_G_proto = mse(t_proto, proto_target)
            loss_G_label = ce(t_logit, y_number)

            loss_G = args.lamda_atten * loss_G_atten + args.lamda_proto * loss_G_proto + args.lamda_label * loss_G_label

            loss_G.backward()
            optimizer_gen.step()

        print('Train Epoch: [{}/{}]\tG_Loss: {:.6f}\tTeacher_acc: {:.6f}%'.format(
            epoch+1, args.global_epoch//2, loss_G.item(), np.mean(np.array(teachers_acc))*100))
    print("Generator Finished!")

    print("Train the global model second!")
    for epoch in range(args.global_epoch // 2):
        for _, (z, y) in enumerate(trainloader):
            z = z.to(device)
            atten_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_proto.numpy())).to(device)
            proto_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_proto.numpy())).to(device)
            y = y.to(device)
            y_number = torch.argmax(y, dim=1)

            fake = cloud_gen(z, y)

            # 全局模型的logit
            a, p, _, s_logit = global_model(fake)

            # 损失函数——1. logit 2. attention
            # loss_S_logit = F.l1_loss(s_prob, t_prob.detach())
            loss_S_atten = mse(a, atten_target)
            loss_S_proto = mse(p, proto_target)
            loss_S_label = ce(s_logit, y_number)
            loss_S = args.beta_label * loss_S_label + args.beta_atten * loss_S_atten + args.beta_proto * loss_S_proto

            loss_S.backward()
            optimizer_global.step()

        with torch.no_grad():
            correct = 0
            total = 0
            for z, y in trainloader:
                z, y = z.to(device), y.to(device)
                y_number = torch.argmax(y, dim=1)
                fake = cloud_gen(z, y)
                outputs = soft(global_model(fake)[-1])

                predicted = torch.argmax(outputs, dim=1)
                total += len(y_number)
                correct += (predicted == y_number).sum().item()
            trainacc = 100 * correct / total

        print('Train Epoch: [{}/{}]\tS_loss: {:.6f}\tTrain_acc: {:.6f}%'.format(
               epoch+1, args.global_epoch//2, loss_S.item(), trainacc))

    print("Global Model Finished!")
    return cloud_gen, global_model, trainacc, teacher_acc

def global_prototype(prototypes, class_num, max_index):
    variable_dim = list(prototypes[0]['data'].values())[0].shape[0]
    global_proto = np.zeros((class_num, variable_dim))

    for i in range(len(max_index)):
        global_proto[i] = prototypes[int(max_index[i])]['data'][i]
        
    return global_proto

def global_attention(attentions, class_num, max_index):
    variable_dim = list(attentions[0]['data'].values())[0].shape[0]
    global_atten = np.zeros((class_num, variable_dim))

    for i in range(len(max_index)):
        global_atten[i] = attentions[int(max_index[i])]['data'][i]
        
    return global_atten

def client_weight(loader, cloud_gen, models, device, class_num, label_client, weight_threshold):
    t_prob = []
    labels = []
    for i, (z,y) in enumerate(loader):
        z = z.to(device)
        y = y.to(device)
        batch_size = z.shape[0]

        y_number = torch.argmax(y, dim=1)

        soft = nn.Softmax(dim=1)
        
        with torch.no_grad():
            fake = cloud_gen(z, y)

            t_prob_batch = torch.zeros((batch_size, len(models), 1))
            idx = 0
            for name, model in models.items():
                if 'CNN' in name:
                    f = fake.unsqueeze(dim=1)
                else:
                    f = fake
                l = soft(model(f)[-1])
                
                for i in range(batch_size):
                    t_prob_batch[i, idx, :] = l[i, y_number[i]]
                idx += 1
            t_prob.append(t_prob_batch)
            labels.append(y_number)

    t_prob = torch.cat(t_prob, dim=0)
    labels = torch.cat(labels, dim=0)

    w = {i:None for i in range(class_num)}

    for i in range(class_num):
        index = torch.where(labels == i)[0]
        w[i] = t_prob[index].squeeze(dim=2)
        w[i] = torch.mean(w[i], dim=0)

    weight = torch.zeros((class_num, len(models)))
    for key, value in w.items():
        weight[key] = value

    weight_true = {i:{} for i in range(class_num)}

    aver_maxweight = 0
    for i in range(weight.shape[0]):
        print(max(weight[i]))
        aver_maxweight += max(weight[i])
        if max(weight[i]) < weight_threshold:
            # 采用样本量加权
            print("---------------- 样本量加权！----------------")
            total_number = 0
            for key, value in label_client[i].items():
                total_number += value
            for key, value in label_client[i].items():
                weight_true[i][key] = value / total_number
        else:
            # 采用精度加权
            print("---------------- 精度加权！----------------")
            total_prob = 0
            for key in label_client[i].keys():
                total_prob += weight[i, key]
            for key in label_client[i].keys():
                weight_true[i][key] = float(weight[i, key]) / float(total_prob)

    return weight_true, aver_maxweight / weight.shape[0]

def client_choose(label_client):
    # 将label_client字典转换成dataframe
    df = pd.DataFrame(label_client)
    df = df.reset_index().rename(columns={'index':'class_id'})
    df = df.sort_values(by="class_id", ascending=True) 
    df.fillna(value=0, inplace=True)
    df = df.astype('int')
    df.drop(columns='class_id', inplace=True)
    n_clients = df.shape[0]
    index_rename = list(map(str, list(np.arange(n_clients))))
    df.index = index_rename

    # 求取每一列（每一类）的最大值所在索引，即知道每一类样本哪个边端拥有最多样本。
    max_index = df.idxmax()
    return max_index
