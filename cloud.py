import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import ipdb
from dataset import dataSet_MLP, dataSet_CNN
from torch.utils.data import DataLoader 
import collections


# train generator first and then train client models
def cloud(args, models, attentions, prototypes, cloud_gen, z_dim, iter, class_num, label_client, device):
    if args.dataset == 'Ku':
        target = torch.tensor(np.arange(9)).to(device)
    if args.dataset == 'TE':
        target = torch.tensor(np.arange(15)).to(device)
    target = F.one_hot(target)

    trainloader = generate_samples(args.total_num, z_dim, args.batch_size_gen, target)

    # 得到边端权重
    weight = client_weight(trainloader, cloud_gen, models, device, class_num, label_client, args.weight_threshold)

    # 边端变量关注向量的加权融合
    mean_atten = torch.tensor(global_attention(attentions, class_num, weight))
    mean_proto = torch.tensor(global_prototype(prototypes, class_num, weight))

    # 训练生成器时让边端模型参数冻结
    for model in models.values():
        model.eval()
    cloud_gen.train()

    # if args.decay == 1:
    #     gen_lr = args.gen_lr * (args.gamma ** iter)
    # else:
    #     gen_lr = args.gen_lr
    gen_lr = args.gen_lr
    optimizer_gen = torch.optim.Adam(cloud_gen.parameters(), lr=gen_lr)

    soft = nn.Softmax(dim=1)
    ce = nn.CrossEntropyLoss()
    print("------------------- Cloud Training Begin! -------------------")
    print("------------------- Train the generator first!---------------")
    for epoch in range(args.gen_epoch):
        teachers_acc = []
        for _, (z, y) in enumerate(trainloader):
            z_1 = z.to(device)
            z_2 = torch.randn_like(z_1)
            atten_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_atten.numpy())).to(device)
            proto_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_proto.numpy())).to(device)
            y = y.to(device)
            y_number = torch.argmax(y, dim=1)

            optimizer_gen.zero_grad()
            fake_1 = cloud_gen(z_1, y)
            fake_2 = cloud_gen(z_2, y)

            # 加倍处理
            fake = torch.cat((fake_1, fake_2), 0)
            y_number = torch.cat((y_number, y_number), 0)
            y = torch.cat((y, y), 0)
            atten_target = torch.cat((atten_target, atten_target), 0)
            proto_target = torch.cat((proto_target, proto_target), 0)

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
                for id, w in weight[int(y_number[i])].items():
                    t_logit[i] += w * logit[id][i]
                    t_prob[i] += w * prob[id][i]
                    t_atten[i] += w * atten[id][i]
                    t_proto[i] += w * proto[id][i]
            
            # 加权的老师网络预测结果的准确率
            teacher_acc = (torch.argmax(t_prob, dim=1) == y_number).sum().item() / (len(y_number))
            teachers_acc.append(teacher_acc)

            loss_G_atten = F.l1_loss(t_atten, atten_target)
            loss_G_proto = F.l1_loss(t_proto, proto_target)
            loss_G_label = ce(t_logit, y_number)

            # mode seeking loss
            lz = torch.mean(torch.abs(fake_2 - fake_1)) / torch.mean(torch.abs(z_2 - z_1))
            eps = 1 * 1e-5
            loss_G_ms = 1 / (lz + eps)

            loss_G = args.lamda_atten * loss_G_atten + \
                     args.lamda_proto * loss_G_proto + \
                     args.lamda_label * loss_G_label + \
                     args.lamda_ms * loss_G_ms

            loss_G.backward()
            optimizer_gen.step()

        print('Train Epoch: [{}/{}]\tG_Loss: {:.6f}\tTeacher_acc: {:.6f}%'.format(
                epoch+1, args.gen_epoch, loss_G.item(), np.mean(np.array(teachers_acc))*100))
    print("------------------- Generator Finished! ---------------------")

    print("------------------- Train Client Models Second!--------------")
    # 将边端模型中的attention层换成全局平均后的attention层
    # attention module averaging for global attention module
    model_atten = nn.ModuleList()
    for value in models.values():
        model_atten.append(value.atten)
    global_atten_dict = para_avg(model_atten)
    for para in models.values():
        para.atten.load_state_dict(global_atten_dict)
    
    # 生成器得先生成一些样本作为训练
    z = torch.randn(args.total_num, z_dim).to(device)
    target = target.detach().cpu().numpy()
    targets = torch.tensor(target.repeat(args.total_num / target.shape[0], axis=0)).to(device)
    samples = cloud_gen(z, targets).detach()
    # 选出能被边端模型加权结果分类准确的样本 ---> 保证样本的真实性
    samples, targets = sample_selection(samples, targets, models, class_num, device, weight)
    # 将训练生成器过程中每个模型产生的梯度清除，并开启训练模式
    for name, model in models.items():
        model.zero_grad()
        if 'CNN' in name:
            trainset = dataSet_CNN(samples, targets)
        else:
            trainset = dataSet_MLP(samples, targets)
        train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=False)
        ce = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(args.teacher_epoch):
            model.train()
            for i, (xi, yi) in enumerate(train_loader):
                # 把生成的较真实的样本输入所有模型中
                # 每个模型的损失函数包含两部分，一部分是预测结果和标签之间的交叉熵；另一部分是模型的attention和atten_target之间的mseloss.
                if xi.shape[0] < 2:
                    break
                xi = xi.to(torch.float).to(device)
                atten_target = torch.mm(torch.FloatTensor(yi.numpy()), \
                                        torch.FloatTensor(mean_atten.numpy())).to(device)
                proto_target = torch.mm(torch.FloatTensor(yi.numpy()), \
                                        torch.FloatTensor(mean_proto.numpy())).to(device)
                yi = yi.to(device)
                optimizer.zero_grad()
                a, p, out = model(xi)

                y_number = torch.argmax(yi, dim=1)
                loss_label = ce(out, y_number)
                if 'CNN' in name:
                    atten_target = atten_target.unsqueeze(dim=1)
                    proto_target = proto_target.unsqueeze(dim=1)

                loss_atten = F.l1_loss(a, atten_target)
                loss_proto = F.l1_loss(p, proto_target)
                loss = loss_label + loss_atten + loss_proto

                loss.backward()
                optimizer.step()

            # 输出训练集上预测精度

            model.eval()  
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in train_loader:
                    inputs = inputs.to(torch.float).to(device)
                    labels = torch.argmax(labels, dim=1).to(device)
                    outputs = model(inputs)[-1]
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                acc = 100 * correct / total

            print ('| Model [{}] | Epoch [{}/{}] | Loss: [{:.4f}] | Accuracy: [{:.4f}] % |'
                    .format(name, epoch+1, args.teacher_epoch, loss.item(), acc))
    print("------------------- Client Models Update Finished!-----------")
    
    return cloud_gen, models


def global_attention(attentions, class_num, weight_true):
    variable_dim = list(attentions[0]['data'].values())[0].shape[0]
    global_atten = np.zeros((class_num, variable_dim))

    for key, client in weight_true.items():
        for id, w in client.items():
            global_atten[key] += w * attentions[id]['data'][key]
        
    return global_atten

def global_prototype(prototypes, class_num, weight_true):
    variable_dim = list(prototypes[0]['data'].values())[0].shape[0]
    global_proto = np.zeros((class_num, variable_dim))

    for key, client in weight_true.items():
        for id, w in client.items():
            global_proto[key] += w * prototypes[id]['data'][key]
        
    return global_proto

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

    for i in range(weight.shape[0]):
        print(max(weight[i]))
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

    return weight_true

def sample_selection(samples, targets, models, class_num, device, weight):
    soft = nn.Softmax(dim=1)
    prob = {}
    logit = {}

    i = 0
    targets_number = torch.argmax(targets, dim=1)
    for name, model in models.items():
        if 'CNN' in name:
            sam = samples.unsqueeze(dim=1)
        else:
            sam = samples
        logit[i] = model(sam)[-1]
        prob[i] = soft(logit[i])
        i = i + 1
    
    t_logit = torch.zeros((len(targets_number), class_num)).to(device)
    t_prob = torch.zeros((len(targets_number), class_num)).to(device)

    for i in range(len(targets)):
        for id, w in weight[int(targets_number[i])].items():
            t_logit[i] += w * logit[id][i]
            t_prob[i] += w * prob[id][i]

    select_idx = torch.where(torch.argmax(t_prob, dim=1) == targets_number)[0]
    samples = samples[select_idx].detach().cpu().numpy()
    targets = targets[select_idx].detach().cpu().numpy()
    return samples, targets

def para_avg(model_atten):
    """
    这是模型参数平均函数！值得记忆！
    输入是多个模型组成的ModuleList
    输出是平均化后的模型参数，目标模型声明以后再使用load_state_dict()函数导入平均的参数即可！
    """
    worker_state_dict = [x.state_dict() for x in model_atten]
    weight_keys = list(worker_state_dict[0].keys())
    fed_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = 0
        for i in range(len(model_atten)):
            key_sum = key_sum + worker_state_dict[i][key]
        fed_state_dict[key] = key_sum / len(model_atten)

    return fed_state_dict
