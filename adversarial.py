import torch
import torch.nn.functional as F
import torch.nn as nn
from utils import *
import collections


def adversarial(args, models, attentions, prototypes, cloud_gen, iter, class_num, label_client, device):
    """
    这个函数里加入了MSGAN中解决模式坍塌问题的损失函数，mode seeking loss
    并且删除了全局模型
    """
    if args.dataset == 'Ku':
        y = torch.tensor(np.arange(9)).to(device)
    if args.dataset == 'TE':
        y = torch.tensor(np.arange(15)).to(device)
    y = F.one_hot(y)

    trainloader = generate_samples(args.total_num, args.z_dim, args.batch_size_global, y)

    # 得到边端权重
    weight = client_weight(trainloader, cloud_gen, models, device, class_num, label_client, args.weight_threshold)

    # 边端变量关注向量&原型的加权融合
    mean_atten = torch.tensor(global_attention(attentions, class_num, weight))
    mean_proto = torch.tensor(global_prototype(prototypes, class_num, weight))

    for model in models.values():
        model.eval()
    cloud_gen.train()

    # # 是否进行学习率衰减
    # if args.decay == 1:
    #     gen_lr = args.gen_lr * (args.gamma ** iter)
    # else:
    #     gen_lr = args.gen_lr
    gen_lr = args.gen_lr
    optimizer_gen = torch.optim.Adam(cloud_gen.parameters(), lr=gen_lr)

    soft = nn.Softmax(dim=1)
    ce = nn.CrossEntropyLoss()

    print("Adversarial Training Begin! ...")
    print("Train the generator first!")
    for epoch in range(args.global_epoch):
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
            y_number = torch.cat((y_number, y_number))
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
              epoch+1, args.global_epoch, loss_G.item(), np.mean(np.array(teachers_acc))*100))
    print("Generator Finished!")

    # attention module averaging for global attention module
    model_atten = nn.ModuleList()
    for value in models.values():
        model_atten.append(value.atten)
    global_atten_dict = para_avg(model_atten, weight)

    return cloud_gen, global_atten_dict
    # return cloud_gen


def adversarial_global(args, models, attentions, prototypes, cloud_gen, global_model, iter, class_num, label_client, device):
    """
    这个函数里加入了MSGAN中解决模式坍塌问题的损失函数，mode seeking loss
    """
    if args.dataset == 'Ku':
        y = torch.tensor(np.arange(9)).to(device)
    if args.dataset == 'TE':
        y = torch.tensor(np.arange(15)).to(device)
    y = F.one_hot(y)

    trainloader = generate_samples(args.total_num, args.z_dim, args.batch_size_global, y)

    # 得到边端权重
    weight = client_weight(trainloader, cloud_gen, models, device, class_num, label_client, args.weight_threshold)

    # 边端变量关注向量&原型的加权融合
    mean_atten = torch.tensor(global_attention(attentions, class_num, weight))
    mean_proto = torch.tensor(global_prototype(prototypes, class_num, weight))

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

    print("Adversarial Training Begin! ...")
    print("Train the generator first!")
    for epoch in range(args.global_epoch // 2):
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
            y_number = torch.cat((y_number, y_number))
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
            loss_S_atten = F.l1_loss(a, atten_target)
            loss_S_proto = F.l1_loss(p, proto_target)
            loss_S_label = ce(s_logit, y_number)
            # loss_S = args.lamda_logit * loss_S_logit + args.lamda_atten * loss_S_atten
            # loss_S = loss_S_label + loss_S_atten
            loss_S = args.beta_label * loss_S_label + \
                     args.beta_atten * loss_S_atten + \
                     args.beta_proto * loss_S_proto

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


def adversarial_grad(args, models, prototypes, cloud_gen, global_model, z_dim, iter, class_num, label_client, device):
    if args.dataset == 'Ku':
        y = torch.tensor(np.arange(9)).to(device)
    if args.dataset == 'TE':
        y = torch.tensor(np.arange(15)).to(device)
    y = F.one_hot(y)

    trainloader = generate_samples(args.total_num, z_dim, args.batch_size_global, y)

    # 得到边端权重
    weight, aver_maxweight = client_weight(trainloader, cloud_gen, models, device, class_num, label_client, args.weight_threshold)

    # 边端变量关注向量的加权融合
    mean_proto = torch.tensor(global_prototype(prototypes, class_num, weight))

    for model in models.values():
        model.eval()
        for _,v in model.named_parameters():
            v.requires_grad = False
    cloud_gen.train()
    global_model.train()

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

    print("Adversarial Training Begin! ...")
    for epoch in range(args.global_epoch):
        for batch_idx, (z, y) in enumerate(trainloader):
            
            z = z.to(device)

            atten_target = torch.mm(torch.FloatTensor(y.numpy()), \
                                    torch.FloatTensor(mean_proto.numpy())).to(device)

            y = y.to(device)
            y_number = torch.argmax(y, dim=1)

            # 先更新全局模型
            optimizer_global.zero_grad()

            # 冻结生成器模型参数
            for k,v in cloud_gen.named_parameters():
                v.requires_grad = False

            fake = cloud_gen(z, y)

            fake.requires_grad = True

            # 全局模型的logit
            _, s_logit = global_model(fake)
            s_prob = soft(s_logit)
            dycdx = torch.autograd.grad((s_prob * y).sum(), fake, create_graph=True, retain_graph=True)[0]

            dycdx[dycdx <= 0] = 0
            dycdx = F.normalize(dycdx, dim=1)

            # 边端模型logit的ensemble，使用weight进行加权
            with torch.no_grad():
                prob = {}
                i = 0
                for name, model in models.items():
                    if 'CNN' in name:
                        f = fake.unsqueeze(dim=1)
                    else:
                        f = fake
                    prob[i] = soft(model(f))
                    i = i + 1
                    
                t_prob = torch.zeros((len(y_number), class_num)).to(device)
                for i in range(len(y_number)):
                    for id, w in weight[int(y_number[i])].items():
                        t_prob[i] += w * prob[id][i]

            # 加权的老师网络预测结果的准确率
            teacher_acc = (torch.argmax(t_prob, dim=1) == y_number).sum().item() / (len(y_number))
            print("accuracy of teachers:", teacher_acc, "%")

            # 损失函数——1. logit 2. attention
            loss_S_logit = F.l1_loss(s_prob, t_prob.detach())
            loss_S_atten = F.l1_loss(dycdx, atten_target)
            # loss_S_label = ce(s_logit, y_number)
            loss_S = args.lamda_logit * loss_S_logit + args.lamda_atten * loss_S_atten
            # loss_S = args.lamda_atten * loss_S_atten + args.lamda_label * loss_S_label

            loss_S.backward()
            optimizer_global.step()

            # 再更新生成器 （每更新10次全局模型后更新一次生成器）
            if batch_idx % args.n_critic == 0:
                z_noise = torch.randn(args.batch_size_global, z_dim).to(device)

                # 解冻生成器模型参数
                for k,v in cloud_gen.named_parameters():
                    v.requires_grad = True

                optimizer_gen.zero_grad()
                fake = cloud_gen(z_noise, y)

                # 全局模型的logit
                _, s_logit = global_model(fake)
                s_prob = soft(s_logit)
                dycdx = torch.autograd.grad((s_prob * y).sum(), fake, create_graph=True, retain_graph=True)[0]
                dycdx[dycdx <= 0] = 0
                dycdx = F.normalize(dycdx, dim=1)

                # 边端模型logit的ensemble
                prob = {}
                logit = {}
                i = 0
                for name, model in models.items():
                    if 'CNN' in name:
                        f = fake.unsqueeze(dim=1)
                    else:
                        f = fake
                    logit[i] = model(f)
                    prob[i] = soft(model(f))
                    i = i + 1
                
                t_logit = torch.zeros((len(y_number), class_num)).to(device)
                t_prob = torch.zeros((len(y_number), class_num)).to(device)
                for i in range(len(y_number)):
                    for id, w in weight[int(y_number[i])].items():
                        t_logit[i] += w * logit[id][i]
                        t_prob[i] += w * prob[id][i]

                # 判断哪些样本在全局模型和老师模型中得到一致的分类结果
                pre_S = torch.argmax(s_prob, dim=1)
                pre_T = torch.argmax(t_prob, dim=1)
                equ_idx = torch.where(pre_T == y_number)[0]
                hard_samples = equ_idx[torch.where(pre_S[equ_idx] != pre_T[equ_idx])[0]]

                if len(hard_samples) != 0:
                    loss_G_logit = - F.l1_loss(s_prob[hard_samples], t_prob[hard_samples]) 
                    loss_G_atten = - F.l1_loss(dycdx[hard_samples], atten_target[hard_samples])
                    # loss_G_logit = - F.l1_loss(s_prob, t_prob) 
                    # loss_G_atten = - F.l1_loss(dycdx, atten_target)
                    loss_G_label = ce(t_logit, y_number)

                loss_G = args.lamda_logit * loss_G_logit + args.lamda_atten * loss_G_atten + args.lamda_label * loss_G_label

                loss_G.backward()
                optimizer_gen.step()
                

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
        
        print('Train Epoch: [{}/{}]\tG_Loss: {:.6f}  S_loss: {:.6f}'.format(
            epoch+1, args.global_epoch, loss_G.item(), loss_S.item()))
        print("Global model accuracy on trainloader (generated samples): {}%".format(trainacc))

    for model in models.values():
        for k,v in model.named_parameters():
            v.requires_grad = True

    return cloud_gen, global_model, aver_maxweight, trainacc, teacher_acc


def global_prototype(prototypes, class_num, weight_true):
    variable_dim = list(prototypes[0]['data'].values())[0].shape[0]
    global_proto = np.zeros((class_num, variable_dim))

    for key, client in weight_true.items():
        for id, w in client.items():
            global_proto[key] += w * prototypes[id]['data'][key]
        
    return global_proto


def global_attention(attentions, class_num, weight_true):
    variable_dim = list(attentions[0]['data'].values())[0].shape[0]
    global_atten = np.zeros((class_num, variable_dim))

    for key, client in weight_true.items():
        for id, w in client.items():
            global_atten[key] += w * attentions[id]['data'][key]
        
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


def para_avg(model_atten, weight):
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
