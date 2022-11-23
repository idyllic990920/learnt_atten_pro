import os
import numpy as np
import matplotlib.pyplot as plt
import re

def line_filter(f):
    if 'Iteration' in f:
        return False
    else:
        return True

def plot_test_acc(path, num_client, iteration=None):
    time = path.split('/')[-1]
    results = read_results(path, iteration=iteration)
    assert len(results) == num_client, 'results的长度不等于边端数量'
    values = np.zeros((num_client, len(list(results.values())[0])))
    i = 0
    for name, value in results.items():
        x = np.arange(len(value))
        value = np.array(value)
        plt.figure()
        plt.plot(x, value, marker='D')
        plt.title(name)
        if not os.path.exists('./img/result/{}'.format(time)):
            os.makedirs('./img/result/{}'.format(time))
        plt.savefig('./img/result/{}/{}_result.png'.format(time, name), bbox_inches='tight', dpi=300)
        values[i] = value
        i = i + 1
    
    mean_acc = np.mean(values, axis=0)
    plt.figure()
    x = np.arange(len(mean_acc))
    plt.plot(x, mean_acc, marker='x')
    plt.title('Average of all clients')
    if not os.path.exists('./img/result/{}'.format(time)):
        os.makedirs('./img/result/{}'.format(time))
    plt.savefig('./img/result/{}/{}.png'.format(time, 'Average_of_all_clients'), bbox_inches='tight', dpi=300)


def read_results(path, iteration=None):
    # 处理读到的实验下面的轮次顺序，按照round_1,2,3...顺序排列
    log_list = os.listdir(path)
    for i in range(len(log_list)):
        log_list[i] = log_list[i].split("_")
    log_list.sort(key=lambda x: int(x[0][-1]))
    for i in range(len(log_list)):
        log_list[i] = '_'.join(log_list[i])

    results = {}
    for i in log_list:
        model_path = os.path.join(path, i)
        f = open(model_path, 'r')
        result = f.readlines()
        total = re.findall("\d+\.?\d*", result[-2])[-1]             # 得到总共应该进行多少轮迭代，采用了正则表达式
        begin = result.index('Iteration [1/{}] \n'.format(total))   # 得到第一轮迭代的起始位置
        result = result[begin:]                                     # 去掉前面的参数写入部分，从第一轮开始读结果
        result = list(filter(line_filter, result))                  # 去掉每一行的 Iteration [] \n
        if iteration != None:
            result = result[:iteration]
        for j in range(len(result)):
            result[j] = float(result[j].split(":")[-1].strip('\n'))
        if iteration != None:
            assert len(result) == iteration, 'result长度不等于迭代轮次'
        f.close()
        model_name = i.split('_')[0]
        results[model_name] = result
    return results


#region
# def result_filter(f):
#     if f[0:5] == 'train':
#         return True
#     else:
#         return False


# def plot_results(path):
#     CNN_path = os.path.join(path, 'CNN_log.txt')
#     MLP_path = os.path.join(path, 'MLP_log.txt')

#     models = [CNN_path, MLP_path]

#     for model in models:
#         f = open(model, 'r')
#         results_ = f.readlines()
#         results = list(filter(result_filter, results_))
#         f.close()
#         model_name = model.split("/")[-1][:3]

#         for i in range(len(results)):
#             results[i] = float(results[i].split(': ')[-1].strip('\n'))
        
#         method = path.split("/")[-2]
#         method_path = "./results/{}".format(method)
#         if not os.path.exists(method_path):
#             os.makedirs(method_path)

#         x = np.arange(len(results))
#         plt.figure()
#         plt.plot(x, results, 'o-', markersize=5)
#         plt.title("{}".format(model_name))
#         plt.xlabel("iterations")
#         plt.ylabel("test accuracy")
#         plt.savefig(os.path.join(method_path, '{}.png'.format(model_name)))

# def plot_record(algorithm, paths, max_rounds, exp_name='compare', layout=220):
#     # 对比实验画图
#     if exp_name == 'compare':
#         f = open('/data/wjy/Promising_idea/FedKA-Ku/Comparision_exp/baselines_comparison.txt', 'w')
#         f.write('This txt is used to record final score(after 50 iterations) of each algorithm \n')
#         f.close()
#         plt.figure()
#         for p in range(len(paths)):
#             path = os.path.join('/data/wjy/Promising_idea/FedKA-Ku/', paths[p])
#             avg_score, _, _ = read_results(path, max_rounds)
#             # f = open('/data/wjy/Promising_idea/FedKA-Ku/Comparision_exp/baselines_comparison.txt', 'a+')
#             # f.write('{}: {} \n'.format(algorithm[p], re_score))
#             # f.close()

#             x = np.arange(len(avg_score))
#             plt.plot(x, avg_score, 'o-', markersize=2, label='{}'.format(algorithm[p]))
#         plt.title("Comparison Experiments", fontsize=20)
#         plt.xlabel("iterations", fontsize=17)
#         plt.ylabel("score", fontsize=17)
#         plt.xticks(size=17)
#         plt.yticks(size=17)
#         plt.legend(fontsize=15)
#         plt.savefig('./img_compare.pdf', bbox_inches='tight', dpi=1200)
    
#     # 消融实验画图——画多个子图的那种
#     if exp_name == 'ablation':
#         path_ka = os.path.join('/data/wjy/Promising_idea/FedKA-Ku/', paths[0])
#         score_ka, _, _ = read_results(path_ka, max_rounds)

#         # f = plt.figure(figsize=(10.5,3.5))
#         f = plt.figure()
#         for p in range(1, len(paths)):
#             ax = plt.subplot(layout+p)
#             ax.set_title('{} ablation'.format(algorithm[p].split('without')[-1].strip()))
#             x = np.arange(len(score_ka))
#             plt.plot(x, score_ka, 'o-', markersize=2, linewidth=1, color='deepskyblue', label='{}'.format(algorithm[0]))

#             path = os.path.join('/data/wjy/Promising_idea/FedKA-Ku/', paths[p])
#             avg_score, _, _ = read_results(path, max_rounds)

#             ablation = algorithm[p].split('without')[-1].strip()
#             x = np.arange(len(avg_score))
#             if 'and' in ablation:
#                 ablation = ablation.split('and')
#                 for i in range(len(ablation)):
#                     ablation[i] =ablation[i].strip()[0]
#                 ablation = ' and '.join(ablation)
#                 label = 'FedKA/{}'.format(ablation)
#             else:
#                 label = 'FedKA/{}'.format(ablation[:5])
#             plt.plot(x, avg_score, 's-', markersize=2, linewidth=1, color='#fe86a4', label=label)
#             plt.xlabel("iterations", fontsize=12)
#             plt.ylabel("score", fontsize=12)
#             plt.legend(fontsize=12)
#             x_major_locator = MultipleLocator(10)
#             ax = plt.gca()
#             ax.xaxis.set_major_locator(x_major_locator)
#             if label == 'FedKA/attri':
#                 plt.ylim(0.7, 1.0)
#             elif label == 'FedKA/repre':
#                 plt.ylim(0.65, 1.0)
#             elif label == 'FedKA/logit':
#                 plt.ylim(0.55, 1.0)
#         # plt.suptitle("Ablation Experiments", fontsize=20)
#         f.tight_layout()
#         plt.savefig('./img_ablation.pdf', bbox_inches='tight', dpi=1200)
#endregion

if __name__ == '__main__':
    plot_test_acc("/data/wjy/Promising_idea/learnt_atten_pro/log/wo_avg/11-23 16:46", 10, 45)