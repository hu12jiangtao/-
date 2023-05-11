# 交叉验证:目标是为了对参数较优的选取，并且提高模型的泛化能力
# 模型需要选择超参数有lr,dropout,num_epochs,其中M1模型为lr=0.1,dropout=0.5,num_epochs=5,M2模型为lr=0.15,dropout=0.4,num_epochs=3
# 因此就引入了交叉验证，其具体步骤是将 原本的训练数据打乱后分为 0.8作为训练数据 和 0.2作为验证数据（5折为例），确保训练集中每一个数据都被当作过验证数据
# 一共分为: 0-0.2作为验证，0.2-1作为训练;0.2-0.4作为验证，0-0.2 + 0.4-1作为训练;0.4-0.6作为验证，0-0.4+0.6-1作为训练;
#          0.6-0.8作为验证，0-0.6+0.8-1作为训练;0.8-1作为验证，0-0.8作为训练 这五种情况分别带入M1中
# 求取M1这样取超参数的情况下 上述5种情况的验证集的准确率(此时由于数据不同，训练出来的模型参数不同，但没有影响)，并求取这5个准确率的均值作为指标得到score1
# 同理将M2的超参数选择方式重复上述操作得到score2，比较score1和score2，当score1的值较大时说明M1这种超参数选择方式优于M2

# 此时就用M1的超参数选择方式在整一个数据集上训练，得到的一个较好模型的参数
import copy

import torch
import single_train
import random


def k_fold(each_give_param, train_dataset, device, k=5):  # 一种参数选择方式输出的分数
    data = []
    for x,y in train_dataset:
        data.append((x,y))
    random.shuffle(data) # 在此将顺序进行打乱
    epoch_length = len(data) // k
    train_acc_lst, valid_acc_lst = [], []
    for i in range(k):
        if i == 0:
            valid_iter = data[:epoch_length]
            deal_train_iter = data[epoch_length:]
        elif i == k - 1:
            valid_iter = data[i * epoch_length:]
            deal_train_iter = data[:i * epoch_length]
        else:
            valid_iter = data[i * epoch_length: (i + 1) * epoch_length]
            deal_train_iter = data[:i * epoch_length] + data[(i + 1) * epoch_length:]
        valid_acc, _ = single_train.train_model(each_give_param,
                                             deal_train_iter, valid_iter, device, is_print=False)
        valid_acc_lst.append(valid_acc)
    return sum(valid_acc_lst) / len(valid_acc_lst)

def func(give_param, device): # 确认在give_param的所有的参数选择中哪一种是最优的，以及最后在测试集上的准确率
    train_iter, test_iter = single_train.load_image(batch_size=128)
    result_param = copy.copy(give_param[0])
    result_score = 0
    # 取出验证集和训练集，训练集的个数:验证集的个数=4
    for each_give_param in give_param:
        valid_acc = k_fold(each_give_param, train_dataset=train_iter, device=device, k=5)
        num_epoch = each_give_param['num_epoch']
        lr = each_give_param['lr']
        dropout = each_give_param['dropout']
        print(f'num_epoch:{num_epoch},lr:{lr:.3f},dropout:{dropout:.3f},valid_acc:{valid_acc:.4f}')
        if result_score < valid_acc:
            result_score = valid_acc
            result_param = copy.copy(each_give_param)
    print('*' * 100)
    # 之后将较优的参数选择方式带入测试集进行测试
    test_acc, net = single_train.train_model(result_param, train_iter, test_iter, device)
    return result_param,test_acc

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 给定参数选择的方式
    give_param = [{'num_epoch':5, 'lr':0.1, 'dropout':0.5},{'num_epoch':3, 'lr':0.05, 'dropout':0.4}]
    result_param,test_acc = func(give_param, device) # test_acc=0.9648
    print(test_acc)
    print(result_param)
    test_acc1,_ = single_train.train_model(give_param[1],train_iter,test_iter,device,is_print=False)
    # test_acc1=0.9324，此时结果和K-折交叉验证结果两种参数的分数比较结果相同
    print(test_acc1)



