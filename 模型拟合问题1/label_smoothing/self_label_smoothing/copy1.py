import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils import data
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import resnet

# 将label smoothing后倒数第二层进行投影的可视化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 导入测试的数据集
trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
testset = torchvision.datasets.CIFAR10(root='D:\\python\\pytorch作业\\知识蒸馏\\my_self_cifir10_2\\dataset',
                                       train=False, download=False, transform=trans) # 原先的内容./data
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

# 此处可以选择不同的降维方式将 语意信息(模型的倒数第二层，激活函数为非线性的) 从高维降低至低维
# 常用的方法有T-SNE(其原理为对于高维的相近的点映射到二维平面时同样应该相近)
# PCA(求解语意信息的协方差，取协方差的特征数最大的对应的w作为重构的坐标轴)

def T_SNE(num_dim,inputs):
    t_sne = TSNE(n_components=num_dim,init='pca',random_state=0)
    outputs_array = t_sne.fit_transform(inputs) # [batch, num_dim]
    return outputs_array

#'''
def save_plt(num_dim,title,down_mode):
    #'''
    net = resnet.ResNet18()
    net.to(device)
    if title == 'no_smoothing':
        net.load_state_dict(torch.load('checkpoint/CrossEntropy.bin'))  # 此时用的是给定的的模型而不是自己的模型
    elif title == 'smoothing':
        net.load_state_dict(torch.load('checkpoint/LabelSmoothing.bin'))
    net.linear = nn.Flatten()
    #'''

    '''
    model = resnet.ResNet18()
    if title == 'no_smoothing':
        state = torch.load('checkpoint/CrossEntropy.bin') # 此时用的是给定的的模型而不是自己的模型
    elif title == 'smoothing':
        state = torch.load('checkpoint/LabelSmoothing.bin')
    model.load_state_dict(state)  # 导入训练好的参数
    model.linear = nn.Flatten()  # 此时相当于去掉倒数第一层的线性层，最后model输出的应当是原本的第二层
    extract = model
    extract.cuda()
    extract.eval()
    '''


    out_target = []  # 其中每一个元素为[128,1]
    out_output = []  # 其中每一个元素为[128,512]
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs = inputs.to(device)
        outputs = net(inputs)
        outputs_np = outputs.data.cpu().numpy()
        targets_np = targets.data.numpy()
        out_output.append(outputs_np)
        out_target.append(targets_np.reshape(-1, 1))
    output_array = np.concatenate(out_output, axis=0)  # [128 * n, 512] = [10000, 512]
    targets_array = np.concatenate(out_target, axis=0)  # [128 * n, 1] = [10000, 1]

    outputs_array = T_SNE(num_dim,output_array)
    plt.rcParams['figure.figsize'] = 10,10
    plt.scatter(outputs_array[:,0],outputs_array[:,1],c=targets_array[:,0])
    plt.title(title)
    plt.savefig('image/'+title+'_'+down_mode+'.png', bbox_inches='tight')
#'''



if __name__ == '__main__':
    #save_plt(num_dim=2, title='no_smoothing', down_mode='TSNE')
    save_plt(num_dim=2, title='smoothing', down_mode='TSNE')








