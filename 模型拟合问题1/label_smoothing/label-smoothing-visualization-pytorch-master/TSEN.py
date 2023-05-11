import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import resnet as RN
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--ce', action='store_true', help='Cross Entropy use')
args = parser.parse_args()

model = RN.ResNet18()
#'''
if args.ce == True:
    path = './checkpoint/CrossEntropy.bin'
    npy_path = './CE.npy'
    npy_target = './CE_tar.npy'
    title = 'TSNE_CrossEntropy'
    states = torch.load(path)
else:
    path = './checkpoint/LabelSmoothing.bin'
    npy_path = './LS.npy'
    npy_target = './LS_tar.npy'
    title = 'TSNE_LabelSmoothing'
    states = torch.load(path)

model.load_state_dict(states)  # 导入训练好的参数
model.linear = nn.Flatten() # 此时相当于去掉倒数第一层的线性层，最后model输出的应当是原本的第二层

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='D:\\python\\pytorch作业\\知识蒸馏\\my_self_cifir10_2\\dataset',
                                       train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

extract = model
extract.cuda()
extract.eval()

out_target = []
out_output = []

for batch_idx, (inputs, targets) in enumerate(testloader):
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = extract(inputs)  # 相当于得到了倒数第二层的输出结果
    output_np = outputs.data.cpu().numpy()  # [128,512]
    target_np = targets.data.cpu().numpy()  # [128,]
    out_output.append(output_np)
    out_target.append(target_np[:,np.newaxis]) # target_np[:,np.newaxis]等价于将其reshape为一个列向量

output_array = np.concatenate(out_output, axis=0)
target_array = np.concatenate(out_target, axis=0)
np.save(npy_path, output_array, allow_pickle=False)  # 用于进行数据的存储的
np.save(npy_target, target_array, allow_pickle=False)

#feature = np.load('./label_smooth1.npy').astype(np.float64)
#target = np.load('./label_smooth_target1.npy')

print('Pred shape :',output_array.shape)
print('Target shape :',target_array.shape)

tsne = TSNE(n_components=2, init='pca', random_state=0) # n_components指代的是降到了多少维度，2代表降到了二维
output_array = tsne.fit_transform(output_array) # [10000, 2]
print(output_array.shape)
plt.rcParams['figure.figsize'] = 10,10
plt.scatter(output_array[:, 0], output_array[:, 1], c= target_array[:,0])
plt.title(title)
plt.savefig('./'+title+'.png', bbox_inches='tight')
#'''

