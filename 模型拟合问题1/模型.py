#训练数据集，测试数据集，验证数据集
#整个网络搭建的过程为在训练数据集中进行超参数的调整，认为调整不错后我们需要放到验证数据集中进行验证（验证数据集和训练数据集中的内容应当全部不相同）
#当在验证数据集中得到的结果较好时，就完成了这个模型，最后拿着这个模型去测试数据集中测试，测试数据集（只用一次），训练数据集和验证数据集在训练参数时会多次使用
#在我们设计模型的代码中test_iters应当是一个验证数据集（与测试数据集的准确率来说还是偏高的）

#当数据的个数较少时，可以利用k-则交叉验证的方法（k一般可以取5或者10）：
#一段数据分成k段，我们设计好模型的参数进行k次的循环，第一次去第一段数据作为验证数据集，其余作为训练数据集；重复这种操作k次，最后一次为取前k-1次数据作为训练数据，第k段作为验证数据
#最后求取所有模型的平均值作为超参数的评估标准

#过拟合和欠拟合：当数据复杂时应当选择较深（复杂）的模型，而数据较为简单的时候应当选择较浅（简单）的模型，较深（复杂）的模型极端情况指的是可以记住训练集的所有点
#数据复杂选择较浅的模型会欠拟合（训练，验证准确率不高），数据较为简单用较深的模型会导致过拟合（训练集准确率远大于验证集）
#泛化误差是指在验证集上的误差，训练误差是在训练集上的误差
#数据复杂度和样本的个数，特征的个数



from d2l import torch as d2l
import numpy as np
import torch
import math
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
def create_data():
    max_degree=20
    n_train,n_test=100,100  # 训练标签和验证标签
    true_w=np.zeros([1,max_degree])   #正确的值，等会需要和训练得到的w进行比较
    true_w[:,0:4]=5,1.2,-3.4,5.6
    features=np.random.normal(size=(n_train+n_test,1))  # 已拥有200个样本的x的值，目标样本维度为200*20
    np.random.shuffle(features) # 将这些样本给打乱
    plot_features=np.power(features,np.arange(max_degree).reshape(1,-1))  # 此时的plot_feature代表的是200*20，每一个样本[1,x,x^2,....,x^19]
    for i in range(max_degree):
        plot_features[:,i]/=math.gamma(i+1)  # math.gamma(i+1)代表的是其阶层
    labels=np.sum(np.multiply(true_w,plot_features),axis=1,keepdims=True)#此时生成的是标准的label
    labels+=np.random.normal(scale=0.1,size=labels.shape)
    #将输入，输出转换为tensor型的，后进行mini_batch操作
    labels=torch.tensor(labels,dtype=torch.float32)
    plot_features=torch.tensor(plot_features,dtype=torch.float32)
    print(labels.shape)  #此时的输入为200*20，输出为200*1
    return plot_features,labels

#通过torch中的函数对其进行每一个样本的特征和标签进行匹配
def load_array(data_array,batch_size=10,is_train=True):
    dataset=data.TensorDataset(*data_array)  #利用TensorDataset指令需要将其中的内容转换为tensor
    return data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train)

#参数的初始化
def init_params(m):   #当参数训练完成后可以利用net[i].weight.data进行查看
    if type(m)==nn.Linear:
        m.weight.data.normal_(0,0.01)

#进行loss的评估
class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def evaluate_loss(loss,net,data_iter):   #用在了测试集的评估上
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(2)
    for X,y in data_iter:
        y_out=net(X)
        l=loss(y_out,y)
        matric.add(l*len(y),y.numel())
    return matric[0]/matric[1]

#构建一个模型
def train_epcho_ch3(net,train_iter,loss,trainer):
    if isinstance(net,nn.Module):
        net.train()
    matric=add_machine(2)
    for X,y in train_iter:
        y_out=net(X)
        l=loss(y_out,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*len(y),y.numel())
    return matric[0]/matric[1]

def train(train_features,test_features,train_labels,test_labels):
    #将训练集，测试集打包成一个dataset
    train_iter=load_array([train_features,train_labels], batch_size=10, is_train=True)
    test_iter=load_array([test_features,test_labels],batch_size=10,is_train=False)
    feature_num=train_features.shape[1]
    net=nn.Sequential(nn.Linear(feature_num,train_labels.shape[1],bias=False))
    net.apply(init_params)
    loss=nn.MSELoss()
    trainer=torch.optim.SGD(net.parameters(),lr=0.03)
    epcho_num=4000
    train_list=[]
    test_list=[]
    for i in range(1,1+epcho_num):
        if i % 50 == 0:
            train_loss=train_epcho_ch3(net,train_iter,loss,trainer)
            test_loss=evaluate_loss(loss,net,test_iter)
            train_list.append(train_loss)
            test_list.append(test_loss)
            print(f'当前迭代次数:{i}')
            print(train_loss)
            print(test_loss)
            print('*'*50)
    plt.plot(np.arange(len(train_list)),train_list)
    plt.plot(np.arange(len(test_list)), test_list)
    plt.show()
    print(net[0].weight.data.numpy())

n_train,n_test=100,100
plot_features,labels=create_data()
train(plot_features[:n_train,:20],plot_features[n_train:,:20],labels[:n_train],labels[n_train:])
#输入的四个参数分别为 输入的前100个作为训练样本，对应labels[:n_train]这100个标签，同理测试利用样本plot_features后100个，对应
