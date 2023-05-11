import torch
from d2l import torch as d2l
from torch.utils import data
from torch import nn

def synthetic_data(w,b,sample_num=1000):  #这一步是为生成随机的特征以及与之对应的标签
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,[sample_num,1])  # [sample_num,1],这一层只有一个神经元
    return X,Y
w=torch.tensor([[2],[3.4]])
b=torch.tensor([[-5]])
features,labels=synthetic_data(w,b)
#利用现有的模具来实现样本的随机抽取
def load_array(data_array,batch_size=10,is_train=True):
    dataset=data.TensorDataset(*data_array)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)  #is_train的意思是确定打乱顺序
date_iter=load_array([features,labels])   #生成一个生成器
net=nn.Sequential(nn.Linear(2,1))  #将每一个层按顺序放到Sequential的容器中，此时net相当于一个列表，net[0]存放第一层
net[0].weight.data.normal_(0,0.01)  #给第一层网络定义这个层的权值的初始值
net[0].bias.data.fill_(0)

loss=nn.MSELoss()  #从库中调用均方误差 ,此时返回的应当是一个tensor的数
trainer=torch.optim.SGD(net.parameters(),lr=0.03)   # net.parameters()应当代表的是网络中的所有的参数

num_epcho=3
for i in range(num_epcho):
    for X,Y in date_iter:
        l=loss(net(X),Y)   #给定了网络后，有给定了输出，就有了输出y_hat
        #print(l)
        trainer.zero_grad()  # 将训练的参数的梯度首先全部清零
        l.backward()
        trainer.step()  # step()代表的是模型的更新
    with torch.no_grad():
        epcho_loss=loss(net(features),labels)
    print(f'第{i + 1}次迭代的损失:{epcho_loss:1.6f}')

