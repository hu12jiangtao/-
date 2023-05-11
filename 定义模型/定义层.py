from torch import nn
import torch
import numpy as np
from torch.utils import data

class MyLinear(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        #初始化参数
        torch.random.manual_seed(1)
        self.weight=nn.Parameter(torch.randn(input,output,requires_grad=True))
        self.bias=nn.Parameter(torch.zeros(1,output,requires_grad=True))
    def forward(self,X):
        return torch.matmul(X,self.weight)+self.bias      #注意此时利用的是self.weight和self.bias，如果利用self.weight.data和self.bias.data会导致梯度不更新




def synthetic_data(w,b,sample_num=1000):  #这一步是为生成随机的特征以及与之对应的标签
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,[sample_num,1])  # [sample_num,1],这一层只有一个神经元
    return X,Y

w=torch.tensor([[2],[3.4]])
b=torch.tensor([[-5]])
features,labels=synthetic_data(w,b)

#利用模板生成小批量的数据
def mini_batch(data_array,batch_size=10,is_train=True):
    dataset=data.TensorDataset(*data_array)   # dataset=NEWS.TensorDataset(data_array[0],data_array[1])
    # TensorDataset的作用生成一个元组，元组的长度为其中一个的shape[0]，元组a索引中的内容为一个元组，包含data_array[0]该索引下的tensor，和data_array[1]索引下的tensor
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
items= mini_batch([features,labels],batch_size=10,is_train=True)
#print(next(iter(items)))

#构建一个网络
#net=nn.Sequential(MyLinear(2,2),nn.Linear(2,1))
net=nn.Sequential(MyLinear(2,1))
net[0].weight.data.normal_(0,0.01) #初始化训练参数
net[0].bias.data.fill_(0)
loss=nn.MSELoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)
epcho_num=3
for i in range(epcho_num):
    for X,Y in items:
        l=loss(net(X),Y)
        trainer.zero_grad()  # 反向传播步骤一
        l.backward() # 反向传播步骤二
        trainer.step() #参数更新
    with torch.no_grad():
        epcho_loss=loss(net(features),labels)
        print(f'第{i+1}次迭代的损失:{epcho_loss:1.6f}')
print(w)

'''
#自己定义一个线性层
class MyLinear(nn.Module):
    def __init__(self,input,output):
        super().__init__()
        #初始化参数
        torch.random.manual_seed(1)
        self.weight=nn.Parameter(torch.randn(input,output))
        self.bias=nn.Parameter(torch.zeros([1,output]))
    def forward(self,X):
        return torch.matmul(X,self.weight.NEWS)+self.bias.NEWS
net=nn.Sequential(MyLinear(2,1))
def xavier(m):
    if type(m)==MyLinear:
        nn.init.xavier_uniform_(m.weight)
net.apply(xavier)
'''



#print(net.__call__(X))  #等价于net(x)，就是他的缩写，在父类的__call__中调用了forward
