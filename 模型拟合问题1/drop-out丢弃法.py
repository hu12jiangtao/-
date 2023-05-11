#其效果会比L2正则化的效果更好

#drop_out的核心是意思是在层之间加入噪音，x为前一层至后一层的输出，x'为前一层至后一层的加入噪音的输出，我们需要保证两者的期望不发生变化

#给定一个概率p，在p的概率里面的神经元的输入变为0，为了保证期望不变，其他神经元的输入应当增大至x(i)/(1-p)

#例子：原始的数据输入a[1]，经过了第二层的神经元（第二层有5个神经元）的激活，得到了a[2],在不添加drop_out的时候a[2]作为输入传入第二层神经元，在添加drop_out层（p=0.5）后（例如p=0.5）
#可能会丢弃掉第二层中的第1，3个神经元（这两个神经元的输出变为0），此时2，4，5神经元为了保证整体输出的期望不变，其输出的值应当扩大1/（1-0.5）倍

#drop_out等正则项只作用与训练数据的时候，在验证数据的时候是不需要的

#丢弃概率p来控制模型的复杂度，p越大效果越明显（常作用多层感知机上），最常见的是（0.9，0.5，0.1）

#此时正则化的效果可以通过loss函数体现出来，正则化后的loss会比没有正则化时的大（防止系统的过拟合）
import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()

def dropout_layer(X,drop_out):
    assert 0<=drop_out<=1
    if drop_out==1:
        return torch.zeros_like(X)
    if drop_out==0:
        return X
    cmp=(torch.rand(size=X.shape)>drop_out)
    return cmp*X/(1-drop_out)

class Net(nn.Module):  # drop_out无法单独定义一个层，或者定义起来很麻烦因此直接定义一个模型比较方便
    def __init__(self,num_inputs,num_hidden1,num_hidden2,num_hidden3,num_output,is_training=True): #在nn.Module的情况下net.eval()会将所有的is_training变为false
        super(Net, self).__init__()  # 继承了nn.Module父类的初始化,此时就不需要对网络的参数进行初始化操作
        self.num_inputs=num_inputs  #定义了输出神经网络的特征个数
        self.training=is_training
        self.lin1=nn.Linear(num_inputs,num_hidden1)
        self.lin2=nn.Linear(num_hidden1,num_hidden2)
        self.lin3=nn.Linear(num_hidden2,num_hidden3)
        self.lin4=nn.Linear(num_hidden3,num_output)
        self.relu=nn.ReLU() #定义后面使用的激活函数
    def forward(self,X):  #其中的X由外部的传参引入
        drop_out1,drop_out2,drop_out3=0.5,0.5,0.5
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        if self.training:
            H1=dropout_layer(H1, drop_out1)
        H2=self.relu(self.lin2(H1))
        if self.training:
            H2=dropout_layer(H2,drop_out2)
        H3=self.relu(self.lin3(H2))
        if self.training:
            H3=dropout_layer(H3,drop_out3)
        out=self.lin4(H3)
        return out

class test_net(Net):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
    def forward(self,X):
        H1 = self.relu(self.lin1(X.reshape(-1, self.num_inputs)))
        H2 = self.relu(self.lin2(H1))
        H3 = self.relu(self.lin3(H2))
        out = self.lin4(H3)
        return out


#超参数的选择
num_epcho,lr,batch_size=5,0.5,256
#数据的生成
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
#网络的构建
net=Net(num_inputs=784,num_hidden1=256,num_hidden2=256,num_hidden3=100,num_output=10)
Test_net=test_net(num_inputs=784,num_hidden1=256,num_hidden2=256,num_hidden3=100,num_output=10)
#损失函数的选择
loss=nn.CrossEntropyLoss()
#更新器的选择
trainer=torch.optim.SGD(net.parameters(),lr=lr)

#进行模型的评估
def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()
class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]
def accuracy_evaluate(net,deal_data):
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(2)
    for X,y in deal_data:
        y_hat=net(X)
        matric.add(accuracy(y_hat,y),y.numel())
    return matric[0]/matric[1]
def epcho_train_ch3(net,train_iters):
    if isinstance(net,nn.Module):
        net.train()
    matric=add_machine(3)
    for X,y in train_iters:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*len(y),accuracy(y_hat,y),y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]

for i in range(num_epcho):
    train_loss,train_acc=epcho_train_ch3(net,train_iters)
    test_acc=accuracy_evaluate(Test_net,test_iters)
    print(f'当前迭代次数为:{i+1}')
    print('train_loss:', train_loss)
    print('train_acc:', train_acc)
    print('test_acc:', test_acc)
    print('*' * 50)
