import torch
from torch import nn
from torch.nn import functional as F

X=torch.randn(2,20)

#这些类的最终目标就是构建你需要的网络(net只求到输出层前一层为被softmax激活的时候)，就是Sequential中的层不能满足你的需求时，我们可以定义一个重新定义一个model或者类

class mymodel(nn.Module):  #这个函数就无法利用nn.Module中的现有层
    def __init__(self):
        super().__init__()
        self.init_weight=torch.randn((20,20),requires_grad=False)
        self.linear=nn.Linear(20,20)
    def forward(self,X):
        X=self.linear(X)
        X=F.relu(torch.matmul(X,self.init_weight)+1)
        X=self.linear(X)
        while X.abs().sum() > 1:
            X/=2
        return X.sum()
X=torch.randn((2,20))
model=mymodel()
print(model(X))

'''
class MLP(nn.Module):  #调用max(0,input)这个relu函数，一般F中的存放这些调用函数的
    def __init__(self):
        super().__init__()  # 继承父类的一些函数,此时创建一个实例  ，此时可以创建一个类net=MLP()，直接调用forward函数
        #原理是nn.Module的__call__()方法中含有forward(self,X)函数（定义死了），此时相当于把一个类的实例化对象变成了可调用对象，把forward的名称换为了forward_1就不行了
        self.hindden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)
    def forward(self,X):
        return self.out(F.relu((self.hindden(X))))  #此时仅仅是调用relu这个函数，就是实现max(0,input),而nn.ReLU代表的是创建一个relu层（用于模型的构建上）
net=MLP()
print(net(X))
print('*'*50)
'''

'''
class MLP1(nn.Module):  # 此为使用nn.ReLU的方法(在模型的构建是创建nn.ReLU层，之后在forward中计算)
    def __init__(self):
        super().__init__()  # 继承父类的一些函数,此时创建一个实例  ，此时可以创建一个类net=MLP()，直接调用forward函数
        #原理是nn.Module的__call__()方法中含有forward(self,X)函数（定义死了），此时相当于把一个类的实例化对象变成了可调用对象，把forward的名称换为了forward_1就不行了
        self.hindden=nn.Linear(20,256)
        self.out=nn.Linear(256,10)
        self.relu=nn.ReLU()
    def forward(self,X):
        return self.out(self.relu((self.hindden(X))))
net=MLP1()
print(net(X))   #等价于net(x)，就是他的缩写，在父类的__call__中调用了forward
print('*'*50)
'''

'''
#将Sequential和自己定义的类联合进行使用

class NestNLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(20,64),nn.ReLU(),nn.Linear(64,32),nn.ReLU())
        self.linear=nn.Linear(32,16)
    def forward(self,x):
        return self.linear(self.net(X))
net=nn.Sequential(NestNLP(),nn.Linear(16,20))
#print(net(X))

#访问参数
#print(net[1].state_dict())
#print(net[1].weight)  #得到的结果是一个元组，其中有两个元素，一个存放数据，另一个存放requires_grad的状态
#print(net[1].weight.grad)
print(net)   #可以用来查看整个网络的结构
'''

'''
#另一种添加网络模型的的方式 ,利用这种方式对网络的每一层都可以进行命名
def block2():
    net=nn.Sequential()
    net.add_module('block1',nn.Linear(20,10))
    net.add_module('block2',nn.ReLU())
    print(net)
    return net
net=block2()

#对网络进行一个初始化
def init_params(m):
    if type(m)==nn.Linear:
        m.weight.NEWS.normal_(mean=0,std=0.01)  #其中normal_中的下划线代表的是进行替换操作
net.apply(init_params)  #apply的作用相当于遍历调用
'''


#实现对网络的模块化，并且对网络的各个模块进行查看，并且赋初值
def block1_1():
    net=nn.Sequential()
    net.add_module('block1_1_1',nn.Linear(20,15))
    net.add_module('activate1',nn.ReLU())
    net.add_module('block1_1_2',nn.Linear(15,10))
    net.add_module('activate2',nn.ReLU())
    return net
def block1_2():
    net = nn.Sequential()
    net.add_module('block1_2_1',nn.Linear(20,15))
    net.add_module('activate1',nn.ReLU())
    return net

class block1(nn.Module):
    def __init__(self,block1_1,block1_2):
        super().__init__()
        self.block1_1=block1_1()
        self.block1_2=block1_2()
    def forward(self,X):
        return self.block1_2(self.block1_1(X))
def block2():
    net = nn.Sequential()
    net.add_module('block2_1',nn.Linear(15,8))
    net.add_module('activate1',nn.ReLU())
    return net
net=nn.Sequential()
net.add_module('block1',block1(block1_1,block1_2))
net.add_module('block2',block2())
print(net)

#初始化的方法
def init_params1(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)  #数据处理中的xavier取初始参数
net[0].apply(init_params1)  #这种方法就是对block1层中的所有参数进行初始化

print(net[1].state_dict())  # 可以用state_dict来查看每个块中的w



