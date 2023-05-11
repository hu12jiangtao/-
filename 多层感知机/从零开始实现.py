import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()

#生成SGD使用的小批量数据
batch_size=256
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
#设置初始函数
input_lawer,hidden_layer,output_layer=784,256,10
def init_parameters(input_lawer,hidden_layer,output_layer):
    w1=torch.normal(0,0.01,[input_lawer,hidden_layer],requires_grad=True)
    w2=torch.normal(0,0.01,[hidden_layer,output_layer],requires_grad=True)
    b1=torch.zeros([1,256],requires_grad=True)
    b2=torch.zeros([1,10],requires_grad=True)
    parameters=[w1,b1,w2,b2]
    return parameters
parameters=init_parameters(input_lawer,hidden_layer,output_layer)
#设计网络
def relu(x):
    a=torch.zeros_like(x)
    return torch.max(a,x)  # torch.max（）中的元素为两个矩阵时，会比较两个矩阵中对应的值，取较大的构成一个矩阵

def net(X,parameters):  #此时求得y_hat,@也可以用来表示矩阵的乘法操作
    layer_2_out=relu(X.reshape(-1,parameters[0].shape[0]) @ parameters[0]+parameters[1])
    output=layer_2_out @ parameters[2]+parameters[3]
    return output

#设计损失函数
loss=nn.CrossEntropyLoss()  #此时求解的就是均值

#设置梯度更新
def update_param(params,lr=0.1):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad
            param.grad.zero_()

#建立一个模型评估
def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]  # 对于zip来说执政对列表或者元组，对于一个矩阵a来说 for i in zip(a) 和 for i in a相同
    def __getitem__(self, item):
        return self.data[item]

def evaluate_accuracy(deal_data,parameters):
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(2)
    for X,y in deal_data:
        matric.add(accuracy(net(X,parameters),y),y.numel())
    return matric[0]/matric[1]

def train_epcho_ch3(net,accuracy,update_param,train_iters,parameters):
    if isinstance(net,nn.Module):
        net.train()
    matric=add_machine(3)
    for X,y in train_iters:
        y_output=net(X,parameters)
        l=loss(y_output,y)
        if isinstance(update_param,torch.optim.Optimizer) == False:
            l.backward()
            update_param(parameters)
            matric.add(l*len(y),accuracy(y_output,y),y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]

epcho_num=10
for i in range(10):
    train_loss,train_acc=train_epcho_ch3(net, accuracy, update_param, train_iters, parameters)
    test_acc=evaluate_accuracy(test_iters,parameters)
    print(f'当前迭代次数:{i + 1}')
    print('train_loss:', train_loss)
    print('train_acc:', train_acc)
    print('test_acc:', test_acc)
    print('*' * 50)

