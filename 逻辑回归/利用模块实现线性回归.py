import torch
from d2l import torch as d2l
from torch import nn
d2l.use_svg_display()
#取出每一个mini_batch
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
#创建网络
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))  # Flatten的作用就是将最开始的输入进行平铺成二维的矩阵，即变成（shape[0],-1）,相当于加入了一个flatten layer
#进行初始化操作
def init_weight(m):
    if type(m)==nn.Linear:
        #nn.init.normal_(m.weight,std=0.01)
        m.weight.data.normal_(0,0.01)
        m.bias.data.fill_(0)
net.apply(init_weight)  # net.apply相当于将net中的每一层传入，此时应该传入两层，（Flatten，Linear），其中将Linear的参数进行了初始化
#交叉熵函数loss
loss=nn.CrossEntropyLoss() # 调用这个损失函数就是最后一层Linear(没有被softmax激活)，经过softmax激活后求交叉熵,且此时求解的是平均值
#此时nn.CrossEntropyLoss()的第一个参数是y_out(线性层的输出)，此时得到的是一个均值
#梯度更新和下降
trainer=torch.optim.SGD(net.parameters(),lr=0.1)

#进行预测
def accuracy(y,y_hat):  #此时的y_hat指代的时Linear的输出（没有经过softmax的激活），这样也是可以的，进行softmax激活并不会影响大小关系，大的还是大的
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)    #确定y_hat和y的数据类型是一样的
    return cmp.type(y.dtype).sum()

class Accumulator():  #定义一个累加器
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, index):
        return self.data[index]

def evaluate_accuracy(net,deal_data):  #用于测试集上
    if isinstance(net,nn.Module):
        net.eval()
    matric=Accumulator(2)
    for X,y in deal_data:
        matric.add(accuracy(y,net(X)),y.numel())
    return matric[0]/matric[1]

#进行单次的迭代
def train_epcho_ch3(net,trainer,iters_train):
    if isinstance(net,nn.Module):
        net.train()
    matric=Accumulator(3)
    for X,y in iters_train:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.mean().backward()
            trainer.step()
            matric.add(float(l*len(y)),accuracy(y,y_hat),y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]

for i in range(10):
    train_matric=train_epcho_ch3(net,trainer,iters_train=train_iter)
    test_acc=evaluate_accuracy(net,test_iter)
    train_loss,train_accuracy=train_matric
    print(train_loss)
    print(train_accuracy)
    print(test_acc)
    print('*'*50)