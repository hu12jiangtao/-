# batch_normal解决的是反向传播造成顶层的权重的梯度大，底层的的梯度小（特别是深层的神经网络），因此顶层参数在底层参数变化
# 但是底层的参数改变时最后几层的权重需要重新训练
# 简单来讲就是不做batch_normal是第一层的方差和均值你归一化了为（0，1），到最后几层你发现你的均值变为了（65，1000），差别很大
# batch_normal做的事情就是将整一个神经网络的所有层的均值和方差变为到一个相似的范围
# 批量归一化层要在全连接或者卷积层的激活函数前面，其优点是可以加快收敛的速度，使用更大的学习率，不会改变模型精度,同时可以减小模型的过拟合

# 批量归一化在全连接层上是作用在特征维上，而对于卷积神经网络是作用在通道维上
# 可以用1*1的卷积来想象:1*1卷积干的就是通道的融合,每一个通道相当于全连接一个样本的一个特征,4个通道相当于4个特征,此时输入为3*28*19*19，相当于3*19*19个样本，每个样本28个特征
# 对于图片来说通道数就是特征数量，一层的元素个数相当于样本个数
# 因此对于卷积层来说一个面的所有像素相当于输入的所有样本的一个特征

# batch_norm会划分为训练模式和测试的模式，因为在测试模式下输入可能为单样本，如果利用这个batch的均值或者方差会算不出来
# 同时对于缩放的gamma和beta的维度应当等价于x_mean,x_var的维度，对于四维来说应当是（1,channal_num,1,1）,对于二维来说应当是(1,feature_num)

#同时由于batch_norm的存在，其中的beta替换了原来卷积中的偏执b，因此在存在batch_norm层就可以不设置卷积层的b（重点）

from torch import nn
import torch
from d2l import torch as d2l

d2l.use_svg_display()


def batch_norm(X,gamma,beta,moving_mean,moving_var,epsilon,momentum):
    #print(X.shape)
    if not torch.is_grad_enabled():  #在测试的阶段
        X_hat=(X-moving_mean)/(moving_var+epsilon)
    else:
        assert len(X.shape) in (2,4)  #只针对全连接或者是2d的卷积
        if len(X.shape)==2:
            X_mean=X.mean(dim=0,keepdim=True)
            X_var=((X-X_mean)**2).mean(dim=0,keepdim=True)
        else:
            X_mean=torch.mean(X,dim=(0,2,3),keepdim=True)
            X_var=torch.mean((X-X_mean)**2,dim=(0,2,3),keepdim=True)
        X_hat=(X-X_mean)/(X_var+epsilon)
        moving_mean=(1-momentum)*X_mean+momentum*moving_mean
        moving_var=(1-momentum)*X_var+momentum*moving_var
    Y=gamma*X_hat+beta
    return Y,moving_mean,moving_var


class BatchNormal(nn.Module):
    def __init__(self,in_channal,num_dims):
        super().__init__()
        assert num_dims in (2,4)
        if num_dims==2:
            shape=(1,in_channal)
        else:
            shape=(1,in_channal,1,1)
        self.gamma=nn.Parameter(torch.ones(shape))
        self.beta=nn.Parameter(torch.zeros(shape))
        self.moving_mean=torch.zeros(shape,dtype=torch.float32) #此时利用self.moving_mean是因为每次迭代self.moving_mean的值不会释放会一直存着
        self.moving_var=torch.zeros(shape,dtype=torch.float32)
    def forward(self,X):
        if self.moving_mean.device != X.device:
            self.moving_mean=self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y, self.moving_mean, self.moving_var=batch_norm(X,self.gamma,self.beta,self.moving_mean,self.moving_var,epsilon=1e-5,momentum=0.9)
        return Y



#此处利用的是LeNet ,这里的每一个BatchNormal都是同类的不同案例
net=nn.Sequential(nn.Conv2d(1, 6, kernel_size=5), BatchNormal(6, num_dims=4), nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(6, 16, kernel_size=5), BatchNormal(16, num_dims=4), nn.Sigmoid(),
                  nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
                  nn.Linear(16*4*4, 120), BatchNormal(120, num_dims=2), nn.Sigmoid(),
                  nn.Linear(120, 84), BatchNormal(84, num_dims=2), nn.Sigmoid(),
                  nn.Linear(84, 10))

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

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


def evaluate_accuracy(net,deal_data,device):
    with torch.no_grad():
        if isinstance(net,nn.Module):
            net.eval()
        matric=add_machine(2)
        for X,y in deal_data:
            if isinstance(X,list):
                X=[x.to(device) for x in X]
            else:
                X=X.to(device)
            y=y.to(device)
            y_hat=net(X)
            matric.add(accuracy(y_hat,y),y.numel())
    return matric[0]/matric[1]

def train(net,train_iters,test_iters,epoch_num,lr,device):
    def init_params(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_params)
    loss=nn.CrossEntropyLoss()
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    net.to(device)
    print('is training:',device)
    matric=add_machine(3)
    for epoch in range(epoch_num):
        net.train()
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*y.shape[0],accuracy(y_hat,y),y.numel())
        test_acc=evaluate_accuracy(net,test_iters,device=try_gpu())
        print('当前迭代的次数:',epoch+1)
        print('train loss:',matric[0]/matric[2])
        print('train acc:', matric[1] / matric[2])
        print('test acc:', test_acc)
        print('*'*50)

batch_size=256
device=d2l.try_gpu()
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
train(net,train_iters,test_iters,epoch_num=20,lr=1,device=try_gpu())