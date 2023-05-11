#在模型的训练过程中，如果没有明确的指明用gpu运算，使用的就是cpu
#网络（网络的参数）的创建是在cpu上的，我们要利用.to将网络移动到gpu上，同时输入的数据也需要在gpu上，这样整个网络就在gpu上计算
import torch
from torch import nn
import random
from torch.utils import data
print(torch.cuda.device_count())  #来显示可用的gpu的数量(该电脑上只有一个gpu)
def try_gpu():
    if torch.cuda.device_count()==1:
        return torch.device('cuda')  #代表的是利用gpu进行运算
    else:
        return torch.device('cpu')   #代表的是利用gpu进行运算

'''
x = torch.randn(5, 4,device=try_gpu())
net=nn.Sequential(nn.Linear(4,1))
net.to(device=try_gpu())
print(net[0].weight.NEWS)
'''


def synthetic_data(sample_num):
    w = torch.tensor([[2], [-3.4]])
    b = 4.2
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,[sample_num,1])  # [sample_num,1],这一层只有一个神经元
    return X,Y

features,labels=synthetic_data(sample_num=1000)
features=features.to(device=try_gpu())
labels=labels.to(device=try_gpu())

def load_array(X,Y,batch_size):
    num=X.shape[0]
    lst=list(range(num))
    random.seed(1)
    random.shuffle(lst)
    for i in range(0,num,batch_size):
        index=lst[i*batch_size:min((i+1)*batch_size,num)]
        yield X[index,:],Y[index,:]

net=nn.Sequential(nn.Linear(2,1))
net=net.to(device=try_gpu())
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
loss=nn.MSELoss()

trainer=torch.optim.SGD(net.parameters(),lr=0.03)

epoch_num=20
for i in range(epoch_num):
    for X,y in load_array(features,labels,batch_size=10):
        if isinstance(net,nn.Module):
            net.eval()
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
    with torch.no_grad():
        epoch_loss=loss(net(features),labels)
        print(f'第{i + 1}次迭代的损失:{epoch_loss.item():1.6f}')
print(net[0].weight.data)