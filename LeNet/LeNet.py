import torch
from torch import nn
from d2l import torch as d2l
d2l.use_svg_display()

print(1)
torch.random.manual_seed(1)
#此时我们需要在gpu上进行运算,如果计算机上存在gpu，可以利用try_gpu去进行调用
def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device(f'cuda:{0}')
    else:
        return torch.device('cpu')

batch_size=256
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)  #图片的宽高是28*28


def MyModel():
    net=nn.Sequential()
    net.add_module('conv1',nn.Conv2d(1,6,padding=2,kernel_size=5))
    net.add_module('activate1',nn.Sigmoid())
    net.add_module('mean_pool1',nn.AvgPool2d(stride=2,kernel_size=2))
    net.add_module('conv2',nn.Conv2d(6,16,kernel_size=5))
    net.add_module('activate2',nn.Sigmoid())
    net.add_module('mean_pool2',nn.AvgPool2d(kernel_size=2,stride=2))
    net.add_module('flatten',nn.Flatten())
    net.add_module('Linear1',nn.Linear(16*5*5,120))
    net.add_module('activate3',nn.ReLU())
    net.add_module('Linear2',nn.Linear(120,84))
    net.add_module('activate4',nn.ReLU())
    net.add_module('Linear3',nn.Linear(84,10))
    return net
net=MyModel()
'''
#可以查看每一层的权重
for layer in net:
    X=layer(X)
    print(layer.__class__.__name__,'output_shape:\t',X.shape)
'''
class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(net,deal_date,device):
    if isinstance(net,nn.Module):
        net.eval()
    net.to(device)
    matric=add_machine(2)
    for X,y in deal_date:
        if isinstance(X,list):
            X=[x.to(device) for x in X]  #如果这个X是一个列表
        else:
            X=X.to(device)
        y=y.to(device)
        y_hat=net(X)
        matric.add(accuracy(y_hat,y),y.numel())
    return matric[0]/matric[1]

def train_ch6(net,train_iters,test_iters,num_epoch,lr,device):
    def inint_params(m):
        if type(m)==nn.Linear or type(m)==nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(inint_params)
    net.to(device)
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    print('training on:',device)
    animator = d2l.Animator(xlabel='iter', xlim=[1, 20],
                            legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epoch):
        matric=add_machine(3)
        net.train()
        batch_iter = 0
        for X,y in train_iters:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*y.shape[0],accuracy(y_hat,y),y.numel())
            batch_iter += 1
        epoch_train_loss=matric[0]/matric[2]
        epoch_train_acc = matric[1] / matric[2]
        epoch_test_acc=evaluate_accuracy(net,test_iters,device)
        animator.add(epoch,
                     (epoch_train_loss, epoch_train_acc, None))

        print(f'第{epoch+1}次迭代:')
        print('train loss:',epoch_train_loss)
        print('train_acc:', epoch_train_acc)
        print('test_acc:', epoch_test_acc)
        print('*'*50)

lr,num_epoch=0.3,20
train_ch6(net,train_iters,test_iters,num_epoch,lr,device=try_gpu())

