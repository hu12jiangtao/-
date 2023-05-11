import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
d2l.use_svg_display()

def drop_out(input_num,drop):
    if drop == 0: # 不进行丢弃
        return input_num
    elif drop == 1:
        return torch.zeros_like(input_num.shape)
    else:
        a=torch.rand(size=input_num.shape,device=input_num.device)
        cmp=(a>drop)
        return input_num*cmp/(1-drop)

class dropout(nn.Module):
    def __init__(self,drop):
        super(dropout, self).__init__()
        self.drop = drop
    def forward(self,X):  # 此时出现net.eval()会自动变为False
        #print(self.training)
        if self.training:
            X=drop_out(X,drop=self.drop)
        else:
            X=drop_out(X,drop=0)
        return X

class simple_net(nn.Module):
    def __init__(self,num_input,num_hiddens1,num_hiddens2,num_hiddens3,output_num,drop):
        super(simple_net, self).__init__()
        self.l1=nn.Linear(num_input,num_hiddens1)
        self.l2=nn.Linear(num_hiddens1,num_hiddens2)
        self.l3=nn.Linear(num_hiddens2,num_hiddens3)
        self.l4=nn.Linear(num_hiddens3,output_num)
        self.dropout1 = nn.Dropout(drop)
        self.dropout2 = dropout(drop)
        self.dropout3 = dropout(drop)
    def forward(self,X):
        Y = F.relu(self.l1(X.reshape(X.shape[0],-1)))
        Y = self.dropout1(Y)

        Y = F.relu(self.l2(Y))
        Y = self.dropout2(Y)

        Y = F.relu(self.l3(Y))
        Y = self.dropout3(Y)

        return self.l4(Y)




#超参数的选择
device=d2l.try_gpu()
num_epcho,lr,batch_size=5,0.5,256
#数据的生成
train_iters,test_iters=d2l.load_data_fashion_mnist(batch_size)
drop = 0.5
num_input,num_hiddens1,num_hiddens2,num_hiddens3,output_num =784,256,256,100,10
net=simple_net(num_input,num_hiddens1,num_hiddens2,num_hiddens3,output_num,drop)
print(net)
class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def evaluate(y_hat,y):
    a=torch.argmax(y_hat,dim=1)
    cmp=(a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def accuracy_evaluate(net,data_iter,device):
    metric=add_machine(2)
    net.eval()
    net.to(device)
    for X,y in data_iter:
        X,y=X.to(device),y.to(device)
        y_hat = net(X)
        metric.add(evaluate(y_hat, y),y.numel())
    return metric[0]/metric[1]

def train(data_iter,num_epoch,lr,device):
    def init_param(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_param)
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    loss=nn.CrossEntropyLoss()
    net.to(device)
    for epoch in range(num_epoch):
        metric=add_machine(3)
        for X,y in data_iter:
            X,y=X.to(device),y.to(device)
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l,evaluate(y_hat,y),y.numel())
        print('当前迭代次数:',epoch)
        print('train_loss:',metric[0]/metric[2])
        print('train_accuracy:',metric[1]/metric[2])
        print('测试样本准确率:',accuracy_evaluate(net,test_iters,device))

train(train_iters,num_epcho,lr,device)