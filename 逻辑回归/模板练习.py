import torch
from d2l import torch as d2l
from torch import nn
d2l.use_svg_display()
#小批量的生成
batch_size=256
train_iter,test_iter=d2l.load_data_fashion_mnist(batch_size)
#生成网络
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
#生成初始化参数
def init_param(m):
    if type(m)==nn.Linear:
        m.weight.data.normal_(0,0.01)
        m.bias.data.fill_(0)
net.apply(init_param)

#生成损失函数
# loss=nn.CrossEntropyLoss()  #此时返回的是一个均值

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self,y_hat,labels,smoothing=0.1):
        final_out = torch.log_softmax(y_hat,dim=-1)
        smoothing_loss = torch.mean(final_out,dim=-1) * smoothing  # [batch,]
        nll_out = torch.gather(final_out,index=labels.reshape(-1,1),dim=-1)
        nll_out = nll_out.squeeze(1)
        nll_loss = (1 - smoothing) * nll_out
        sum_loss = nll_loss + smoothing_loss
        return - torch.mean(sum_loss)
# 下面是利用了label smoothing,虽然在验证中利用了交叉熵损失函数、训练中利用了自定义的label smoothing函数
# 但是随着训练时的label smoothing损失的下降,在验证集中的交叉熵损失一样下降
loss = LabelSmoothingCrossEntropy()

#构建更新方式
trainer=torch.optim.SGD(net.parameters(),lr=0.1)

#进行模型准确率的判断
def accuracy(y_hat,y):
    y_hat=torch.argmax(y_hat,dim=1)
    cmp=(y_hat.type(y.dtype)==y)
    return (cmp.type(y.dtype)).sum()

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self, *args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def evaluate_accuracy(net,accuracy,deal_data):
    if isinstance(net,nn.Module):
        net.eval()
    matric=add_machine(3)
    loss = nn.CrossEntropyLoss()
    for X,y in deal_data:
        y_hat = net(X)
        l = loss(y_hat,y)
        matric.add(accuracy(y_hat,y),l * y.numel(),y.numel())
    return matric[0]/matric[2],matric[1] / matric[2]  #准确率

def train_epcho_ch3(trainer,net,loss,train_iter):
    if isinstance(net,nn.Module):
        net.train()
    matric=add_machine(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(trainer,torch.optim.Optimizer):
            trainer.zero_grad()
            l.backward()
            trainer.step()
            matric.add(l*len(y),accuracy(y_hat,y),y.numel())
    return matric[0]/matric[2],matric[1]/matric[2]

epcho_num=10
for i in range(epcho_num):
    train_loss,train_acc=train_epcho_ch3(trainer, net, loss, train_iter)
    test_acc, test_loss=evaluate_accuracy(net,accuracy,test_iter)
    print(f'当前迭代次数:{i+1}')
    print('train_loss:',train_loss)
    print('train_acc:', train_acc)
    print('test_loss:', test_loss)
    print('test_acc:', test_acc)
    print('*'*50)

