# rnn模块的简洁实现（和自己定义的rnn中的state，Y有所区别，此时的state=[1,batch_size,num_hiddens],在模块中的Y指代的是隐藏层的输出，shape=[seq,batch_size,num_hiddens]）
# 主要就是修正了从头开始的代码中的初始化权重和前向传播的过程
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import math

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device=try_gpu()
num_hiddens=512
num_steps=35
batch_size=32
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)

#定义整个网络
rnn_layer=nn.RNN(len(vocab),num_hiddens)

class RnnModel(nn.Module):
    def __init__(self,rnn_layer,vocab_size,num_hiddens):
        super().__init__()
        self.rnn_layer=rnn_layer  # 此时的输出为[序列长度，batch_size,vocab_size]
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens

        self.num_direction=1  # 这句话说明rnn为单向传播的
        self.linear=nn.Linear(self.num_hiddens,self.vocab_size)  # rnn的输出层

    def forward(self,X,state):
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32)  # 此时得到的X.shape=[seq,batch_size,vocab_size]
        Y,state=self.rnn_layer(X,state)  # 此时Y的输出为[seq,batch_size,num_hiddens]
        out=self.linear(Y.reshape(-1,Y.shape[-1]))
        return out,state

    def begin_state(self,batch_size,device):
        return torch.zeros([1,batch_size,self.num_hiddens],device=device)

def predict(give_vocab,num_need,net,vocab,device):
    state=net.begin_state(batch_size=1,device=device)  # 得到了初始的状态
    out=[vocab[give_vocab[0]]] # 存放字符对应的数字
    input_vocab=lambda : torch.tensor([out[-1],],device=device).reshape(1,1)
    for y in give_vocab[1:]:
        _,state=net(input_vocab(),state)
        out.append(vocab[y])
    for _ in range(num_need):
        y,state=net(input_vocab(),state)
        out.append(torch.argmax(y,dim=1))
    return ''.join(vocab.idx_to_token[i] for i in out)


# 创建整个网络.
net=RnnModel(rnn_layer,len(vocab),num_hiddens)
net.to(device)

#进行梯度的剪裁
def grad_clipping(net,params,theta):
    if isinstance(net,nn.Module):
        params=[param for param in params if param.requires_grad]
    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

# 进行单次的迭代
def epoch_train(train_iter,net,loss,updater,is_random=False,device=device):
    state=None
    metric=add_machine(2)
    for X,y in train_iter:
        if state==None or is_random==True:
            state=net.begin_state(batch_size=X.shape[0],device=device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
        X,y=X.to(device),y.to(device)
        y_hat,state=net(X,state)
        y=y.T.reshape(-1)
        l=loss(y_hat,y).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,net.parameters(),1)
            updater.step()
        metric.add(l*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1])



def train(train_iter,net,lr,num_epochs,is_random,device):
    loss=nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    updater=torch.optim.SGD(net.parameters(),lr=lr)
    for epoch in range(num_epochs):
        ppl=epoch_train(train_iter,net,loss,updater,is_random,device=device)
        if (epoch+1)%10==0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度为:{ppl:1.3f}')
        print(predict('time traveller',50,net,vocab,device))
        print(predict('traveller', 50, net, vocab, device))


train(train_iter,net,lr=0.8,num_epochs=500,is_random=False,device=device)