import torch
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F
import math

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)

def get_param(vocab_size,num_hiddens,device):
    output_size=input_size=vocab_size
    normal=lambda shape:torch.randn(shape,device=device)*0.01
    def three():
        return (normal((input_size,num_hiddens)),
                normal((num_hiddens,num_hiddens)),
                torch.zeros([num_hiddens,],device=device))
    W_xz,W_hz,bz=three()
    W_xr,W_hr,br=three()
    W_xh,W_hh,bh=three()
    W_ho=normal((num_hiddens,output_size))
    bo=torch.zeros([output_size,],device=device)
    params=[W_xz,W_hz,bz,W_xr,W_hr,br,W_xh,W_hh,bh,W_ho,bo]
    for param in params:
        param.requires_grad_(True)
    return params


def init_state(batch_size,num_hiddens,device):
    return (torch.zeros([batch_size,num_hiddens],device=device),)

def GRU_forward(input,state,params):  # input.shape=[seq,batch_size,vocab_size]
    out=[]
    H,=state
    W_xz,W_hz,bz,W_xr,W_hr,br,W_xh,W_hh,bh,W_ho,bo=params
    for X in input:
        R=torch.sigmoid(X @ W_xr + H @ W_hr + br)
        Z=torch.sigmoid(X @ W_xz + H @ W_hz + bz)
        H_new=torch.tanh((H*R) @ W_hh + X @ W_xh + bh)
        H=(1-Z)*H_new+Z*H
        Y=H @ W_ho + bo
        out.append(Y)
    return torch.cat(out,dim=0),(H,)


class My_GRU():
    def __init__(self,vocab_size,num_hiddens,device,get_param,init_state,GRU_forward):
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.get_params=get_param(self.vocab_size,self.num_hiddens,device)
        self.init_state=init_state
        self.forward_fn=GRU_forward

    def __call__(self,X,state):  # 此时的X.shape=[batch_size,seq]
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32)
        Y,state=self.forward_fn(X,state,self.get_params)
        return Y,state

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)



vocab_size, num_hiddens, device = len(vocab), 256, d2l.try_gpu()
num_epochs, lr = 500, 1
net=My_GRU(len(vocab),num_hiddens,device,get_param,init_state,GRU_forward)

def grad_clipping(net,limit_value):  # 进行梯度裁剪
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=net.get_params

    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))

    if norm>limit_value:
        for param in params:
            param.grad[:]*=limit_value/norm


def predict(give_inform,num_predict,net,vocab,device):
    state=net.begin_state(batch_size=1,device=device) # 定义初始状态
    out=[vocab[give_inform[0]]]  # 存放第一个给定单词的序列
    input_choose=lambda : torch.tensor([out[-1],],device=device).reshape(1,1) # 利用前一个单词预测后一个单词
    for y in give_inform[1:]:
        _,state=net(input_choose(),state)
        out.append(vocab.token_to_idx[y])
    for _ in range(num_predict):
        y, state = net(input_choose(), state)  # 此时的输出为[batch_size*时间序列,vocab_size]
        out.append(torch.argmax(y,dim=1).reshape(1))
    return ''.join(vocab.idx_to_token[i] for i in out)

class add_machin():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def train_epochs(train_iter,is_random,net,device,loss,updater):
    state=None
    metric=add_machin(2)
    for X,y in train_iter:
        if state==None or is_random==True:
            state=net.begin_state(X.shape[0],device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        X,y=X.to(device),y.to(device)
        y_hat,state=net(X,state)
        y=y.T.reshape(-1)
        l=loss(y_hat,y).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net,1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net,1)
            updater(1)
        metric.add(l*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1])

def sgd(batch_size,lr,params):
    with torch.no_grad():
        for param in params:
            param-=param.grad*lr/batch_size
            param.grad.zero_()

def train(train_iter,is_random,net,lr,num_epochs):
    loss=nn.CrossEntropyLoss()
    updater=lambda batch_size:sgd(batch_size,lr,net.get_params)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        ppl=train_epochs(train_iter, is_random, net, device, loss, updater)
        if (epoch+1)%10==0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}')
        print(predict('time traveller', 50, net, vocab, device))
        print(predict('traveller', 50, net, vocab, device))



train(train_iter,False,net,lr=1,num_epochs=500)
