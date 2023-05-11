# GRU相对于rnn模型来说可以记住更多的信息（通过更新门和遗忘门进行筛选），且在遗忘门R的输出为1，更新门Z的输出为0时的GPU等价于RNN的网络模型

from d2l import torch as d2l
import torch
from torch.nn import functional as F
import math
from torch import nn

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device=try_gpu()
batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(batch_size,num_steps)
num_hiddens=256

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
        R=torch.sigmoid(X @ W_xr + H @ W_hr + br) # R为遗忘门，以往之前的信息
        Z=torch.sigmoid(X @ W_xz + H @ W_hz + bz) # Z为更新门，根据Z将当前状态更新入H中
        H_new=torch.tanh((H*R) @ W_hh + X @ W_xh + bh)  # 这一步R（R为0，1之间）的作用是尽量去关注当前的X中的信息，原因(H*R)<H,因此使得(H*R) @ W_hh该项权重变小
        H=(1-Z)*H_new+Z*H
        Y=H @ W_ho + bo
        out.append(Y)
    return torch.cat(out,dim=0),(H,)

class My_GRU():
    def __init__(self,vocab_size,num_hiddens,device,get_param,init_state,GRU_forward):
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.get_param=get_param(self.vocab_size,self.num_hiddens,device)
        self.init_state=init_state
        self.forward_fn=GRU_forward

    def __call__(self,X,state):  # 此时的X.shape=[batch_size,seq]
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32)
        Y,state=self.forward_fn(X,state,self.get_param)
        return Y,state

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

def grad_clipping(net,theta):  # 对
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=net.get_param

    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))

    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm


def predict(give_vocab,num_need,net,vocab,device):  # 对
    state=net.begin_state(batch_size=1,device=device)
    out=[vocab[give_vocab[0]]]
    give_input=lambda :torch.tensor([out[-1],],device=device).reshape(1,1)
    for y in give_vocab[1:]:
        _,state=net(give_input(),state)
        out.append(vocab[y])
    for _ in range(num_need):
        y,state=net(give_input(),state)
        out.append(torch.argmax(y,dim=1).reshape(-1))
    return ''.join(vocab.idx_to_token[i] for i in out)


class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def epoch_train(train_iter,net,loss,updater,is_random,device):
    state=None
    metric=add_machine(2)
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


def sgd(lr,batch_size,params):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

def train(train_iter,num_epochs,net,lr,device,is_random):
    loss=nn.CrossEntropyLoss()
    updater=lambda batch_size:sgd(lr=lr,batch_size=batch_size,params=net.get_param)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        ppl=epoch_train(train_iter, net, loss, updater, is_random,device)
        if (1+epoch)%10==0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度:{ppl:1.4f}')
        print(predict('the traveller',50,net,vocab,device))
        print(predict('traveller', 50, net, vocab, device))


net=My_GRU(len(vocab),num_hiddens,device,get_param,init_state,GRU_forward)  # 此时net存在于cuda上
train(train_iter, 500, net, 1, device, is_random=False)

