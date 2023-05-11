import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import math

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

#首先我们要获得整个数据集
device=try_gpu()
num_hiddens=512
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps) # 此时的train_iter.shape=[batch_size,num_steps]

def get_param(num_vocab,num_hidden,device):  # 参数的初始化
    num_input=num_output=num_vocab
    normal=lambda shape:torch.randn(shape,device=device)*0.01
    W_hh=normal((num_hidden,num_hidden))
    W_xh=normal((num_input,num_hidden))
    W_ho=normal((num_hidden,num_output))
    b_h=torch.zeros((num_hidden,),device=device)
    b_o=torch.zeros((num_output,),device=device)
    params=[W_hh,W_xh,W_ho,b_h,b_o]
    for param in params:
        param.requires_grad_(True)
    return params

def init_state(batch_size,num_hiddens,device): # 状态的初始化
    return (torch.zeros([batch_size,num_hiddens],device=device),)

def forward_fn(inputs,state,params):  # state即使init_state中传过来的元组，inputs应当为shape=[时间序列，batch_size,vocab_size]的三维矩阵
    W_hh,W_xh,W_ho,b_h,b_o=params
    H,=state
    out=[]
    for x in inputs:
        H=torch.tanh(torch.matmul(H,W_hh)+torch.matmul(x,W_xh)+b_h)
        y_hat=torch.matmul(H,W_ho)+b_o
        out.append(y_hat)
    return torch.cat(out,dim=0),(H,)  # 排列顺序为先第一个序列的所有batch之后再跟其他序列

class RNNModelScratch:
    def __init__(self,num_vocab,num_hidden,device,get_param,init_state,forward_fn):
        self.num_hidden=num_hidden
        self.num_vocab=num_vocab
        self.get_param=get_param(num_vocab,num_hidden,device)
        self.init_state=init_state
        self.forward_fn=forward_fn

    def __call__(self, X, state):  # 此时的X为一个批量中的X，shape=[batch,时间步长]
        X=F.one_hot(X.T,self.num_vocab).type(torch.float32)  # 此时变为了[时间步长,batch,num_vocab]
        return self.forward_fn(X,state,self.get_param)

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hidden,device)



def grad_clipping(net,theta):  # 梯度的裁剪
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=net.get_param
    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

# 进行预测（由前一个预测后一个的值）
def predict_ch8(prefix, num_preds, net, vocab, device): # 相当于进行一个前向的传播
    state=net.begin_state(batch_size=1,device=device)
    output=[vocab[prefix[0]]]  # output 存放对应的序列
    predict=lambda : torch.tensor([output[-1],],device=device).reshape((1,1)) # 代表利用前一个来预测下一个
    for y in prefix[1:]:
        _,state=net(predict(),state)
        output.append(vocab[y])
    for y in range(num_preds):
        y,state=net(predict(),state)
        output.append(torch.argmax(y,dim=1).reshape(1))
    return ''.join(vocab.idx_to_token[i] for i in output)

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def sgd(lr,params,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()


# 进行模型的训练（单词迭代的训练）
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    state = None
    metric = d2l.Accumulator(2)
    for X, Y in train_iter:
        if state == None or use_random_iter == True:  # 一次epoch迭代开始需要将state初始化，每次batch迭代的开始同时use_random_iter==True时也要初始化
            state = net.begin_state(X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                state.detach_()
            else:
                for s in state:
                    s.detach_()
        X, Y = X.to(device), Y.to(device)
        y_hat,state = net(X, state)
        y = Y.T.reshape(-1)
        l = loss(y_hat, y).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)  # 此时loss已经做过平均了
        metric.add(l * Y.numel(), Y.numel())
    return math.exp(metric[0] / metric[1])


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    loss=nn.CrossEntropyLoss()
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    else:
        updater=lambda batch_size:sgd(lr,net.get_param,batch_size)
    for epoch in range(num_epochs):
        ppl=train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch+1) %10==0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}')
        print(predict_ch8('time traveller', 50, net, vocab, device))
        print(predict_ch8('traveller', 50, net, vocab, device))



net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_param,init_state,forward_fn)
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())



