# LSTM相对于GRU来说其每一个序列的输出变为了当前的观测状态，上一个序列传入的记忆单元c和状态单元state，其中c和state的形状大小是一样的，都是[batch_Size,num_hiddens]
import math
import torch
from d2l import torch as d2l
from torch.nn import functional as F
from torch import nn

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device=try_gpu()
batch_size,batch_step=32,35
train_iter,vocab=d2l.load_data_time_machine(batch_size,batch_step)
num_hiddens=512

def get_param(vocab_size,num_hiddens,device):
    input_size=output_size=vocab_size
    def normal(shape):
        return torch.randn(shape,device=device)*0.01
    def three():
        return normal((input_size,num_hiddens)),normal((num_hiddens,num_hiddens)),torch.zeros([num_hiddens,],device=device)
    W_xi, W_hi, bi = three()  # 输入门
    W_xf, W_hf, bf = three()  # 遗忘门
    W_xo, W_ho, bo = three()  # 输出门
    W_xc, W_hc, bc = three()  # 候选记忆单元
    W_hy=normal((num_hiddens,output_size))
    by=torch.zeros([output_size,],device=device)
    params=[W_xi, W_hi, bi, W_xf, W_hf, bf, W_xo, W_ho, bo, W_xc, W_hc, bc, W_hy, by]
    for param in params:
        param.requires_grad_(True)
    return params

def init_state(batch_size,num_hiddens,device):
    return (torch.zeros([batch_size,num_hiddens],device=device),
            torch.zeros([batch_size,num_hiddens],device=device))

def LSTM_forward(inputs,state,params):  # 此时的X为[seq,batch_size,vocab_Size]
    W_xi, W_hi, bi, W_xf, W_hf, bf, W_xo, W_ho, bo, W_xc, W_hc, bc, W_hy, by = params
    H,C=state
    out=[]
    for x in inputs:  # 此时x.shape=[batch_size,vocab_size]
        I = torch.sigmoid(H @ W_hi + x @ W_xi + bi)
        O = torch.sigmoid(H @ W_ho + x @ W_xo + bo)
        F = torch.sigmoid(H @ W_hf + x @ W_xf + bf)
        C_prepare = torch.tanh(H @ W_hc + x @ W_xc + bc)
        C = F * C + I * C_prepare
        H = O * torch.tanh(C) # 此处使用tanh的原因F,I的取值范围(0,1),C_prepare的取值范围(-1,1),初始C为0，经过每一次迭代C会逐渐变大（因此tanh相当于对其进行归一化操作）
        Y = H @ W_hy + by
        out.append(Y)
    return torch.cat(out,dim=0),(H,C)


class My_LSTM():
    def __init__(self,vocab_size,num_hiddens,get_param,device,init_state,LSTM_forward):  # 此时无batch_size原因为训练和预测时batch不相同
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.get_param=get_param(self.vocab_size,self.num_hiddens,device)
        self.init_state=init_state
        self.forward_fn=LSTM_forward
    def __call__(self,X,state):
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32)
        Y,state=self.forward_fn(X,state,self.get_param)
        return Y,state
    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

net=My_LSTM(len(vocab),num_hiddens,get_param,device,init_state,LSTM_forward)

'''
def predict(give_vocab,num_need,net,vocab,device):
    state=net.begin_state(batch_size=1,device=device)
    out=[vocab[give_vocab[0]]]

    input_data=lambda :torch.tensor([out[-1],],device=device).reshape(1,1)

    for y in give_vocab[1:]:
        _,state=net(input_data(),state)
        out.append(vocab[y])
    for _ in range(num_need):
        y,state=net(input_data(),state)
        out.append(torch.argmax(y,dim=1).reshape(1))
    return ''.join(vocab.idx_to_token[i] for i in out)
'''
def predict(give_char,num,net,vocab,device): # 一个字符一个字符的输入，
    state = net.begin_state(batch_size=1,device=device)
    output = [vocab[give_char[0]]]  # 输入的第一个字符的序列
    input_content = lambda : torch.tensor(output[-1],device=device).reshape(1,1)
    for i in give_char[1:]:
        _,state = net(input_content(),state) # Y=[seq,batch,vocab_size]
        output.append(vocab[i])
    for _ in range(num):
        Y, state = net(input_content(), state)
        out = torch.argmax(Y,dim=-1).reshape(1).item()
        output.append(out)
    return ''.join(vocab.to_tokens(output))

def grad_clipping(net,theta):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=net.get_param
    norm=torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm

def epoch_train(train_iter,net,loss,updater,is_random):
    state=None
    metric=add_michine(2)
    for X,y in train_iter:
        if state == None or is_random == True:
            state=net.begin_state(X.shape[0],device=device)
        else:
            if isinstance(net,nn.Module) and not isinstance(state,tuple): # 此时为针对利用模板写的GRU和RNN
                state.detach_()
            else:
                for s in state:  # 争对不利用模板写的和利用模板写的LSTM，LSTM的state为含有两个元素的元组
                    s.detach_()
        X,y=X.to(device),y.to(device)
        y_hat,state=net(X,state)  # 此时输出为[seq*batch_size,vocab_size]
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

class add_michine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def sgd(batch_size,lr,params):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

def train(train_iter,net,lr,num_epochs,is_random=False):
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    loss=nn.CrossEntropyLoss()
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    else:
        updater=lambda batch_size:sgd(batch_size,lr,net.get_param)
    for epoch in range(num_epochs):
        ppl=epoch_train(train_iter,net,loss,updater,is_random)
        if (1+epoch)%10 == 0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}')
        print(predict('time traveller', 50, net, vocab, device))
        print(predict('traveller', 50, net, vocab, device))

train(train_iter,net,lr=1,num_epochs=500,is_random=False)