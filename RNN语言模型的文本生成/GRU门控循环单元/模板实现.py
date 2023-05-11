# GRU的模板和rnn是相同的，只是将nn.RNN换成了nn.GRU

from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import math
import re
import collections
import random

def load_file():
    path='D:\\python\\pytorch作业\\序列模型\\data\\timemachine.txt'
    with open(path,'r') as f:
        lines=f.readlines()  # 生成列表
    lines=[re.sub('[^A-Za-z]+',' ',line).lower().strip() for line in lines]
    return lines # 此时一行为一个列表元素

def create_token(lines,mode='word'):
    if mode == 'word':
        tokens=[line.split() for line in lines]
    elif mode == 'char':
        tokens=[list(line) for line in lines]
    return tokens

def collect_count(tokens):
    if isinstance(tokens[0],list):
        return collections.Counter(token for line in tokens for token in line)
    else:
        return collections.Counter(token for token in tokens)


class Vocab():
    def __init__(self,tokens,limit_num,collect_count):
        self.tokens=tokens
        self.sort_token=sorted(collect_count(self.tokens).items(),key=lambda X:X[1],reverse=True)
        self.save_vocab=['unk']
        self.unk=0
        self.save_vocab+=[name for name,time in self.sort_token if time>limit_num and name not in self.save_vocab]
        self.idx_to_token,self.token_to_idx=[],{}
        for i in self.save_vocab:
            self.idx_to_token.append(i)
            self.token_to_idx[i]=len(self.idx_to_token)-1

    def __getitem__(self, input_tokens): # 将字符转换为数字
        if not isinstance(input_tokens,(list,tuple)):
            return self.token_to_idx.get(input_tokens,self.unk)
        return [self.__getitem__(i) for i in input_tokens]

    def to_tokens(self,input_num):
        if not isinstance(input_num,(list,tuple)):
            return self.idx_to_token[input_num]
        return [self.idx_to_token[i] for i in input_num]

    def __len__(self):
        return len(self.idx_to_token)

def create_vocab(max_tokens=-1):
    lines=load_file()
    tokens=create_token(lines,mode='char')
    vocab=Vocab(tokens,0,collect_count)
    corpus=[vocab[token] for line in tokens for token in line]
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return vocab,corpus

# 形成批量的数据（定义两种批量的分类方法）

def seq_data_iter_random(corpus,batch_size,num_steps): # num_steps=seq
    corpus=corpus[random.randint(0,num_steps-1):]
    num_little=len(corpus)//num_steps
    start_index=list(range(0,num_little*num_steps,num_steps))
    random.shuffle(start_index)

    def data_output(start):
       return corpus[start:start+num_steps]

    num_batch=num_little//batch_size
    for i in range(0,num_batch*batch_size,batch_size):
        batch_start_index=start_index[i:i+batch_size]
        X=[data_output(i) for i in batch_start_index]
        Y=[data_output(i+1) for i in batch_start_index]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_seq(corpus,batch_size,num_steps):
    offset=random.randint(0,num_steps)
    num_use_data=(len(corpus)-offset-1)//batch_size*batch_size
    X_s=torch.tensor(corpus[offset:offset+num_use_data])
    Y_s = torch.tensor(corpus[offset+1:offset + num_use_data+1])  # 当(len(corpus)-offset-1)整除时正好取到corpus的最后一个元素
    X_s,Y_s=X_s.reshape(batch_size,-1),Y_s.reshape(batch_size,-1)
    num_batch=X_s.shape[1]//batch_size
    for i in range(0,num_batch*batch_size,batch_size):
        X=X_s[:,i:i+num_steps]
        Y=Y_s[:,i:i+num_steps]
        yield X,Y

# 进行一次总结

class create_data1():
    def __init__(self,batch_size,num_steps,max_tokens,is_random,seq_data_iter_random,seq_data_iter_seq):
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.vocab, self.corpus = create_vocab(max_tokens)
        if is_random == True:
            self.output=seq_data_iter_random
        else:
            self.output=seq_data_iter_seq

    def __iter__(self):
        out=self.output(self.corpus,self.batch_size,self.num_steps)
        return out

def load_data_time_machine(batch_size,num_steps,is_random=False,max_tokens=10000):
    train_iter=create_data1(batch_size,num_steps,max_tokens,is_random,seq_data_iter_random,seq_data_iter_seq)
    return train_iter,train_iter.vocab


batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(32,35)  # 此时添加了上面自己的创建的数据集，下面代码已进行过验证，是正确的

# 创建GRU网络
def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device=try_gpu()
num_hiddens=256
gru_layer=nn.GRU(len(vocab),num_hiddens)

def init_state(batch_size,num_hiddens,device):
    return torch.zeros([1,batch_size,num_hiddens],device=device)

class My_GRU(nn.Module):
    def __init__(self,vocab_size,num_hiddens,gru_layer,init_state):
        super().__init__()
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.gru_layer=gru_layer
        self.init_state=init_state
        self.linear=nn.Linear(self.num_hiddens,self.vocab_size)

    def forward(self,X,state):  # 此时的X为[batch_size,seq]
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32)
        Y,state=self.gru_layer(X,state)
        out=self.linear(Y.reshape(-1,Y.shape[-1]))
        return out,state

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)

net=My_GRU(len(vocab),num_hiddens,gru_layer,init_state)
net.to(device)


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
    if not isinstance(net,nn.Module):
        updater=lambda batch_size:sgd(lr=lr,batch_size=batch_size,params=net.get_param)
    else:
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        ppl=epoch_train(train_iter, net, loss, updater, is_random,device)
        if (1+epoch)%10==0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度:{ppl:1.4f}')
        print(predict('the traveller',50,net,vocab,device))
        print(predict('traveller', 50, net, vocab, device))

train(train_iter, 500, net, 1, device, is_random=False)
