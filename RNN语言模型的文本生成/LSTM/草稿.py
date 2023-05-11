# 和GRU相类似，需要自己加最后的输出层，state为
import math
import torch
from d2l import torch as d2l
from torch import nn
from torch.nn import functional as F

def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class My_LSTM(nn.Module):
    def __init__(self,vocab_size,num_hiddens,num_layers,dropout=0):
        super(My_LSTM, self).__init__()
        self.layer=nn.LSTM(vocab_size,num_hiddens,num_layers=num_layers,dropout=dropout)
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens
        self.vocab_size=vocab_size
        self.linear=nn.Linear(num_hiddens,vocab_size)
    def init_state(self,batch,device):
        return (torch.zeros([self.num_layers,batch,self.num_hiddens],device=device),
                torch.zeros([self.num_layers,batch,self.num_hiddens],device=device))
    def forward(self,X,state):
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32)
        out,state = self.layer(X,state)
        out = self.linear(out)
        return out,state

def predict(give_char,num,net,vocab,device):
    state=net.init_state(batch=1,device=device)
    output = [vocab[give_char[0]]]
    input_char =lambda :torch.tensor(output[-1],device=device).reshape(1,1)
    for i in give_char[1:]:
        _,state = net(input_char(),state)
        output.append(vocab[i])
    for i in range(num):
        y,state = net(input_char(),state)
        out = torch.argmax(y,dim=-1).reshape(-1).item()
        output.append(out)
    return ''.join(vocab.to_tokens(output))

def train(data_iter,lr,num_epochs,is_random,device):
    net.to(device)
    loss = nn.CrossEntropyLoss(reduction='mean')
    trainer=torch.optim.SGD(net.parameters(),lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        ppl=epoch_train(data_iter, loss, trainer, net, is_random)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}')
        print(predict('time traveller', 50, net, vocab, device))
        print(predict('traveller', 50, net, vocab, device))


class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def grad_clipping(net,limit):
    params = [param for param in net.parameters() if param.requires_grad]
    norm = torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm > limit:
        for param in params:
            param.grad[:]*=limit/norm

def epoch_train(data_iter,loss,trainer,net,is_random):
    state = None
    net.train()
    metric = add_machine(2)
    for X,y in data_iter:
        X,y = X.to(device),y.to(device)
        if state == None or is_random == True:
            state = net.init_state(X.shape[0],device)
        else:
            if isinstance(net,nn.Module) and isinstance(state,tuple):
                for s in state:
                    s.detach_()
        y_hat,state = net(X,state) # y_hat=[seq,batch,vocab_size],y=[batch,seq]
        y_hat=y_hat.permute(1,2,0)
        l=loss(y_hat,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
        metric.add(l*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1])

batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(32,35)  # 此时添加了上面自己的创建的数据集，下面代码已进行过验证，是正确的
num_hiddens=256
device = try_gpu()
net = My_LSTM(vocab_size=len(vocab),num_hiddens=num_hiddens,num_layers=1)
train(data_iter=train_iter,lr=1,num_epochs=500,is_random=False,device=device)