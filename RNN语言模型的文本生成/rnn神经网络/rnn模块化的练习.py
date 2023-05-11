import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(32,35)
num_hiddens=512

rnn_layer=nn.RNN(len(vocab),num_hiddens)


def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device=try_gpu()

class RnnModel(nn.Module):
    def __init__(self,rnn_layer,num_hiddens,vocab_size):
        super().__init__()
        self.rnn_layer=rnn_layer
        self.num_hiddens=num_hiddens
        self.vocab_size=vocab_size
        self.linear=nn.Linear(num_hiddens,vocab_size)

    def forward(self,X,state): # 此时的X为[batch_size,seq]
        X=F.one_hot(X.T.long(),self.vocab_size).type(torch.float32) #[seq,batch_size,vocab_size]
        Y,state=self.rnn_layer(X,state) # Y.shape=[seq,batch_size,num_hiddens]
        out=self.linear(Y.reshape(-1,Y.shape[-1]))
        return out,state

    def begin_state(self,batch_size,device):
        return torch.zeros([1,batch_size,self.num_hiddens],device=device)

net=RnnModel(rnn_layer,num_hiddens,len(vocab))
net.to(device)
X=torch.ones([batch_size,num_steps],device=device)
state=net.begin_state(batch_size,device=device)
Y,state=net(X,state)
print(Y.shape,state.shape)
