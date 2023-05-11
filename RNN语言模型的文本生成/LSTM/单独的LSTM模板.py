# 因为在LSTM中的训练参数为这些w，b，和batch，seq都是无关，同时在训练模型中每一个seq的LSTM模块的这些参数是相同的，因此在训练时为较长的seq训练，而在预测时可以利用seq=1的一个单次输入模型中训练
# 对于LSTM的公式来说
# 输入门:I_t = sigmoid(X_t * W_xi + H_t-1 * W_hi + B_i)
# 输出门:O_t = sigmoid(X_t * W_xo + H_t-1 * W_ho + B_o)
# 遗忘门:F_t = sigmoid(X_t * W_xf + H_t-1 * W_hf + B_f)
# 临时记忆单元C_hat = tanh(X_t * W_xc + H_t-1 * W_hc + B_c)
# 记忆单元C_t = F_t * C_t-1 + C_hat * I_t
# 输出状态H_t = O_t * tanh(C_t)
# LSTM能比RNN网络存储更长的时序记忆原因是:C_t的变化较为缓慢，因此在t-k时序存储的信息，在t时刻仍然能够进行保存
from torch import nn
import torch
from torch.nn import functional as F
from d2l import torch as d2l
import math


def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class My_LSTM(nn.Module):  # 相关参数的维度
    # state中的记忆单元和状态:[num_layers,batch,num_hiddens]，最后一个序列的所有的state
    # 输入X:[seq,batch,vocab_size] ,seq 和 batch 由输入决定
    # 输出Y:[seq,batch,vocab_size]
    # 输出给线性层的out:[seq,batch,num_hiddens]，为所有序列的最后一层state的组合
    def __init__(self,vocab_size,num_hiddens,num_layers,dropout=0):
        super(My_LSTM, self).__init__()
        self.layer = nn.LSTM(vocab_size,num_hiddens,num_layers=num_layers,dropout=dropout)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.linear = nn.Linear(num_hiddens,vocab_size)
        self.vocab_size=vocab_size

    def init_state(self,batch,device):
        return (torch.zeros([self.num_layers,batch,self.num_hiddens],device=device),
                torch.zeros([self.num_layers,batch,self.num_hiddens],device=device))

    def forward(self,X,state):
        # batch_first默认为False，代表输入的应当为[seq,batch,embed_size],若batch_first=True则输入的应当为[batch,seq,embed_size]
        X = F.one_hot(X.long(),self.vocab_size).permute(1,0,2).type(torch.float32)
        out,state=self.layer(X,state)  # state中的每一个元素 = [num_layers,batch,num_hiddens] , out=[seq,batch,num_hiddens]
        out = self.linear(out)  # [seq,batch,vocab_size]
        return out,state




def predict(give_char,num,net,vocab,device): # 一个字符一个字符的输入，
    # 因为在LSTM中的训练参数为这些w，b，和batch，seq都是无关，同时在训练模型中每一个seq的LSTM模块的这些参数是相同的，因此在训练时为较长的seq训练，而在预测时可以利用seq=1的一个单次输入模型中训练
    state = net.init_state(batch=1,device=device)
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

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def train(data_iter,lr,num_epochs,is_random,device):
    net.to(device)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    loss = nn.CrossEntropyLoss()
    # 此处可以进行选择(l=loss(y_hat,y)),当reduction=mean则求出整个数的平均数，reduction=sum则求出整个数的总数，reduction=None则求出每个batch的交叉熵，默认为mean
    trainer = torch.optim.SGD(net.parameters(),lr=lr)
    for epoch in range(num_epochs):
        ppl=epoch_train(data_iter,trainer,loss,is_random,device)
        if (epoch+1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}')
        print(predict('time traveller', 50, net, vocab, device))
        print(predict('traveller', 50, net, vocab, device))


def grad_clipping(net,limit_num):
    params = [param for param in net.parameters() if param.requires_grad]
    norm = torch.sqrt(sum(torch.sum(param.grad**2) for param in params))
    if norm > limit_num:
        for param in params:
            param.grad[:] *= limit_num/norm

def epoch_train(data_iter,trainer,loss,is_random,device):  # 对于每一次的的迭代后的state进行重新赋值 ,代表每一个batch的开始
    metric = add_machine(2)
    net.train()
    state = None
    for X,y in data_iter:
        X,y=X.to(device),y.to(device)
        if state == None or is_random == True:
            state = net.init_state(X.shape[0],device)
        else:
            if isinstance(net,nn.Module) and isinstance(state,tuple):
                for s in state:
                    s.detach_()
        y_hat,state = net(X,state) # y_hat = [seq,batch,vocab_size] ,y=[batch,seq]
        y_hat=y_hat.permute(1,2,0)
        l=loss(y_hat,y).mean()
        trainer.zero_grad()
        l.backward()
        grad_clipping(net,limit_num=1)
        trainer.step()
        metric.add(l*y.numel(),y.numel())
    return math.exp(metric[0]/metric[1])

batch_size,num_steps=32,35
train_iter,vocab=d2l.load_data_time_machine(32,35)  # 此时添加了上面自己的创建的数据集，下面代码已进行过验证，是正确的
num_hiddens=256
device = try_gpu()
net = My_LSTM(vocab_size=len(vocab),num_hiddens=num_hiddens,num_layers=1)
train(data_iter=train_iter,lr=1,num_epochs=500,is_random=False,device=device)