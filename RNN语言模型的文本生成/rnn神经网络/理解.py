# rnn的网络时mlp的一种改进，当前时刻的隐藏层的输出和当前时刻的输入X有关，同时和上一个隐藏层的输出有关（若当前时刻的隐藏层输出和上一时刻的隐藏层输出无关相当于一个mlp）
# 此时网络的需要训练的网络参数有5个分别是W_xh,W_hh,W_ho,b_h,b_o，同时需要对最开始的网络隐藏层的输出赋初值
# 语言模型相当于一个分类问题，只是存在多个时序，对于单个时序来说（已知字典的长度为m），相当于一个m的分类问题（从字典中选取一个概率最大的标签）
# 因此对于模型的好坏的判断会用交叉熵进行判定（softmax），此时的loss=每一个时间序列的交叉熵的平均值
# 由于历史的原因利用exp（loss）判定模型的好坏程度，称为困惑度，最理想的状态是困惑度为1表示当前序列只存在一种情况，困惑度为3表示当前序列存在3种情况
# rnn无法进行长序列的处理的原因是:当你的序列长度很长时，此时你的隐藏层的num_hidden无法记住所有的序列信息，但是如果增加隐藏层神经元个数会导致模型的过拟合（和mlp相类似）

from d2l import torch as d2l
import torch
from torch.nn import functional as F
import math
from torch import nn

# 之前的导入数据集
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps) # 此时的train_iter.shape=[batch_size,num_steps]

'''
#将标签进行独热化(练习)
a=F.one_hot(torch.tensor([0,2]),len(vocab)) # 此时返回了一个[len([0,2]),len(vocab)]的矩阵
print(a)
b=torch.arange(10).reshape(2,5) # 此时的2代表的是batch_size,5代表的是时间步长
b=F.one_hot(b.T,len(vocab)) # 此时最高维为时间步长（序列长度），对于二维矩阵进行独热化会变为三维矩阵
print(b.shape)
'''


def get_param(num_vocab,num_hidden,device): # 初始化网络中的参数
    num_input=num_output=num_vocab
    def normal(shape):
        return torch.randn(size=shape,device=device)*0.01
    W_xh=normal((num_input,num_hidden))
    W_hh=normal((num_hidden,num_hidden))
    W_ho=normal((num_hidden,num_output))
    b_h=torch.zeros(num_hidden,device=device)
    b_o=torch.zeros(num_output,device=device)
    params=[W_xh,W_hh,W_ho,b_h,b_o]
    for param in params:
        param.requires_grad_(True)
    return params


def init_state(batch_size,num_hiddens,device): # 初始化隐藏状态,放在了tuple中
    return (torch.zeros([batch_size,num_hiddens],device=device),)



def forward_fn(inputs,state,params):  #此时inputs的输出为[时间序列长度,batch_size,vocab_size]
    W_xh,W_hh,W_ho,b_h,b_o=params
    H,=state
    outputs=[]
    for X in inputs:
        H=torch.tanh(torch.matmul(X,W_xh)+torch.matmul(H,W_hh)+b_h) # H进行了更新
        Y=torch.matmul(H,W_ho)+b_o
        outputs.append(Y)
    return torch.cat(outputs,dim=0),(H,)  # 此时的输出形状为[每一个时间序列的batch_size*序列的长度,vocab_size]


class RNNModelScratch:
    def __init__(self,vocab_size,num_hiddens,device,get_param,init_state,forward_fn):
        self.vocab_size=vocab_size
        self.num_hiddens=num_hiddens
        self.get_param=get_param(self.vocab_size,self.num_hiddens,device)
        self.init_state=init_state
        self.forward_fn=forward_fn

    def __call__(self,X,state): # 此时的X为当前批量train_iter，__call__为创建一个类对象后，输入X,state会直接调用这个函数
        X=F.one_hot(X.T,self.vocab_size).type(torch.float32) # 独热化后为整型
        return self.forward_fn(X,state,self.get_param)

    def begin_state(self,batch_size,device):
        return self.init_state(batch_size,self.num_hiddens,device)


# 该函数是用来进行预测的prefix为给定的单词，num_preds代表着需要预测之后的多少个词(相当于一个前向传播，需要定义states)
def predict_ch8(prefix, num_preds, net, vocab, device):
    states=net.begin_state(batch_size=1,device=device)  # 最开始的初始H
    outputs=[vocab[prefix[0]]]  # 将单词转换为了对应的数字
    get_input=lambda : torch.tensor([outputs[-1],],device=device).reshape((1,1))
    for y in prefix[1:]:  # 此时对于prefix来更新states的状态（此时给定了2个标准词预测之后出现的单词，则states进行了2次的更新）
        _,states=net(get_input(),states)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 正式的开始预测
        y, states = net(get_input(), states) # y.shape=[1,vocab_size]
        outputs.append(torch.argmax(y,dim=1).reshape(1))
    return ''.join(vocab.idx_to_token[i] for i in outputs)


def grad_clipping(net,theta):  # 梯度的裁剪
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=net.get_param
    norm=torch.sqrt(sum(torch.sum((param.grad**2)) for param in params))
    if norm>theta:
        for param in params:
            param.grad[:]*=theta/norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):  # 对于一次的迭代
    state=None
    metric = d2l.Accumulator(2)
    for X,Y in train_iter:  # Y.shape=[batch_szie,序列长度]
        if state==None or use_random_iter: # use_random_iter说明两个batch之间不相关，因此需要重新对state进行赋值
            state=net.begin_state(X.shape[0],device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):# 此时的两个batch之间是相关的因此state不会清零
                # detach_的作用为将state这个量从中间节点变为了叶子节点，但是不会改变state的值（反向传播中的中间节点的梯度是不保存的，只保存叶子节点上同时require=True的梯度）
                # 叶子节点的定义为require_grad=True同时是用户直接创造的或者require_grad=False ；中间节点为require_grad=True，但是是通过运算操作的得到的
                # detach_()相当于把中间节点变成了require_grad=False的叶子节点
                # 此时的state是上个batch计算出来的，如果没有detach_, 此时的反向传播会反向传播到第一个batch的初始定义的后一个state的params上面（因此相当于清除之前batch的计算图）
                # 在之前的卷积和mlp或者是rnn系列中的use_random_iter=True中每个batch之间是独立的，但是use_random_iter=False中每个batch之间相互联系（前面的batch的最后的state等于后一个初始的batch的state，因此不同batch的计算图是连接在一起的）
                state.detach_() # 此时的detach_是清除之前batch的计算图的
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        X,Y=X.to(device),Y.to(device)
        y=Y.T.reshape(-1) # [batch_szie*序列长度] ,此时的排列顺序为一个时间序列中的每个batch的标签和y_hat对应
        y_hat,state=net(X,state)
        l=loss(y_hat,y).mean()
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            updater(batch_size=1)  # 之前已经平均过了
        metric.add(l * y.numel(), y.numel())
    return math.exp(metric[0]/metric[1])



def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss=nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    if isinstance(net,nn.Module):
        updater=torch.optim.SGD(net.parameters(),lr=lr)
    else:
        updater=lambda batch_size:sgd(net.get_param,lr,batch_size)
    for epoch in range(num_epochs):
        ppl=train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch+1) % 10==0:
            animator.add(epoch + 1, [ppl])
        print(f'困惑度 {ppl:.1f}')
        print(predict_ch8('time traveller', 50, net, vocab, device))
        print(predict_ch8('traveller', 50, net, vocab, device))


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_param,init_state,forward_fn)
num_epochs, lr = 500, 1
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())


