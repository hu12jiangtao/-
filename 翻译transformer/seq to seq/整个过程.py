from torch import nn
import torch
from d2l import torch as d2l

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        raise NotImplementedError

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def init_state(self,src_outputs):
        raise NotImplementedError
    def forward(self,X,state):
        raise NotImplementedError

class seq2seqEncoder(Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0): # embed_size是用来做映射的
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.GRU=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    def forward(self,X):  # 输入的X为[batch_Size,seq],embedding只在后面添加一个维度变为[batch_size,seq,embed_Size]
        X=self.embedding(X).permute(1,0,2)
        Y,state=self.GRU(X)
        return Y,state

class seq2seqDecoder(Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.GRU=nn.GRU(num_hiddens+embed_size,num_hiddens,num_layers,dropout=dropout)
        self.linear=nn.Linear(num_hiddens,vocab_size)
    def init_state(self,src_outputs):
        return src_outputs[1]
    def forward(self,X,state):
        X=self.embedding(X).permute(1,0,2)
        context=state[-1].repeat(X.shape[0],1,1) # [seq,batch_size,num_hiddens]
        X_to_context=torch.cat((X,context),dim=2)
        Y,state=self.GRU(X_to_context,state)  # 此时的Y=[seq,batch_size,num_hiddens]
        Y=self.linear(Y).permute(1,0,2)  # [seq,batch_size,vocab_size]
        return Y,state

class EncodeDecoder(nn.Module):
    def __init__(self,encode,decode):
        super().__init__()
        self.encode=encode
        self.decode=decode
    def forward(self,scs_X,tgt_X):
        src_outputs=self.encode(scs_X)
        tgt_state=self.decode.init_state(src_outputs)
        return self.decode(tgt_X,tgt_state)

embed_size, num_hiddens, num_layers, drop_out = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
# 生成数据集
train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
# 创建整一个网络
encode=seq2seqEncoder(len(src_vocab),embed_size,num_hiddens,num_layers,dropout=drop_out)
decode=seq2seqDecoder(len(tgt_vocab),embed_size,num_hiddens,num_layers,dropout=drop_out)
net=EncodeDecoder(encode,decode)
#print(net)


# 创造损失函数（之作用在译码器上，同时设置一个权重，<pad>对应为0）
def loss_weight(lines,valid_num,fill_value=0):  # 此时的lines为一个和label对应的全1矩阵，valid_num为除去<pad>的数量
    num_steps=lines.shape[1]
    mask=torch.arange(num_steps,device=device)
    weight=(mask[None,:]<valid_num[:,None])
    lines[~weight]=fill_value
    return lines


class MaskLoss(nn.CrossEntropyLoss): # 利用loss_weight函数给出一个权重
    def forward(self,y_pred,labels,valid_num): # y_pred=[batch_size,seq,vocab_size]
        weight=torch.ones_like(labels)  # 此时的weight=[batch_size,seq]
        weight=loss_weight(weight, valid_num)
        self.reduction='none'
        no_weight_loss=super().forward(y_pred.permute(0,2,1),labels)
        weight_loss=(no_weight_loss*weight).mean(1)
        return weight_loss

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def grad_clipping(net,limit):
    if isinstance(net,nn.Module):
        params=[param for param in net.parameters() if param.requires_grad]
    else:
        params=[param for param in net.get_params]
    norm=sum(torch.sum(param.grad**2) for param in params)
    if norm>limit:
        for param in params:
            param.grad*=limit/norm

def seq2seqpredict(data_iter,net,lr,tgt_vocab,num_epchos,device):
    net.to(device)
    net.train()
    def init_param(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names:
                if 'weight' in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    net.apply(init_param)
    loss=MaskLoss()
    trainer=torch.optim.Adam(net.parameters(),lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epchos):
        metric=add_machine(2)
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len=[i.to(device) for i in batch]
            trainer.zero_grad()
            add=torch.tensor([tgt_vocab['<bos>']]*Y.shape[0],device=device).reshape(-1,1)
            last_Y=torch.cat((add,Y[:,:-1]),dim=1)
            y_hat,_=net(X,last_Y)  # [batch_size,seq,vocab_size]
            l=loss(y_hat,Y,Y_valid_len)
            l.sum().backward()
            grad_clipping(net,1)
            trainer.step()
            valid_sum=sum(Y_valid_len)
            metric.add(l.sum(),valid_sum)
        if (epoch+1)%10==0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}')

seq2seqpredict(train_iter,net,lr,tgt_vocab,num_epochs,device)


def padding_seq(line,num_steps,pad_content='<pad>'):  # 此时的line应当为对应的数字列表
    if len(line)>num_steps:
        return line[:num_steps]
    else:
        return line+[src_vocab[pad_content]]*(num_steps-len(line))

def predict(give_sentence,src_vocab,tgt_vocab,net,num_steps):
    net.eval()
    index_s=src_vocab[give_sentence.lower().split(' ')]+[src_vocab['<eos>']] # 将需要被翻译的词转换为数字列表
    index_s=d2l.truncate_pad(index_s, num_steps, src_vocab['<pad>'])  # 将其补充成编码器的输入的seq
    index_s=torch.tensor(index_s,dtype=torch.long,device=device).reshape(1,-1)
    srs_outputs=net.encode(index_s)
    out=[]
    tgt_input=torch.tensor([tgt_vocab['<bos>']],dtype=torch.long,device=device).reshape(1,-1) # 解码器的输入
    tgt_state=net.decode.init_state(srs_outputs)
    for i in range(num_epochs):
        Y,tgt_state=net.decode(tgt_input,tgt_state)  # 此时的输出Y=[batch_size,seq,vocab_size]
        tgt_input=torch.argmax(Y,dim=2) # [1,1]
        predict_one=tgt_input.reshape(-1).item()
        if predict_one == tgt_vocab['<eos>']:
            break
        out.append(predict_one)
    return ' '.join(tgt_vocab.to_tokens(i) for i in out)



engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for i,j in zip(engs,fras):
    print(predict(i,src_vocab,tgt_vocab,net,num_steps))
    print(j)
    print('*'*50)