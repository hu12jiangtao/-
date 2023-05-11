# 整个seq2seq的训练流程为：
# 1.train_iter, src_vocab, tgt_vocab中train_iter包含着X_array, X_valid_len, Y_array, Y_valid_len
# 2.在编码器阶段，X_array代表着编码器的输入，X_array中一行包含着 需翻译的单词+<EOS>(长度为X_valid_len)，和(seq-X_valid_len)个<pad>
# 3.在译码器阶段： Y_array作为用来计算损失的标签，而译码器的输入应当为：<bos>+翻译后的单词  加上1个<EOS> 加上个（seq-Y_valid_len-1）个<pad>（相当于删去最后一个<pad>）
#             这样子的原因是对于译码器的输入设置的形状为[batch_size,seq],如果不减1seq的长度对不上 ，同时删去最后一个pad对loss的计算并不产生影响
#             在编码器中根据前一个预测后一个值（编码器中<bos>于测翻译后的第一个值，通过标准预测翻译后的后一个值）
# 在预测过程中我们使用的是贪心搜索来预测序列（去之后概率最大的序列进行输出）

from d2l import torch as d2l
from torch import nn
import torch

# 创建编码器和解码器的模型
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        raise NotImplementedError # 如果该类的子类调用了这个函数，但是没有重新定义是会报错

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
    def init_state(self,enc_outputs):
        raise NotImplementedError
    def forward(self,X,state):
        raise NotImplementedError

# 将编码器和解码器整合在一起构建一个网络
class EncoderDecoder(nn.Module):
    def __init__(self,encode,decode):
        super().__init__()
        self.encode=encode
        self.decode=decode
    def forward(self,enc_X,tgt_X):
        enc_outputs=self.encode(enc_X)
        tgt_state=self.decode.init_state(enc_outputs)
        return self.decode(tgt_X,tgt_state)


class Seq2SeqEncoder(Encoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,drop_out=0):
        super().__init__()
        self.embeding=nn.Embedding(vocab_size,embed_size) # 第一个参数为需要映射的词的个数，第二个为每一个词映射成的词向量的长度
        # 输入的X为[batch_size,seq],且必须是一个长整型;输出为[batch_size,seq,embed_size]
        # one_hot是一种最简单的编码方式，一个单词的序列映射成一个独热向量
        self.rnn=nn.GRU(embed_size,num_hiddens,num_layers,dropout=drop_out) # GRU的第一个参数为一个单词对应的序列映射成的词向量的长度
    def forward(self, X, ):  # 此时输入的X为[batch_size,seq]，训练中编码部分的输入（被翻译的内容）
        X=self.embeding(X) # 此时的X相当于[batch_size,seq,embed_size]
        X=X.permute(1,0,2)  # [seq,batch_size,embed_size]
        Y,state=self.rnn(X) # 输入中没有state，自动将初始state赋值为0？
        # 且其为最后一个序列中的Y和state，此时的Y为最后一层的状态层输出，为[seq,batch_size,num_hiddens],state.shape=[num_layers,batch_size,num_hiddens]
        return Y,state # 主要传出state

class Seq2SeqDecoder(Decoder):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,drop_out=0):  # 编码部分的参数权重不同，vocab_size,embed_size解码和编码部分不一定相同？
        super().__init__()
        self.embeding=nn.Embedding(vocab_size,embed_size)
        self.rnn=nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers,dropout=drop_out)
        # 此时vocab_size+num_hiddens的原因是对target的每一个单词输入都加入了被翻译句子的所有信息
        self.dense=nn.Linear(num_hiddens,vocab_size)
    def init_state(self, enc_outputs):  # enc_outputs为编码部分的输出
        return enc_outputs[1]  # [num_layers,batch_size,num_hiddens],由此可得的编码和解码部分的num_layers,batch_size,num_hiddens应当相同
    def forward(self, X, state): # X=[batch_size,seq],type=long，训练中解码部分的输入
        X=self.embeding(X).permute(1,0,2)
        context=state[-1].repeat(X.shape[0],1,1)  # state[-1]代表着翻译内容的最后一个序列的最后一个状态层的状态，其包括了被翻译内容的所有信息
        # repeat(X.shape[0],1,1)是将state[-1](shape=[batch_size,num_hiddens])扩展为了[seq,batch_size,num_hiddens]
        X_and_context=torch.cat((X,context),dim=2) # 此时也要求了解码部分和编码部分的seq长度相同
        # 解码部分的每一个序列的输出和翻译内容的最后一个序列的最后一个状态层相结合，提高准确率，同时和self.rnn第一个参数相应
        Y,state=self.rnn(X_and_context,state) # Y.shape=(seq,batch_size,num_hiddens)
        Y=self.dense(Y).permute(1,0,2)  # 将batch_size放到维度的最前面,Y.shape=(batch_size,seq,vocab_size)
        return Y,state


'''
# 原先验证
encoder = Seq2SeqEncoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encoder.eval()
X = torch.zeros((4, 7), dtype=torch.long)
decoder = Seq2SeqDecoder(vocab_size=10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decoder.eval()
state = decoder.init_state(encoder(X))
output, state = decoder(X, state)
print(output.shape, state.shape)
'''


def sequence_mask(X, valid_len, value=0): # valid_len记录了每个batch中的每一个样本的实际seq长度为通过模型得到的预测内容，shape=[batch_size,规定的seq]
    maxlen=X.shape[1]
    # 在模型训练的过程中填充，在训练样本中为了凑成一个批量将所有行的seq固定为了一个值，若一行的seq长度不够利用'<pad>'进行填充，在计算损失的过程中这一部分应当删去
    mask=torch.arange((maxlen),dtype=torch.float32,device=X.device)
    # 列向量矩阵(m,1)可以和行向量矩阵(n,1)通过广播机制比较大小，得到(n,m)的bool矩阵
    # 两个不为行矩阵或者列矩阵的矩阵不能比较大小，一个不为行矩阵或者列矩阵的矩阵(m,n)可以通过广播和(m,1)的列矩阵,(1,n)的行矩阵进行大小的比较
    mask=(mask[None,:]<valid_len[:,None])  # valid_len[:,None]等价于valid_len.unsqueeze(0)在None所在的地方增加一个维度
    X[~mask]=value # ~符号代表取反
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,y_pred,labels,valid_len): # 输入的y_pred.shape(batch_size,seq,vocab_size)
        weight=torch.ones_like(labels)
        weight=sequence_mask(weight,valid_len) # 作为一个权重参数，如果标签中为<pad>对应的字符则权重设为0
        self.reduction='none'
        weight_loss=super().forward(y_pred.permute(0,2,1),labels) # 得到的结果为[batch_size,seq],三维矩阵和二维矩阵做交叉熵，三维矩阵需要做交叉向的放中间
        no_weight_loss=(weight_loss*weight).mean(1) # 返回的是每一个batch中每一行的总损失
        return no_weight_loss

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    def init_param(m):
        if type(m)==nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m)==nn.GRU:
            for param in m._flat_weights_names:  # _flat_weight_names的具体作用为将一层的所有参数的名称组合为一个列表(_flat_weights_names只针对于LSTM，GPU)
               if 'weight' in param:
                   nn.init.xavier_uniform_(m._parameters[param]) # 将一个特定的层的所有参数作为一个字典进行展开(_parameters对于Linear，Conv2d都适用)
    net.apply(init_param)
    net.to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr)
    loss=MaskedSoftmaxCELoss()
    net.train()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(2)
        for batch in data_iter:
            X_array, X_valid_len, Y_array, Y_valid_len=[i.to(device) for i in batch]
            optimizer.zero_grad()

            bos=torch.tensor([tgt_vocab['<bos>']] * Y_array.shape[0],device=device).reshape(-1,1)
            dec_input=torch.cat((bos,Y_array[:,:-1]),1) # 得到解码器的输入
            y_hat,_=net(X_array,dec_input)
            l=loss(y_hat,Y_array,Y_valid_len)
            l.sum().backward()
            d2l.grad_clipping(net, 1)
            optimizer.step()
            num_tokens = Y_valid_len.sum() # 获得一个batch中除去pad的单词的个数
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, (metric[0] / metric[1],))
        print(f'loss {metric[0] / metric[1]:.3f}')

embed_size, num_hiddens, num_layers, drop_out = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 10, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        drop_out)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        drop_out)
net=EncoderDecoder(encoder,decoder)  # net的输入为编码器和解码器的X输入，输出为解码器的Y_hat(shape=(seq,batch_size,vocab_size))
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)


# 进行预测
def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,device):
    net.eval()
    src_tokens=src_vocab[src_sentence.lower().split(' ')]+[src_vocab['<eos>']] # 生成了编码器的输入
    src_tokens = d2l.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])  # 将其长度扩展为seq的序列长度，此时的输出为一个列表
    src_tokens=torch.tensor(src_tokens,dtype=torch.long,device=device).reshape(1,-1) # 将其转换为适应编码器的输入[batch_size,seq]的形式
    enc_out=net.encode(src_tokens)  # 获得编码器的输出
    dec_state=net.decode.init_state(enc_out) # 得到译码器的初始输入状态
    dec_X=torch.tensor([tgt_vocab['<bos>']],dtype=torch.long,device=device).reshape(1,-1) # 得到译码器的初始输入[batch_size,seq]=[1,1]
    out_seq=[]
    for _ in range(num_steps):
        Y,dec_state=net.decode(dec_X,dec_state)  # 此时Y.shape=[batch_size,seq,vocab_size],输出Y=[batch_size,seq,vocab_size]=[1,1,vocab_size]
        dec_X=torch.argmax(Y,dim=2) # 得到之后的一个预测值shape=[1,1]
        pred=dec_X.reshape(-1).item()
        if pred == tgt_vocab['<eos>']:
            break
        out_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(i) for i in out_seq)

engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
for i,j in zip(engs,fras):
    print(predict_seq2seq(net, i, src_vocab, tgt_vocab, num_steps, device))
    print(j)
    print('*'*50)


