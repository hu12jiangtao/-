import os
from torch import nn
import torch
from d2l import torch as d2l
from torch.nn import functional as F
import math
import beam_search
# 此时在encode加入倒三角的编码结构的效果不是特别好(预测不出来内容)
# 在使用了label smoothing后在验证集上加入beam search 用来提高预测的准确率


class AdditiveAttention(nn.Module):
    def __init__(self,quiries_size,keys_size,num_hiddens,dropout):
        super(AdditiveAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.w_k = nn.Linear(keys_size,num_hiddens)
        self.w_q = nn.Linear(quiries_size,num_hiddens)
        self.w_v = nn.Linear(num_hiddens,1)
    def forward(self,quiries,keys,values):
        quiries = self.w_q(quiries) # [batch,num_q,quiry_size]
        keys = self.w_k(keys) # [batch,num_k,key_size]
        weight = quiries[:,:,None,:]+keys[:,None,:,:]
        weight = torch.tanh(weight) # [batch,num_q,num_k,key_size]
        weight = self.w_v(weight).squeeze(-1) # [batch,num_q,num_k],此时对num_k进行遮掩分配权重，因此需要enc_values_lens
        mask_weight = F.softmax(weight,dim=-1)
        return torch.bmm(self.dropout(mask_weight),values) # [batch,num_q,value_size]


class Encode(nn.Module):
    def __init__(self):
        super(Encode, self).__init__()
    def forward(self,X):
        raise NotImplementedError

class Seq2SeqEncoder(Encode): # 将encode修正为多层的金字塔型的序列模型,用来减少输出的seq的长度，有利于attention中更有效的抽取特征
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.GRU = nn.GRU(embed_size,num_hiddens,num_layers=num_layers,dropout=dropout)
    def forward(self,X): # X=[batch,seq]
        X = self.embed(X.T.long())
        y,state = self.GRU(X)
        return y,state

class Decode(nn.Module):
    def __init__(self):
        super(Decode, self).__init__()
    def init_state(self,enc_outputs):
        raise NotImplementedError
    def forward(self,X,state):
        raise NotImplementedError

class Seq2SeqDecode(Decode):
    def __init__(self,vocab_size,embed_size,num_hiddens,num_layers,dropout):
        super(Seq2SeqDecode, self).__init__()
        self.attention = AdditiveAttention(num_hiddens,num_hiddens,num_hiddens,dropout=dropout)
        self.GRU = nn.GRU(embed_size+num_hiddens,num_hiddens,num_layers=num_layers,dropout=dropout)
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.linear = nn.Linear(num_hiddens,vocab_size)

    def init_state(self,enc_outputs): # 有解码器得到的部分
        enc_Y,enc_out_state=enc_outputs  # enc_Y 作为keys，values , enc_out_state为给编码器的初始状态
        return enc_Y.permute(1,0,2),enc_out_state

    def forward(self,X,state):
        enc_Y,dec_input_state = state  # dec_input_state = [num_layers,batch_size,num_hiddens] = [2,1,hidden_size]
        X = self.embed(X).permute(1,0,2) # [seq,batch,embed_size]
        out = []
        for x in X:
            x = x.unsqueeze(1) # x=[1,batch,embed_size]
            context = self.attention(dec_input_state[-1].unsqueeze(1),enc_Y,enc_Y) # context=[batch,1,hiddens_size]
            context_x=torch.cat((context,x),dim=-1) # [batch,1,embed_size+num_hidden]
            Y,dec_input_state = self.GRU(context_x.permute(1,0,2),dec_input_state)
            out.append(Y)
        out =torch.cat(out,dim=0)  # out=[seq,batch,num_hiddens]
        out = self.linear(out) # out=[seq,batch,vocab_size]
        return F.log_softmax(out.permute(1,0,2),dim=-1),(enc_Y,dec_input_state) # [batch,seq,vocab_size]

class EncoderDecoder(nn.Module):
    def __init__(self,encode,decode):
        super(EncoderDecoder, self).__init__()
        self.encode = encode
        self.decode = decode
    def forward(self,enc_X,dec_X):
        enc_outputs=self.encode(enc_X)
        state=self.decode.init_state(enc_outputs)
        return self.decode(dec_X,state)

# 此时使用label smoothing的损失函数
def label_smoothing_loss(pred_y,true_y,label_smoothing=0.1): # 三维矩阵的label smoothing
    pred_y = F.log_softmax(pred_y,dim=-1)
    true_y = F.one_hot(true_y,pred_y.shape[-1])
    true_y[:,:,1] = torch.zeros(size=(true_y.shape[0],true_y.shape[1]),device=pred_y.device)
    class_dim = true_y.shape[-1]
    seq_lens = torch.sum(torch.sum(true_y,dim=-1),-1,keepdim=True) # [batch,]
    nll_loss = (1 - label_smoothing) * true_y * pred_y # [batch,max_label_len]
    smoothing_loss = label_smoothing / class_dim * pred_y * torch.sum(true_y,dim=-1,keepdim=True) # [batch,max_label_len]
    loss = nll_loss + smoothing_loss # [batch,max_label_len]
    loss = - torch.mean(torch.sum(loss,dim=-1) / seq_lens, dim=-1)
    return torch.sum(loss)

class add_machine():
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def grad_clipping(net, limit):
    params = [param for param in net.parameters() if param.requires_grad]
    norm = torch.sqrt(sum(torch.sum(param.grad ** 2) for param in params))
    if norm > limit:
        for param in params:
            param.grad[:] *= limit / norm


def train(data_iter, net, lr, tgt_vocab, num_epochs, device):
    def init_param(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for i in m._flat_weights_names:
                if 'weight' in i:
                    nn.init.xavier_uniform_(m._parameters[i])

    net.apply(init_param)
    net.to(device)
    trainer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        net.train()
        metric = add_machine(2)
        for batch in data_iter:
            src_X, src_valid_len, tgt_X, tgt_valid_len = [i.to(device) for i in batch]  # tgt_X=[batch,seq],tgt_valid_len=[batch,]
            enc_outputs = net.encode(src_X)
            state = net.decode.init_state(enc_outputs)  # 传递进入解码器的初始state
            dec_context = torch.tensor([tgt_vocab['<bos>']] * tgt_X.shape[0], dtype=torch.long, device=device).reshape(
                -1, 1)
            dec_input = torch.cat((dec_context, tgt_X[:, :-1]),dim=-1)
            y_hat, state = net.decode(dec_input, state) # 此时的y_hat=[batch,seq,vocab_size] ,tgt_X = [batch,seq]
            l = label_smoothing_loss(y_hat,tgt_X)
            trainer.zero_grad()
            l.sum().backward()
            grad_clipping(net, 1)
            trainer.step()
            num_tokens = sum(tgt_valid_len)
            metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 30 == 0:
            print(f'loss {metric[0] / metric[1]:.3f}')


def pad_content(lines,length,pad='<pad>'):
    l = len(lines)
    if l > length:
        return lines[:length]
    else:
        return lines + (length - l) * [pad]

def predict(input_sentence,net,num_steps,src_vocab, tgt_vocab,device):
    # 首先将句子转换为网络的输入的格式: 句子 + <eos> + <pad>
    input_sentence = input_sentence.split(' ') + ['<eos>']
    pad_sentence = pad_content(input_sentence,num_steps)
    pad_enc_index = src_vocab[pad_sentence]
    enc_input = torch.tensor(pad_enc_index,dtype=torch.long,device=device).reshape(1,-1) # [batch,seq] = [1,num_steps]
    enc_outputs=net.encode(enc_input)
    state=net.decode.init_state(enc_outputs)
    dec_input = torch.tensor(tgt_vocab['<bos>'],dtype=torch.long,device=device).reshape(1,1)
    out = []
    for i in range(num_steps):
        y,state=net.decode(dec_input,state) # y=[batch,seq,vocab_size] = [1,1,vocab_size]
        pred = torch.argmax(y,dim=-1)
        dec_input = pred
        output = pred.reshape(-1).item()
        if tgt_vocab['<eos>'] == output:
            break
        out.append(output)
    return ' '.join(tgt_vocab.to_tokens(out))

def blue(pre_str,label_str,k): # 其中的k代表的是k以下个词元的个数所相乘
    label_str ,pre_str = label_str.split(' ') ,pre_str.split(' ')
    label_len ,pre_len = len(label_str),len(pre_str)
    blue_value = math.exp(min(0,1-label_len/pre_len))
    for n in range(1,1+k):
        save_dict,count = {},0
        for i in range(label_len-n+1):
            if save_dict.get(' '.join(label_str[i:i+n])) is None:
                save_dict[' '.join(label_str[i:i+n])] = 0
            save_dict[' '.join(label_str[i:i+n])] += 1
        for i in range(pre_len-n+1):
            if save_dict.get(' '.join(pre_str[i:i+n]),0) > 0:
                save_dict[' '.join(pre_str[i:i+n])] -= 1
                count += 1
        blue_value *= pow(count/(pre_len-n+1),pow(0.5,n))
    return blue_value


if __name__ == '__main__':
    embed_size, num_hiddens, num_layers, drop_out = 32, 32, 2, 0.1
    batch_size, num_steps = 64, 10
    lr, num_epochs, device = 0.005, 300, d2l.try_gpu()
    train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
    encode = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, drop_out)
    decode = Seq2SeqDecode(len(tgt_vocab), embed_size, num_hiddens, num_layers, drop_out)
    net = EncoderDecoder(encode, decode)
    net.to(device)
    if not os.path.exists('param.pkl'):
        train(train_iter, net, lr, tgt_vocab, num_epochs, device)
        torch.save(net.state_dict(),'param.pkl')
    else:
        net.load_state_dict(torch.load('param.pkl'))
        net.to(device)


    # 进行验证(单独的一个句子的验证)
    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    # 首先需要将输入的句子转换为encode所熟悉的输入
    test_sentence = []
    for i in engs: # i,j为其中的一个句子对
        input_sentence = i.split(' ') + ['<eos>'] # 添加截止的符号
        pad_sentence = pad_content(input_sentence,num_steps) # 取出规定steps
        pad_sentence_seq = src_vocab[pad_sentence]
        test_sentence.append(torch.tensor(pad_sentence_seq).reshape(1,-1))
    test_sentence_seq = torch.cat(test_sentence) # [batch,steps]
    test_sentence_seq = test_sentence_seq.to(device) # 对于encode输入的序列来说其时相同的
    enc_outs, enc_last_h = encode(test_sentence_seq)
    n_best = 3
    batch_sentence_pred = beam_search.beam_search_decoding(decode,enc_outs, enc_last_h,
                        beam_width=5, n_best=n_best, sos_token=2, eos_token=3, max_dec_steps=1000,device=device)
    for i in range(len(fras)):
        for j in range(n_best):
            print(f'第{i}个样本的第{j+1}种可能')
            sentence_pred = batch_sentence_pred[i][j]
            sentence_pred = tgt_vocab.to_tokens(sentence_pred) # 包含起始符号和结尾的符号
            sentence_pred = ' '.join(sentence_pred[1:-1])
            print(sentence_pred)
            print('blue:', blue(sentence_pred, fras[i], k=2))
        print('真实的翻译句子:',fras[i])
        print('*' * 50)

