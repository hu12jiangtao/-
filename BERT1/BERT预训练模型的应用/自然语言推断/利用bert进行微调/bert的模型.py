import random
import math
from torch.utils import data
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
import collections
'''
# 创建数据集
给定tokens：_read_wiki
nsp:_get_next_sentence,_get_tokens_and_segments,_get_nsp_data_from_paragragh
mlm_:replace_mlm_tokens,_get_mlm_data_from_tokens
pad:_pad_bert_inputs
汇总：WikiTextDataset
创建batch：load_data_wiki
'''
def _read_wiki(path):
    with open(path,'r',encoding='utf-8') as f:
        paragraghs= f.readlines()
    paragraghs = [paragragh.strip().lower().split(' . ') for  # 文章(列表1)->段落(列表2)->句子
                  paragragh in paragraghs if len(paragragh.split(' . ')) >= 2]
    return paragraghs

def tokenize(lines,mode='word'):
    if mode == 'word':
        return [line.split(' ') for line in lines]
    elif mode == 'char':
        return [list(line) for line in lines]

def collection_nums(tokens):
    if isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
    else:
        tokens = [token for token in tokens]
    return collections.Counter(tokens)

class Vocab():
    def __init__(self,tokens,limit_seq,reverse_tokens):
        super(Vocab, self).__init__()
        if tokens is None:
            tokens = []
        if reverse_tokens is None:
            reverse_tokens = []
        self.collect_tokens = sorted(collection_nums(tokens).items(),key=lambda x:x[1],reverse=True)
        self.unk = 0
        self.vocab_token = ['<unk>'] + reverse_tokens
        self.vocab_token += [name for name,num in self.collect_tokens
                             if num >= limit_seq and name not in self.vocab_token]
        self.idx_to_token,self.token_to_idx = [],{}
        for i in self.vocab_token:
            self.idx_to_token.append(i)
            self.token_to_idx[i] = len(self.idx_to_token) - 1
    def __getitem__(self, item):
        if not isinstance(item,(tuple,list)):
            return self.token_to_idx.get(item,self.unk)
        else:
            return [self.__getitem__(i) for i in item]
    def to_tokens(self,input_idx):
        if not isinstance(input_idx,(tuple,list)):
            return self.idx_to_token[input_idx]
        else:
            return [self.idx_to_token[i] for i in input_idx]
    def __len__(self):
        return len(self.idx_to_token)

def _get_next_sentence(sentence,next_sentence,paragraghs):
    if random.random() < 0.5:
        is_next = True
    else:
        next_sentence = random.choice(random.choice(paragraghs))
        is_next = False
    return sentence,next_sentence,is_next

def _get_tokens_and_segments(sentence,next_sentence=None):
    tokens = ['<cls>'] + sentence + ['<seq>']
    segments = [1] * (len(sentence) + 2)
    if next_sentence is not None:
        tokens = tokens + next_sentence + ['<seq>']
        segments = segments + [0] * (len(next_sentence) + 1)
    return tokens,segments

def _get_nsp_data_from_paragragh(paragragh,paragraghs,max_len):
    nsp_data_from_paragragh = []
    for i in range(len(paragragh)-1):
        sentence,next_sentence = paragragh[i],paragragh[i+1]
        sentence,next_sentence,is_next = _get_next_sentence(sentence, next_sentence, paragraghs)
        tokens,segments = _get_tokens_and_segments(sentence, next_sentence)
        if len(tokens) > max_len:  # 当拼接后的序列长度大于max_lens则抛弃这个样本
            continue
        nsp_data_from_paragragh.append((tokens,segments,is_next))
    return nsp_data_from_paragragh

def replace_mlm_tokens(tokens,effective_token_index,mask_len,vocab):
    mlm_pred_position_label = []
    tokens_copy = [token for token in tokens]
    random.shuffle(effective_token_index) # 打乱序列进行随机抽取
    for mask_position in effective_token_index:
        if len(mlm_pred_position_label) >= mask_len:
            break
        if random.random() < 0.8:
            mask_content = '<mask>'
        else:
            if random.random() < 0.5:
                mask_content = tokens[mask_position]
            else:
                mask_content = random.choice(vocab.idx_to_token)
        mlm_pred_position_label.append((mask_position,tokens[mask_position])) # 被遮掩或者替代的内容，位置
        tokens_copy[mask_position] = mask_content # 编码器的输入
    return tokens_copy,mlm_pred_position_label

def _get_mlm_data_from_tokens(tokens,vocab):  # 这个函数的输出是被遮掩或者替代的内容，位置，编码器的输入
    mask_len = max(1,round(len(tokens) * 0.15))
    effective_token_index = []
    for i in range(len(tokens)):
        if tokens[i] in ['<seq>','<cls>']:
            continue
        effective_token_index.append(i)
    tokens_copy,mlm_pred_position_label = replace_mlm_tokens(tokens, effective_token_index, mask_len, vocab)
    mlm_pred_position_label = sorted(mlm_pred_position_label,key=lambda x:x[0],reverse=False) # 重新进行排序
    mlm_pred_position = [i[0] for i in mlm_pred_position_label]
    mlm_pred_label = [i[1] for i in mlm_pred_position_label]
    return vocab[tokens_copy],mlm_pred_position,vocab[mlm_pred_label]

def _pad_bert_inputs(examples,max_len,vocab):
    mask_max_len = round(0.15 * max_len)
    all_segments,all_tokens_ids,valid_lens = [],[],[] # 长度为 max_len
    all_mlm_pred_position,all_mlm_pred_label,all_mlm_weight = [],[],[]  # 长度为 0.15 * max_len
    nsp_labels = []
    for tokens_input,mlm_pred_position,mlm_pred_label,segments,is_next in examples:
        all_segments.append(torch.tensor(segments + (max_len-len(segments))* [0],dtype=torch.long))
        all_tokens_ids.append(torch.tensor(tokens_input + (max_len-len(tokens_input)) * [vocab['<pad>']],dtype=torch.long))
        valid_lens.append(torch.tensor(len(tokens_input)))
        all_mlm_pred_position.append(torch.tensor(mlm_pred_position + (mask_max_len - len(mlm_pred_position)) * [0],dtype=torch.long))
        all_mlm_pred_label.append(torch.tensor(mlm_pred_label + (mask_max_len - len(mlm_pred_label)) * [0],dtype=torch.long))
        all_mlm_weight.append(torch.tensor([1] * len(mlm_pred_label) + (mask_max_len - len(mlm_pred_label)) * [0]))
        nsp_labels.append(torch.tensor(is_next,dtype=torch.long))
    return all_tokens_ids, all_segments, valid_lens,all_mlm_pred_position, all_mlm_weight, all_mlm_pred_label,nsp_labels

class WikiTextDataset(data.Dataset):
    def __init__(self,paragraghs,max_len):
        super(WikiTextDataset, self).__init__()
        # 首先创建词表
        paragraghs = [tokenize(paragragh) for paragragh in paragraghs] # 文章(列表1)->段落(列表2)->句子(列表3)->token
        sentence = [line for paragragh in paragraghs for line in paragragh] # 句子(列表1)->token
        self.vocab = Vocab(sentence,limit_seq=5,reverse_tokens=['<pad>','<cls>','<mask>','<seq>'])
        # 进行nsp的任务,得到编码器的输入
        all_paragraghs_tokens_segments = []
        for paragragh in paragraghs:
            all_paragraghs_tokens_segments.extend(_get_nsp_data_from_paragragh(paragragh, paragraghs,max_len))
        # 进行mlm的任务，得到新的编码器的输入
        examples = [_get_mlm_data_from_tokens(tokens,self.vocab)+(segments,is_next)
                    for tokens,segments,is_next in all_paragraghs_tokens_segments]
        # 进行填充操作（之前已经将合并后长度大于max_lend的剔除了）
        self.all_tokens_ids, self.all_segments, self.valid_lens, self.all_mlm_pred_position, \
        self.all_mlm_weight, self.all_mlm_pred_label, self.nsp_labels = _pad_bert_inputs(examples,max_len,self.vocab)

    def __getitem__(self, item):
        return self.all_tokens_ids[item], self.all_segments[item], self.valid_lens[item], self.all_mlm_pred_position[item], \
        self.all_mlm_weight[item], self.all_mlm_pred_label[item], self.nsp_labels[item]

    def __len__(self):
        return len(self.valid_lens)

def load_data_wiki(batch_size,max_len):
    path = 'D:\\python\\pytorch作业\\序列模型\\data\\wikitext-2\\wiki.train.tokens'
    paragraghs = _read_wiki(path)
    train_set = WikiTextDataset(paragraghs,max_len)
    train_iter = data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    return train_iter,train_set.vocab

# 多头注意力层
def prepare_softmax(X,valid_lens,fill):
    a = X.shape[1]
    mask = torch.arange(a,device=X.device)
    cmp = (mask[None,:] < valid_lens[:,None])
    X[~cmp] = fill
    return X

def mask_softmax(X,valid_lens):
    if valid_lens is None:
        F.softmax(X,dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens,X.shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        out = prepare_softmax(X.reshape(-1,X.shape[-1]), valid_lens, -1e6)
        return F.softmax(out.reshape(shape),dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    def forward(self,quiry,key,value,valid_lens):
        k = quiry.shape[-1]
        out = torch.bmm(quiry,key.permute(0,2,1))/math.sqrt(k) # [batch,num_q,num_k]
        mask_out = mask_softmax(out,valid_lens)
        return torch.bmm(self.dropout(mask_out),value)

def transpose_qkv(X,num_head): # [batch,num_q,num_hiddens]
    X = X.reshape(X.shape[0],X.shape[1],num_head,-1)
    X =X.permute(0,2,1,3)
    X = X.reshape(-1,X.shape[2],X.shape[3])
    return X

def transpose_outputs(X,num_head):
    X = X.reshape(-1,num_head,X.shape[1],X.shape[2])
    X = X.permute(0,2,1,3)
    X = X.reshape(X.shape[0],X.shape[1],-1)
    return X

class MultiHeadAttention(nn.Module):
    def __init__(self,quiry_size,key_size,value_size,num_hiddens,num_head,dropout):
        super(MultiHeadAttention, self).__init__()
        self.w_q = nn.Linear(quiry_size,num_hiddens)
        self.w_k = nn.Linear(key_size,num_hiddens)
        self.w_v = nn.Linear(value_size,num_hiddens)
        self.w_o = nn.Linear(num_hiddens,num_hiddens)
        self.num_head = num_head
        self.attention = DotProductAttention(dropout)
    def forward(self,quiries,keys,values,valid_lens):
        quiries = transpose_qkv(self.w_q(quiries),self.num_head)  # [batch*num_head,num_q,num_hiddens/num_head]
        keys = transpose_qkv(self.w_k(keys), self.num_head)
        values = transpose_qkv(self.w_v(values), self.num_head) # [batch*num_head,num_k,num_hiddens/num_head]
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens,self.num_head)
        out = self.attention(quiries,keys,values,valid_lens) # [batch*num_head,num_q,num_hiddens/num_head]
        out = transpose_outputs(out,self.num_head) # [batch,num_q,num_hiddens]
        return self.w_o(out) # [batch,num_q,num_hiddens]

class PositionWiseFFN(nn.Module):
    def __init__(self,ffn_num_input, ffn_num_hiddens,num_hiddens):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(ffn_num_input,ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(ffn_num_hiddens,num_hiddens)
    def forward(self,X):
        return self.linear2(self.relu(self.linear1(X)))

class AddNorm(nn.Module):
    def __init__(self,norm_shape,dropout):
        super(AddNorm, self).__init__()
        self.layernorm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)
    def forward(self,X,Y):
        return self.layernorm(self.dropout(Y)+X)

class EncoderBlock(nn.Module):
    def __init__(self,quiry_size,key_size,value_size,num_hiddens,norm_shape,
                 ffn_num_input, ffn_num_hiddens,num_head,dropout):
        super(EncoderBlock, self).__init__()
        self.addnorm1 = AddNorm(norm_shape,dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.attention = MultiHeadAttention(quiry_size,key_size,value_size,num_hiddens,num_head,dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,num_hiddens)
    def forward(self,X,valid_lens):
        Y1 = self.attention(X,X,X,valid_lens)
        Y2 = self.addnorm1(X,Y1)
        Y3 = self.ffn(Y2)
        Y4 = self.addnorm2(Y2,Y3)
        return Y4

class BERTEncoder(nn.Module):
    def __init__(self,vocab_size,quiry_size,key_size,value_size,num_hiddens,norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers,max_len,dropout):
        super(BERTEncoder, self).__init__()
        self.blks = nn.Sequential()
        self.token_embed = nn.Embedding(vocab_size,num_hiddens)
        self.segment_embed = nn.Embedding(2,num_hiddens)
        self.position = nn.Parameter(torch.randn([1, max_len, num_hiddens]))
        for i in range(num_layers):
            self.blks.add_module(f'blk{i}',EncoderBlock(quiry_size,key_size,value_size,num_hiddens,norm_shape,
                 ffn_num_input, ffn_num_hiddens,num_heads,dropout))
    def forward(self,tokens,segments,valid_lens):
        X = self.token_embed(tokens) + self.segment_embed(segments)
        X = X + self.position[:,X.shape[1],:]
        for blk in self.blks:
            X = blk(X,valid_lens)
        return X

class MaskLM(nn.Module): # 这一部分针对的是对掩码的预测
    def __init__(self,mlm_in_feature,norm_shape,num_hiddens,vocab_size):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(mlm_in_feature,num_hiddens),nn.ReLU(),
                                 nn.LayerNorm(norm_shape),nn.Linear(num_hiddens,vocab_size))
    def forward(self,X,pre_position): # 挑选出被掩盖的mask,X为编码器的输出
        batch = X.shape[0]
        mask_lens = pre_position.shape[1]
        pre_position = pre_position.reshape(-1)
        a = torch.repeat_interleave(torch.arange(batch,device=X.device),mask_lens)
        new_X = X[a,pre_position,:] # 此时为一个二维矩阵=[batch*每一个batch的mask_lens,num_hiddens]
        new_X = new_X.reshape(batch,mask_lens,-1)
        out = self.mlp(new_X)
        return out

class NextSentencePred(nn.Module):
    def __init__(self,nsp_in_feature):
        super(NextSentencePred, self).__init__()
        self.linear = nn.Linear(nsp_in_feature,2)
    def forward(self,X):
        return self.linear(X)

class BERTModel(nn.Module):
    def __init__(self,vocab_size,num_hiddens,norm_shape,ffn_num_input,ffn_num_hiddens,num_heads,
                 num_layers,dropout,max_len,key_size,quiry_size,value_size,hid_in_features,mlm_in_features,nsp_in_features):
        super(BERTModel, self).__init__()
        self.attention = BERTEncoder(vocab_size,quiry_size,key_size,value_size,num_hiddens,norm_shape,
                 ffn_num_input, ffn_num_hiddens, num_heads,
                 num_layers,max_len,dropout)
        self.MaskLM = MaskLM(mlm_in_features,norm_shape,num_hiddens,vocab_size)
        self.hiddens = nn.Linear(num_hiddens,hid_in_features)
        self.NextSentencePred = NextSentencePred(nsp_in_features)
    def forward(self,tokens,segments,valid_lens,pre_position=None):
        encode_X = self.attention(tokens,segments,valid_lens)
        nsp_y_hat = self.NextSentencePred(self.hiddens(encode_X[:,0,:])) # [batch,2]
        if pre_position is not None:
            mlm_y_hat = self.MaskLM(encode_X,pre_position)  # [batch,max_lens*0.15,vocab_size]
        else:
            mlm_y_hat = None
        return encode_X,mlm_y_hat,nsp_y_hat

class add_machine():
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self,*args):
        self.data = [i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

class MaskLoss(nn.CrossEntropyLoss):
    def forward(self,pred,labels,all_mlm_weight):
        self.reduction = 'none'
        l = super(MaskLoss, self).forward(pred.permute(0,2,1),labels)
        mask_l = (all_mlm_weight * l).mean()
        return mask_l

def compute_loss(mlm_y_hat,nsp_y_hat,all_mlm_weight, all_mlm_pred_label, nsp_labels):
    mlm_l = MaskLoss()(mlm_y_hat,all_mlm_pred_label,all_mlm_weight)
    nsp_l = loss(nsp_y_hat,nsp_labels) # 求解得到平均值
    l = mlm_l + nsp_l
    return mlm_l,nsp_l,l


def train_bert(train_iter, net, loss, vocab_size, device, num_steps):
    net.train()
    net.to(device)
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    trainer = torch.optim.Adam(net.parameters(),lr=1e-3)
    metric = add_machine(3)
    flag = False
    step = 0
    while flag == False and step < num_steps:
        for batch in train_iter:
            all_tokens_ids, all_segments, valid_lens, all_mlm_pred_position, \
            all_mlm_weight, all_mlm_pred_label, nsp_labels = [i.to(device) for i in batch]
            encode_X,mlm_y_hat,nsp_y_hat = net(all_tokens_ids,all_segments,valid_lens,all_mlm_pred_position)
            mlm_l,nsp_l,l = compute_loss(mlm_y_hat,nsp_y_hat,all_mlm_weight, all_mlm_pred_label, nsp_labels)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(mlm_l,nsp_l,1)
            animator.add(step + 1,
                         (metric[0] / metric[2], metric[1] / metric[2]))
            step += 1
            if step >= num_steps:
                flag = True
                break
        print(f'MLM loss {metric[0] / metric[2]:.3f}, '
              f'NSP loss {metric[1] / metric[2]:.3f}')

device = d2l.try_gpu()
batch_size, max_len = 512, 64
train_iter, vocab = load_data_wiki(batch_size, max_len)
net = BERTModel(len(vocab),num_hiddens=128,norm_shape=[128], ffn_num_input=128, ffn_num_hiddens=256, num_heads=2,
                 num_layers=2,dropout=0.2,max_len=1000,key_size=128, quiry_size=128, value_size=128,
                 hid_in_features=128, mlm_in_features=128,nsp_in_features=128)
loss = nn.CrossEntropyLoss()
train_bert(train_iter, net, loss, len(vocab), device, num_steps=50)