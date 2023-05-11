import collections
import torch
from torch import nn
from d2l import torch as d2l
import re
import random

def load_file():
    path = 'D:\\python\\pytorch作业\\序列模型\\data\\timemachine.txt'
    with open(path,'r',encoding='utf-8') as f:
        lines=f.readlines()
    lines=[re.sub('[^A-Za-z]',' ',line).strip().lower() for line in lines]
    return lines # 一行为一个列表元素

def create_tokens(lines,mode='char'):
    if mode == 'char':
        tokens = [list(line) for line in lines]
    elif mode == 'word':
        tokens = [line.split() for line in lines]
    return tokens

def collections_collect(tokens):
    if isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    else:
        tokens=[i for i in tokens]
    return collections.Counter(tokens)

class Vocab():
    def __init__(self,tokens,limit_num=0):
        if tokens is None:
            tokens = []
        # 首先要计算出每个tokens出现的次数
        self.tokens_num=sorted(collections_collect(tokens).items(),key=lambda x:x[1],reverse=True)
        self.unk=0
        self.vocab_sum=['<unk>']
        self.vocab_sum+=[token for token,num in self.tokens_num if num>limit_num and token not in self.vocab_sum]
        self.idx_to_vocab,self.vocab_to_idx=[],{}
        for i in self.vocab_sum:
            self.idx_to_vocab.append(i)
            self.vocab_to_idx[i] = len(self.idx_to_vocab)-1

    def __getitem__(self, item):
        if not isinstance(item,(tuple,list)):
            return self.vocab_to_idx.get(item,self.unk)
        else:
            return [self.__getitem__(i) for i in item]

    def to_tokens(self,input_idx):
        if not isinstance(input_idx,(tuple,list)):
            return self.idx_to_vocab[input_idx]
        else:
            return [self.idx_to_vocab[i] for i in input_idx]

    def __len__(self):
        return len(self.idx_to_vocab)

def create_vocab(max_tokens=-1):
    lines=load_file()  # 输出为一个元素个数为1的列表
    tokens=create_tokens(lines,mode='word')
    if max_tokens >0:
        tokens=tokens[:max_tokens]
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    return vocab,corpus

vocab,corpus=create_vocab()
print(len(vocab),len(corpus))

def seq_data_iter_random(corpus, batch_size, num_steps): # 此时每个batch之间不存在联系 # num_steps为步长
    offset = random.randint(0,num_steps-1)
    corpus = corpus[offset:]
    steps_num = len(corpus) // num_steps  # step个数
    start_index = list(torch.arange(0,num_steps*steps_num,num_steps))
    random.shuffle(start_index)

    def output(pos):
        return corpus[pos:pos+num_steps]

    batch_num = steps_num // batch_size
    for i in range(0,batch_num*batch_size,batch_num):
        index = start_index[i:i+batch_size]  # 获得了一个batch的所有的起始索引
        X=[output(i) for i in index]
        Y=[output(i+1) for i in index]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0,num_steps)
    num_seq = (len(corpus)-offset-1) // batch_size * batch_size  # 总共可用的长度
    X_c=torch.tensor(corpus[offset:offset+num_seq])
    Y_c=torch.tensor(corpus[offset+1:offset+num_seq+1])
    X_c,Y_c = X_c.reshape(batch_size,-1),Y_c.reshape(batch_size,-1)
    sum_steps = X_c.shape[1]
    steps_num = sum_steps // num_steps # steps_num为一行有多少个steps
    for i in range(0,steps_num*num_steps,num_steps):
        X = X_c[:,i:i+num_steps]
        Y = Y_c[:,i:i+num_steps]
        yield X,Y

# 创建一个类对象选择两种模式
class SeqDataLoader:
    def __init__(self,batch_size, num_steps,is_random=False,max_tokens=10000):
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.vocab,self.corpus=create_vocab(max_tokens=max_tokens)
        if is_random:
            self.way = seq_data_iter_random
        else:
            self.way = seq_data_iter_sequential
    def __iter__(self):
        return self.way(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps,is_random=False,max_tokens=10000)
    return data_iter,data_iter.vocab

data_iter,vocab=load_data_time_machine(batch_size=100, num_steps=5,
                           use_random_iter=False, max_tokens=10000)

print(vocab.vocab_to_idx)  # 导出的data_iter.vocab可以查看对象中的vocab_to_idx