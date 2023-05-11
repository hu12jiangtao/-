import collections
import re
import random
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import math

def read_file():
    path='D:\\python\\pytorch作业\\序列模型\\data\\timemachine.txt'
    with open(path,'r') as f:
        datas=f.readlines()
    lines=[re.sub('[^A-Za-z]+',' ',data).strip().lower() for data in datas]
    return lines  # 此时每一行为一个列表元素

def load_tokens(lines,mode='word'):
    if mode == 'word':
        tokens=[line.split() for line in lines]
    elif mode == 'char':
        tokens=[list(line) for line in lines]
    return tokens

def create_data(tokens): # 统计单词出现的数量
    if isinstance(tokens[0],list):
        out_dict=collections.Counter([token for line in tokens for token in line])
    else:
        out_dict=collections.Counter([token for token in tokens])
    return out_dict


class MyDict():
    def __init__(self,tokens,limit_num=0):
        self.tokens=tokens
        out_dict=create_data(self.tokens)
        self.out_tuple=sorted(out_dict.items(),key=lambda x:x[1],reverse=True) # 生成按顺序排列的可遍历的元组
        self.unk=0
        self.vocab_list=['<unk>']
        self.vocab_list+=[name for name,times in self.out_tuple if times>limit_num and name not in self.vocab_list]
        self.idx_to_token,self.token_to_idx=[],{}
        for i in self.vocab_list:
            self.idx_to_token.append(i)
            self.token_to_idx[i]=len(self.idx_to_token)-1

    def __getitem__(self, item):  # 将单词转换为数字
        if not isinstance(item,(list,tuple)):
            return self.token_to_idx.get(item,self.unk)
        else:
            return [self.__getitem__(i) for i in item]

    def to_head(self,nums):  #将数字转换为单词
        if not isinstance(nums,(list,tuple)):
            return self.idx_to_token[nums]
        else:
            return [self.idx_to_token[num] for num in nums]

    def __len__(self):
        return len(self.idx_to_token)


def load_corpus_time_machine(max_tokens=-1):
    lines=read_file()
    tokens=load_tokens(lines,mode='char')
    vocab=MyDict(tokens)
    corpus = [token for line in tokens for token in line]
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return vocab,corpus

# 生成批量大小
def seq_data_iter_random(corpus,batch_size,num_steps):  # num_steps代表的是时间序列
    corpus=corpus[random.randint(0,num_steps-1):]
    num_little=(len(corpus)-1)//num_steps  # 子片段的数量
    start_index=list(torch.arange(0,num_little*num_steps,num_steps))
    random.shuffle(start_index)

    def output(start_index):
        return corpus[start_index:start_index+num_steps]

    num_batch=num_little//batch_size  # 批量的数量
    for i in range(0,batch_size*num_batch,batch_size):
        batch_start_index=start_index[i:i+batch_size]
        X=[output(i) for i in batch_start_index]
        Y=[output(i+1) for i in batch_start_index]
        yield X,Y

def seq_data_iter_sequential(corpus,batch_size,num_steps):  # 此时前一个batch的第i个样本和后一个batch的第i个样本相联系
    offset=random.randint(0,num_steps)
    need_numel=(len(corpus)-offset-1)//batch_size*batch_size  # 可以分为多少个元素

    X_s=torch.tensor(corpus[offset:need_numel+offset])  # 一共有need_numel个元素，且这些元素可以被batch_size整除
    Y_s=torch.tensor(corpus[offset+1:need_numel+offset+1])

    X_s,Y_s=X_s.reshape(batch_size,-1),Y_s.reshape(batch_size,-1)
    num_batch=X_s.shape[1]//num_steps
    for i in range(0,num_batch*batch_size,batch_size):
        X=X_s[:,i:i+num_steps]
        Y=Y_s[:,i:i + num_steps]
        yield X,Y

# 创建一个类，可以选择不同的方式
class SeqDataLoader:
    def __init__(self,use_random_iter,batch_size,num_steps,max_tokens):
        self.vocab,self.corpus=load_corpus_time_machine(max_tokens)
        self.batch_size,self.num_steps=batch_size,num_steps
        if use_random_iter:
            self.load=seq_data_iter_random
        else:
            self.load=seq_data_iter_sequential

    def __iter__(self):
        return self.load(self.corpus,self.batch_size,self.num_steps)

def load_data_time_machine(batch_size, num_steps,use_random_iter=False,max_tokens=10000):
    train_iter=SeqDataLoader(use_random_iter,batch_size,num_steps,max_tokens)
    return train_iter,train_iter.vocab


