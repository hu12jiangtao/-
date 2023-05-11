import re
import collections
import random
import sysconfig

import torch

def load_file():
    path='D:\\python\\pytorch作业\\序列模型\\data\\timemachine.txt'
    with open(path,'r') as f:
        lines=f.readlines()  # 生成列表
    lines=[re.sub('[^A-Za-z]+',' ',line).lower().strip() for line in lines]
    return lines # 此时一行为一个列表元素

def create_token(lines,mode='word'):
    if mode == 'word':
        tokens=[line.split() for line in lines]
    elif mode == 'char':
        tokens=[list(line) for line in lines]
    return tokens

def collect_count(tokens):
    if isinstance(tokens[0],list):
        return collections.Counter(token for line in tokens for token in line)
    else:
        return collections.Counter(token for token in tokens)


class Vocab():
    def __init__(self,tokens,limit_num,collect_count):
        self.tokens=tokens
        self.sort_token=sorted(collect_count(self.tokens).items(),key=lambda X:X[1],reverse=True)
        self.save_vocab=['unk']
        self.unk=0
        self.save_vocab+=[name for name,time in self.sort_token if time>limit_num and name not in self.save_vocab]
        self.idx_to_token,self.token_to_idx=[],{}
        for i in self.save_vocab:
            self.idx_to_token.append(i)
            self.token_to_idx[i]=len(self.idx_to_token)-1

    def __getitem__(self, input_tokens): # 将字符转换为数字
        if not isinstance(input_tokens,(list,tuple)):
            return self.token_to_idx.get(input_tokens,self.unk)
        return [self.__getitem__(i) for i in input_tokens]

    def to_tokens(self,input_num):
        if not isinstance(input_num,(list,tuple)):
            return self.idx_to_token[input_num]
        return [self.idx_to_token[i] for i in input_num]

    def __len__(self):
        return len(self.idx_to_token)

def create_vocab(max_tokens=-1):
    lines=load_file()
    tokens=create_token(lines,mode='word')
    vocab=Vocab(tokens,0,collect_count)
    corpus=[vocab[token] for line in tokens for token in line]
    if max_tokens>0:
        corpus=corpus[:max_tokens]
    return vocab,corpus

# 形成批量的数据（定义两种批量的分类方法）

def seq_data_iter_random(corpus,batch_size,num_steps): # num_steps=seq
    corpus=corpus[random.randint(0,num_steps-1):]
    num_little=len(corpus)//num_steps
    start_index=list(range(0,num_little*num_steps,num_steps))
    random.shuffle(start_index)

    def data_output(start):
       return corpus[start:start+num_steps]

    num_batch=num_little//batch_size
    for i in range(0,num_batch*batch_size,batch_size):
        batch_start_index=start_index[i:i+batch_size]
        X=[data_output(i) for i in batch_start_index]
        Y=[data_output(i+1) for i in batch_start_index]
        yield torch.tensor(X),torch.tensor(Y)

def seq_data_iter_seq(corpus,batch_size,num_steps):
    offset=random.randint(0,num_steps)
    num_use_data=(len(corpus)-offset-1)//batch_size*batch_size
    X_s=torch.tensor(corpus[offset:offset+num_use_data])
    Y_s = torch.tensor(corpus[offset+1:offset + num_use_data+1])  # 当(len(corpus)-offset-1)整除时正好取到corpus的最后一个元素
    X_s,Y_s=X_s.reshape(batch_size,-1),Y_s.reshape(batch_size,-1)
    num_batch=X_s.shape[1]//batch_size
    for i in range(0,num_batch*batch_size,batch_size):
        X=X_s[:,i:i+num_steps]
        Y=Y_s[:,i:i+num_steps]
        yield X,Y

# 进行一次总结

class create_data1():
    def __init__(self,batch_size,num_steps,max_tokens,is_random,seq_data_iter_random,seq_data_iter_seq):
        self.batch_size=batch_size
        self.num_steps=num_steps
        self.vocab, self.corpus = create_vocab(max_tokens)
        if is_random == True:
            self.output=seq_data_iter_random
        else:
            self.output=seq_data_iter_seq

    def __iter__(self):
        out=self.output(self.corpus,self.batch_size,self.num_steps)
        return out

def load_data_time_machine(batch_size,num_steps,is_random=False,max_tokens=10000):
    train_iter=create_data1(batch_size,num_steps,max_tokens,is_random,seq_data_iter_random,seq_data_iter_seq)
    return train_iter,train_iter.vocab

train_iter,vocab=load_data_time_machine(batch_size=32,num_steps=35,is_random=True,max_tokens=10000)
for X,y in train_iter:
    print(X.shape)
    print(y.shape)
    break