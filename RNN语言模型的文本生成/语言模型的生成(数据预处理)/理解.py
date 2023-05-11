# 语言模型的原理:求解文本序列的联合概率
# 相对于之前的mlp来说预测的长度变长了（但是任然是属于比较短的，在100以内）

# 语言模型可以用来预测多个序列的可能性大小
# 语言模型的作用:作为一个预训练模型
# 语言模型的另一个运用:通过已知的样本预测之后可能出现的单词

# 使用计数的方式对其进行建模(序列长度为3):此时联合概率p(x1,x2,x3)=p(x1)*p(x2|x1)*p(x3|x1,x2)=[n(x1)/n]*[n(x2,x1)/n(x1)]*[n(x1,x2,x3)/n(x1,x2)]
# 其中n代表文本中含有的单词的个数，n1代表x1出现的次数；n(x2,x1)代表的是x1，x2顺序出现的次数；n(x1,x2,x3)代表的是x1，x2，x3顺序出现的次数
# 当序列很长，但是文本规模较小的时候，很有可能会出现 某个序列在整个文本中出现的次数为0 ，此时就会引入一个n元语法，n元语法当前只和前面n-1个次有关（引入了马尔可夫假设,一般使用2、3元语法）
# 例如二元语法为一个词只和前面一个词有关p(x1,x2,x3)=p(x1)*p(x2|x1)*p(x3|x1,x2)
# 就可以化简为p(x1)*p(x2|x1)*p(x3|x2)=[n/n(x1)]*[n(x1,x2)/n(x1)]*[n(x2,x3)/n(x2)]

# 例如此时Vocab的大小为1000，此时利用二元语法时，其空间复杂度为o(n^2),1000^2单元存储Vocab的长度为2的组合在token(文本)中出现的次数；
# 如果没用利用二元语法，序列长度为10的句子的空间复杂度为o(n^10)，大大的降低了复杂度

import random
import torch
from d2l import torch as d2l
import re
import collections

def load_file():
    path='D:\\python\\pytorch作业\\序列模型\\data\\timemachine.txt'
    with open(path,'r') as f:
        data=f.readlines()
    lines=[re.sub('[^A-Za-z]+',' ',i).lower().strip() for i in data] # strip()用来移除每行的首尾的换行符
    return lines

def create_mode(lines,mode='word'):
    if mode == 'word':
        return [i.split() for i in lines] # 将每行的每个单词分割开来,如果利用的是词模式时不用考虑空格，只要在预测时一个词预测结束时加上空格即可
    elif mode == 'char':
        return [list(i) for i in lines]

def collection(tokens):
    if len(tokens)==0 or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab():
    def __init__(self,token=None,limit_freq=0):
        if token == None:
            token=[]
        vocab_dict=collection(token)
        self.seq_vocab_dict=sorted(vocab_dict.items(),key=lambda x:x[1],reverse=True) # 从大到小进行排列
        self.unk,self.vocab_list=0,['<unk>']
        self.vocab_list+=[i for i,j in self.seq_vocab_dict if j>limit_freq and i not in self.vocab_list]
        self.vocab_to_idx,self.idx_to_vocab=dict(),[]
        for i in self.vocab_list:
            self.idx_to_vocab.append(i)
            self.vocab_to_idx[i]=len(self.idx_to_vocab)-1

    def __len__(self):
        return len(self.idx_to_vocab)

    def __getitem__(self, index): # 输入vocab 给出 序列
        if not isinstance(index,(tuple,list)):
            return self.vocab_to_idx.get(index,self.unk)
        return [self.__getitem__(i) for i in index]

    def to_head(self,num):  # 输入数字给出字符
        if not isinstance(num,(tuple,list)):
            return self.idx_to_vocab[num]
        return [self.idx_to_vocab[i] for i in num]

def create_vocab(max_token=-1):  # 创建了一个对应的字典
    lines=load_file()
    tokens=create_mode(lines,mode='word')
    vocab=Vocab(tokens)
    corpus=[vocab[token] for line in tokens for token in line]  # token为一个单词的对应标号
    if max_token>0:
        corpus=corpus[:max_token]
    return vocab, corpus

vocab,corpus=create_vocab()
print(len(vocab),len(corpus))
'''
vocab,corpus=create_vocab()
print(len(vocab),len(corpus))

#利用二元语法
tokens=create_mode(load_file())
double_token=[pair for pair in zip(vocab.to_head(corpus[:-1]),vocab.to_head(corpus[1:]))]
double_vocab=Vocab(double_token)
print(double_vocab.seq_vocab_dict[:10])
'''



# 生成训练样本（没有顺序的），两个batch之间是独立的，不相关的
# 之前数据处理中的滑动取批量的做法非常贵
def seq_data_iter_random(corpus,batch_size,num_steps): # 遍历一次文本产生的随机的data_iter,randint中的参数都是可以取到的
    # corpus为一个文本中按顺序排列的单词，batch_size为生成的train_iters的批量个数，num_steps代表利用num_steps个点来预测之后的一个点
    corpus=corpus[random.randint(0,num_steps-1):] # 随即切除开头的一部分
    num_subseq=(len(corpus)-1)//num_steps  # 遍历整个corpus，剩下的corpus可以多少个子序列
    # 这里的-1是防止randint正好删去一个steps_size的长度，这样当num_subseq//batch_size也正好整除时会导致Y超出index
    initial_indices=list(range(0,num_steps*num_subseq,num_steps))  # 每一个子序列开头的单词的序列构成的列表
    random.shuffle(initial_indices)  # 将这些的序列的顺序打乱（随机）

    def data(pos):  # 传入初始序列，输出为在corpus上的单词
        return corpus[pos:num_steps+pos]

    num_batch=num_subseq//batch_size  # 每一个batch中包含了多少个子序列
    for i in range(0,num_batch*batch_size,batch_size):
        initial_indices_per_batch=initial_indices[i:i+batch_size]  # 其为1个batch开头单词索引组成的列表
        X=[data(i) for i in initial_indices_per_batch]
        # 此时通过X中第一个元素预测Y中第一个元素，X中第一、二个元素预测Y中第二个元素 直至 X中第5个元素预测Y中第五个元素
        Y=[data(i+1) for i in initial_indices_per_batch] # 例如X=[13,18,17,19,16],此时Y=[18,17,19,16,14] ，此时输入输出的是corpus上的单词(或者对应的数字)
        yield torch.tensor(X), torch.tensor(Y)

'''
random.seed(1)
my_seq = [0,4,3,5,9,8,6,2,10,7,15,13,18,17,19,16,14,12,11,1]
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
'''


'''
my_seq = list(range(35))
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5): # 生成的个数为(35-5)/6/2=3 存在3个batch
    print('X: ', X, '\nY:', Y)
'''

# 第二种生成训练样本的方式(第一个batch中的第一个样本的最后一个值和第二个batch中的第一个样本的开头是相互连接的)，两个batch之间是相互关联的
def seq_data_iter_sequential(corpus, batch_size, num_steps): # corpus中应当包含0
    offset=random.randint(0,num_steps)  # 随机删去开头的部分元素
    num_tokens= (len(corpus)-offset-1)//batch_size*batch_size # 此时表现出参与batch构建的corpus中元素的个数，此时一定会被batch_size整除
    # (len(corpus)-offset-1)可以被batch_size整除的时候变为corpus[offset:len(corpus)-1],
    # 确保了Xs的最后一个元素为corpus倒数第二个元素，# 同时Ys的最后一个元素为corpus的最后一个元素，（注意:列表的最后一个序列为len(corpus)-1）
    # (len(corpus)-offset-1)不被整除时，此时Xs的取值也不会超过corpus索引范围
    Xs=torch.tensor(corpus[offset:offset+num_tokens])
    Ys=torch.tensor(corpus[offset+1:offset+num_tokens+1]) # Ys为Xs向后偏移一位

    Xs,Ys=Xs.reshape(batch_size,-1),Ys.reshape(batch_size,-1)
    num_batch=Xs.shape[1]//num_steps  # 遍历一次后可以创造出来的batch的个数
    for i in range(0,num_batch*num_steps,num_steps):
        X=Xs[:,i:i+num_steps]
        Y=Ys[:,i:i+num_steps]
        yield X,Y

#'''
random.seed(1)
my_seq = list(range(35))
for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
#'''

class SeqDataLoader:
    def __init__(self,batch_size,num_steps,use_random_iter,max_tokens):
        if use_random_iter:
            self.create_iter=seq_data_iter_random
        else:
            self.create_iter = seq_data_iter_sequential

        self.vocab,self.corpus=create_vocab(max_tokens)
        self.batch_size,self.num_steps=batch_size,num_steps

    def __iter__(self):  # 其为一个迭代器，只在创建类对象的时候运行一次（用法和__init__相类似）
        return self.create_iter(self.corpus,self.batch_size,self.num_steps)

def load_data_time_machine(batch_size, num_steps,
                           use_random_iter=False, max_tokens=10000):
    data_iter=SeqDataLoader(batch_size,num_steps,use_random_iter,max_tokens)
    return data_iter,data_iter.vocab

'''
data_iter,vocab=load_data_time_machine(batch_size=100, num_steps=5,
                           use_random_iter=False, max_tokens=10000)

print(vocab.vocab_to_idx)  # 导出的data_iter.vocab可以查看对象中的vocab_to_idx
'''

'''
# 同时__item__,__next__可以构成一个迭代器
class Fib:
    def __init__(self,max):
        self.max=max
    def __iter__(self):
        print('__iter__ call')
        self.a=0
        self.b=1
        return self
    def __next__(self):
        print('__next__ call')
        fib=self.a
        if fib>self.max:
            raise StopIteration  # 停止进行迭代，跳出这个实例
        self.a,self.b=self.b,self.a+self.b
        return fib  # 返回当前迭代时a的值

for i in Fib(3):
    print(i)
'''

