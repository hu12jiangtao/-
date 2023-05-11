# '\u202f'，'\xa0'在utf-8中代表的是不同类型的空格符号

import os
import torch
from d2l import torch as d2l
import collections
from torch.utils import data

d2l.DATA_HUB['fra-eng'] = (d2l.DATA_URL + 'fra-eng.zip',
                           '94646ad1522d915e7b0f9296181140edcf86a4f5')
data_dir = d2l.download_extract('fra-eng')

def read_data_nmt():
    with open(os.path.join(data_dir, 'fra.txt'), 'r',
             encoding='utf-8') as f:
        return f.read() # 此时输出的为整一篇文章，且整一篇文章当作一个字符串，字符串是可以利用索引的,此时每一行利用\t隔开

def preprocess_nmt(text): #对文本进行预处理，在标点符号前如果为空格不做任何的处理，如果不是空格则加入一个空格(被翻译的英文标点和文字没有被分割开)
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower() # 将文本中不同类型的空格替换成同一种类型
    def no_space(char,prev_char):  # char代表的是当前的字符，prev_char代表的是上一个字符
        return prev_char != ' ' and char in [',','.','?','!']
    text=[' '+char if i>0 and no_space(char,text[i-1]) else char for i,char in enumerate(text)]  # 此时为列表，需转换为文本
    return ''.join(text)  # 将文本转换为了字符串

def tokenize_nmt(text, num_examples=None):  # num_examples用来确认取文本的多少行
    source,target=[],[]
    for i,line in enumerate(text.split('\n')):  # print(text.split('\n'))使字符串变为了一个列表，每一行为一个元素
        if num_examples and i>num_examples:
            break
        parts=line.split('\t')  # 一行变为了一个列表，其中的元素变为了翻译前的内容和翻译后的内容
        if len(parts)==2:
            source.append(parts[0].split(' '))  # 被翻译内容
            target.append(parts[1].split(' '))
    return source,target  # 此时source中每一个元素为1个列表（包含一行中所有的需被翻译的单词或者标点），该列表中的元素为一行中的需被翻译的一个单词或标点

def collect_num(tokens):
    if len(tokens)==0 or isinstance(tokens[0],(tuple,list)):
        tokens=[token for line in tokens for token in line]
    else:
        tokens=[token for token in tokens]
    return collections.Counter(tokens)

class Vocab():
    def __init__(self,tokens,min_freq,reserved_tokens): # 是一个列表
        if tokens == None:
            tokens=[]
        if reserved_tokens == None:
            reversed_tokens=[]
        self.tokens_num=sorted(collect_num(tokens).items(),key=lambda x:x[1],reverse=True)
        self.vocab_list=['<unk>']+reserved_tokens
        self.unk=0
        self.vocab_list+=[name for name,num in self.tokens_num if num>=min_freq and name not in self.vocab_list]
        self.idx_to_tokens,self.tokens_to_idx=[],{}
        for i in self.vocab_list:
            self.idx_to_tokens.append(i)
            self.tokens_to_idx[i]=len(self.idx_to_tokens)-1
    def __getitem__(self, input_tokens):
        if not isinstance(input_tokens,(tuple,list)):
            return self.tokens_to_idx.get(input_tokens,self.unk)
        else:
            return [self.__getitem__(i) for i in input_tokens]
    def to_tokens(self,input_num):
        if not isinstance(input_num,(tuple,list)):
            return self.idx_to_tokens[input_num]
        else:
            return [self.idx_to_tokens[i] for i in input_num]
    def __len__(self):
        return len(self.idx_to_tokens)

def truncate_pad(line, num_steps, padding_token): # 将每一行填充或者删减到一个固定的长度num_steps，这样子输出序列的长度相同,line应该是一个列表，每个元素对应一个单词
    if len(line)>num_steps:
        return line[:num_steps]
    # 在语言模型中的训练样本的长度也是固定的
    return line+[padding_token]*(num_steps-len(line)) # 每一行都是一个训练样本，将训练样本的长度进行固定，这样后面可以进行批量操作

def build_array_nmt(lines, vocab, num_steps):  # lines对应的是单词
    lines=[vocab[line] for line in lines] # 将单词转换为序列
    lines=[line+[vocab['<eos>']] for line in lines] # 每一行结束时给出停止符 ，此列lines为列表套列表
    array=torch.tensor([truncate_pad(line, num_steps, vocab['<pad>']) for line in lines])
    valid_num=(array!=vocab['<pad>']).type(torch.int32).sum(1)  # valid_num为一个行向量，长度为batch_size
    return array,valid_num


def load_data_nmt(batch_size, num_steps, num_examples=600):  # 只取前600行
    raw_text = read_data_nmt()
    text = preprocess_nmt(raw_text)
    source, target = tokenize_nmt(text, num_examples)
    src_vocab=Vocab(source,min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab=Vocab(target,min_freq=2,reserved_tokens=['<pad>', '<bos>', '<eos>'])
    source_array,source_valid_num=build_array_nmt(source, src_vocab, num_steps)
    print(source_valid_num)
    target_array,target_valid_num=build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays=(source_array,source_valid_num,target_array,target_valid_num)
    def load_array(data_array,batch_size,is_train=True):
        dataset=data.TensorDataset(*data_array)
        return data.DataLoader(dataset,shuffle=is_train,batch_size=batch_size)
    train_iter=load_array(data_arrays,batch_size)
    return train_iter,src_vocab,tgt_vocab


batch_size=2
train_iter,src_vocab,tgt_vocab=load_data_nmt(batch_size, num_steps=8, num_examples=600)
for source_array,source_valid_num,target_array,target_valid_num in train_iter:
    print('X:', source_array.type(torch.int32))
    print('X的有效长度:', source_valid_num)
    print('Y:', target_array.type(torch.int32))
    print('Y的有效长度:', target_valid_num)
    break