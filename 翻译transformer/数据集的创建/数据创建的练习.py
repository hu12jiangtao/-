# '\u202f'，'\xa0'在utf-8中代表的是不同类型的空格符号
import collections
import torch
from torch.utils import data


def load_file():
    path='D:\\python\\pytorch作业\\序列模型\\data\\fra-eng\\fra.txt'
    with open(path,'r',encoding='utf-8') as f:
        data=f.read()
    return data  # 输出的为一个字符串

def deal_data(data):
    data=data.replace('\u202f',' ').replace('\xa0',' ').lower()
    def add_space(char,pre_char):
        return char in [',','.','!','?'] and pre_char != ' '
    data=[' '+char if i>0 and add_space(char,data[i-1]) else char for i,char in enumerate(data)]
    return ''.join(data)

def create_tokens(data,max_tokens=None):
    target,source=[],[]
    data=data.split('\n')
    for i,line in enumerate(data):
        if max_tokens and i>max_tokens:
            break
        parts=line.split('\t')
        if len(parts)==2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))

    return source,target


def collection(tokens):
    if tokens==None or isinstance(tokens[0],list):
        tokens=[token for line in tokens for token in line]
    else:
        tokens=[token for token in tokens]
    return collections.Counter(tokens)

class Vocab():   #reserved_tokens=['<pad>', '<bos>', '<eos>']
    def __init__(self,tokens,limit_num,reserved_tokens):
        if tokens == None:
            tokens=[]
        if reserved_tokens == None:
            reserved_tokens=[]
        self.name_num=sorted(collection(tokens).items(),key=lambda x:x[1],reverse=True)
        self.unk=0
        self.vocab_list=['<unk>']+reserved_tokens
        self.vocab_list+=[name for name,num in self.name_num if name not in self.vocab_list and num>=limit_num]
        self.idx_to_tokens,self.tokens_to_idx=[],{}
        for i in self.vocab_list:
            self.idx_to_tokens.append(i)
            self.tokens_to_idx[i]=len(self.idx_to_tokens)-1
    def __len__(self):
        return len(self.idx_to_tokens)
    def __getitem__(self, input_vocab):
        if not isinstance(input_vocab,(list,tuple)):
            return self.tokens_to_idx.get(input_vocab,self.unk)
        return [self.__getitem__(i) for i in input_vocab]
    def to_tokens(self,input_num):
        if not isinstance(input_num,(tuple,list)):
            return self.idx_to_tokens[input_num]
        return [self.idx_to_tokens[i] for i in input_num]



def prepare_padding_char(line,num_steps,pad_char_num):  # 此时的line为一行词汇对应的序列,为一个列表
    if len(line)>num_steps:
        return line[:num_steps]
    return line+[pad_char_num]*(num_steps-len(line))

def padding_char(lines,vocab,num_steps):  # 此时的lines为所有行的词汇
    lines=[vocab[line] for line in lines]
    lines=[line+[vocab['<eos>']] for line in lines]
    array=torch.tensor([prepare_padding_char(line,num_steps,vocab['<pad>']) for line in lines])
    valid_num=(array!=vocab['<pad>']).type(torch.int32).sum(1)
    return array,valid_num

def create_data_iter(batch_size,num_steps,num_exampls=600):
    text = deal_data(load_file())
    source,target=create_tokens(text, max_tokens=num_exampls)
    src_vocab=Vocab(source,2,['<pad>', '<bos>', '<eos>'])
    tgt_vocab=Vocab(target,2,['<pad>', '<bos>', '<eos>'])
    source_array,source_valid_num=padding_char(source, src_vocab, num_steps)
    target_array, target_valid_num = padding_char(target, tgt_vocab, num_steps)
    data_array=(source_array,source_valid_num,target_array, target_valid_num)
    def load_array(data_array,batch_size,is_train=True):
        dataset=data.TensorDataset(*data_array)
        return data.DataLoader(dataset,batch_size=batch_size,shuffle=is_train)
    train_iter=load_array(data_array,batch_size)
    return train_iter,src_vocab,tgt_vocab

batch_size=2
train_iter,src_vocab,tgt_vocab=create_data_iter(batch_size, num_steps=8, num_exampls=600)
for source_array,source_valid_num,target_array,target_valid_num in train_iter:
    print('X:', source_array.type(torch.int32))
    print('X的有效长度:', source_valid_num)
    print('Y:', target_array.type(torch.int32))
    print('Y的有效长度:', target_valid_num)
    break