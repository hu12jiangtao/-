import torch
from torch import nn
from d2l import torch as d2l
import json
import os
import re
from torch.utils import data

'''
load_pretrained_model
bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],  # 加载训练好的参数的bert模型
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
read_snli  premises,hypothese
get_tokens_and_segments
SNLIBERTDataset
  _preprocess
   _mp_worker
   _truncate_pair_of_tokens
'''

def load_pretrained_model(num_hiddens,ffn_num_hiddens,max_len):
    path = 'D:\\python\\pytorch作业\\序列模型\\BERT\\BERT预训练模型的应用\\data\\bert.small.torch'
    vocab = d2l.Vocab()
    vocab.idx_to_token = json.load(open(os.path.join(path,'vocab.json')))
    vocab.token_to_idx = {token:index for index,token in enumerate(vocab.idx_to_token)}
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256],  # 加载训练好的参数的bert模型
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    bert.load_state_dict(torch.load(os.path.join(path,'pretrained.params')))
    return bert,vocab

def read_snli(data_dir,is_train=True):
    def deal_sentence(s):
        s = re.sub('\(',' ',s)
        s = re.sub('\)',' ',s)
        s = re.sub('\s{2,}',' ',s)
        return s
    filename = os.path.join(data_dir,'snli_1.0_train.txt' if is_train==True else 'snli_1.0_test.txt')
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    with open(filename,'r',encoding='utf-8') as f:
        rows = [row.lower().split('\t') for row in f.readlines()] # 文章->行->句子
    first_sentence = [deal_sentence(row[1]) for row in rows if row[0] in label_set] # 文章->句子
    next_sentence = [deal_sentence(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return first_sentence,next_sentence,labels

def tokenize(lines,mode='word'):
    if mode == 'word':
        return [line.split(' ') for line in lines]
    elif mode == 'char':
        return[list(line) for line in lines]

def get_tokens_and_segments(first_sentence,next_sentence=None):
    tokens = ['<cls>'] + first_sentence + ['<seq>']
    segments = [1] * (len(first_sentence) + 2)
    if next_sentence is not None:
        tokens = tokens + next_sentence + ['<seq>']
        segments = segments + [0] * (len(next_sentence) + 1)
    return tokens,segments

class SNLIBERTDataset(data.Dataset):
    def __init__(self,dataset,max_lens,vocab=None):
        super(SNLIBERTDataset, self).__init__()
        self.first_next_all_tokens = [[token_a,token_b] for token_a,token_b in
                                  zip(*[tokenize(sentence) for sentence in dataset[:2]])]
        # [tokenize(sentence) for sentence in dataset[:2]] 得到 文章[first,next]->first->句子->词
        # *首先将文章分为first,next，之后zip将first中的元素1和next中的元素1进行组合
        # 最后得到结果 文章->一个batch的first_sentence,next_sentence->first_sentence->词元
        self.vocab = vocab
        self.labels = torch.tensor(dataset[2])
        self.max_lens = max_lens
        self.all_tokens_ids,self.all_segments,self.valid_lens = (self._preprocess(self.first_next_all_tokens))
        print('read ' + str(len(self.all_tokens_ids)) + ' examples')

    def _preprocess(self,first_next_all_tokens):
        out = map(self._mp_worker,first_next_all_tokens) # 取出列表中的单个元素
        out = list(out)
        tokens = [i[0] for i in out]
        segments = [i[1] for i in out]
        valid_lens = [i[2] for i in out]
        return torch.tensor(tokens,dtype=torch.long),torch.tensor(segments,dtype=torch.long),torch.tensor(valid_lens)

    def _mp_worker(self,first_next_tokens):
        first_sentence,next_sentence = first_next_tokens
        self._truncate_pair_of_tokens(first_sentence,next_sentence,self.max_lens)
        tokens,segments = get_tokens_and_segments(first_sentence,next_sentence)
        tokens = tokens + (self.max_lens - len(tokens)) * ['<pad>']
        segments = segments + (self.max_lens - len(segments)) * [0]
        valid_lens = len(tokens)
        return vocab[tokens],segments,valid_lens

    def _truncate_pair_of_tokens(self,first_sentence,next_sentence,max_lens):
        while len(first_sentence) + len(next_sentence) > max_lens - 3:
            if len(first_sentence) < len(next_sentence):
                next_sentence.pop()
            else:
                first_sentence.pop()

    def __getitem__(self, item):
        return (self.all_tokens_ids[item],self.all_segments[item],self.valid_lens[item]),self.labels[item]

    def __len__(self):
        return len(self.labels)


data_dir = 'D:\\python\\pytorch作业\\序列模型\\BERT\\data\\snli_1.0\\snli_1.0'
num_hiddens,ffn_num_hiddens,max_len = 256,512,512
bert,vocab = load_pretrained_model(num_hiddens,ffn_num_hiddens,max_len)
test_dataset = read_snli(data_dir,is_train=False)
test_set = SNLIBERTDataset(test_dataset,128,vocab)
test_iter = data.DataLoader(test_set,batch_size=64,shuffle=False)
for batch in test_iter:
    print(batch[0][0].shape)
    break




a = [[[1,2],[3,4]],[[5,6],[7,8]]]
for i in zip(a):
    print(i)

