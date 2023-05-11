import os
import torch
import re
import collections
from torch.utils import data
from torch import nn

def load_file():
    path='D:\\python\\pytorch作业\\序列模型\\data\\timemachine.txt'
    with open(path,'r',encoding='utf-8') as f:
        load_data = f.readlines()
    lines = [re.sub('[^A-Za-z]+',' ',i).strip().lower() for i in load_data]
    return lines

def create_mode(lines,mode='word'):
    if mode == 'word':
        return [line.split(' ') for line in lines]
    elif mode == 'char':
        return [list(line) for line in lines]

def collection_num(tokens):
    if isinstance(tokens[0],list):
        tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

class Vocab:
    def __init__(self,tokens,limit_seq):
        if tokens is None:
            tokens = []
        self.collect_tokens = sorted(collection_num(tokens).items(),key=lambda x:x[1],reverse=True)
        self.unk = 0
        self.tokens_lst = ['unk']
        self.tokens_lst += [name for name,num in self.collect_tokens if num >= limit_seq and name not in self.tokens_lst]
        self.vocab_to_idx,self.idx_to_vocab=dict(),[]
        for name in self.tokens_lst:
            self.idx_to_vocab.append(name)
            self.vocab_to_idx[name] = len(self.idx_to_vocab) - 1

    def __len__(self):
        return len(self.idx_to_vocab)

    def __getitem__(self, item):
        if not isinstance(item,(tuple,list)):
            return self.vocab_to_idx.get(item,self.unk)
        return [self.__getitem__(i) for i in item]

    def to_tokens(self,item):
        if not isinstance(item,(tuple,list)):
            return self.idx_to_vocab[item]
        return [self.idx_to_vocab[i] for i in item]

def create_batch_data(tokens,vocab,n_gram,batch_size): # n_gram代表有之前的n_gram个词决定之后的一个词
    corpus = [vocab[token] for line in tokens for token in line]
    lst = []
    for i in range(len(corpus) - n_gram):
        lst.append(corpus[i:i + n_gram + 1])
    train_data = torch.tensor([i[:-1] for i in lst])
    test_data = torch.tensor([i[-1] for i in lst])
    dataset = data.TensorDataset(*[train_data,test_data])
    dataloader = data.DataLoader(dataset,shuffle=True,batch_size=batch_size)
    return dataloader

class Net(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,n_gram):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embed_size)
        self.linear = nn.Linear(n_gram * embed_size, hidden_size)
        self.activate = nn.Tanh()
        self.linear2 = nn.Linear(hidden_size,vocab_size)

    def forward(self,x): # [batch,3]
        embed_out = self.embedding(x) # [batch,3,embedding_size]
        embed_out = embed_out.reshape(embed_out.shape[0], -1) # [batch,3 * embedding_size]
        out = self.activate(self.linear(embed_out))
        out = self.linear2(out)
        return out

def predict(sentence,pred_num,vocab,net,device,n_gram): # sentence为一个字符串
    sentence_lst = list(sentence)
    sentence_index_lst = vocab[sentence_lst]
    net = net.to(device)
    for _ in range(pred_num):
        input_data = torch.tensor(sentence_index_lst[-n_gram:],device=device).reshape(1,-1) # [1,3]
        y_hat = net(input_data) # [1,vocab_size]
        prob_index = torch.argmax(y_hat,dim=-1)[0].item() # 常数
        sentence_index_lst.append(prob_index)
    return ''.join(vocab.to_tokens(sentence_index_lst))


if __name__ == '__main__':
    lines = load_file()
    tokens = create_mode(lines, mode='char')
    vocab = Vocab(tokens,limit_seq=1)
    # 创建数据集
    n_gram = 10
    batch_size = 256
    embed_size = 100
    hidden_size = 256
    vocab_size = len(vocab)
    device = torch.device('cuda')
    num_epoch = 100
    dataloader = create_batch_data(tokens, vocab, n_gram, batch_size)
    # 创建神经网络语言模型
    net = Net(embed_size,hidden_size,vocab_size,n_gram)
    net.to(device)
    # 迭代方式
    opt = torch.optim.Adam(net.parameters(),lr=1e-3)
    loss = nn.CrossEntropyLoss()
    # 进行迭代
    if not os.path.exists('params.param'):
        for i in range(num_epoch):
            for index,(X,y) in enumerate(dataloader):
                X,y = X.to(device),y.to(device)
                y_hat = net(X)
                l = loss(y_hat,y)
                opt.zero_grad()
                l.backward()
                opt.step()
                if (index + 1) % 200 == 0:
                    print(f'epoch:{i + 1} iter:{index + 1} loss:{l:1.4f}')
            pred_sentence = predict(sentence='time machine', pred_num=50, vocab=vocab, net=net, device=device,n_gram=n_gram)
            print(pred_sentence)
            print('*' * 100)
            torch.save(net.state_dict(),'params.param')
    else:
        net.load_state_dict(torch.load('params.param'))

    # 进行预测
    pred_num = 50
    sentence = 'time machine'
    pred_sentence = predict(sentence,pred_num,vocab,net,device,n_gram)
    print(pred_sentence)