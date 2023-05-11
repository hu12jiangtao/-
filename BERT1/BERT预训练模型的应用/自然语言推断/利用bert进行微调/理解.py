# bert的模型一般是用在语言的分类任务上的，对于生成类的一些任务并不好用
# 利用bert主要有两个部分:pretrain和fine-tuning两个任务，在pre-train中会使用mask对部分的词进行遮掩，但是在fine-tuning中是不存在对部分词的遮掩
# 这部分的是fine-tuning（微调任务）
import torch
from d2l import torch as d2l
import os
import json
import re
from torch.utils import data
from torch import nn

def load_pretrained_model(data_dir, num_hiddens, ffn_num_hiddens,  # 错了
                          num_heads, num_layers, dropout, max_len, devices):
    vocab = d2l.Vocab() # 加载一个空词表(bert模型的词典需要加载训练好的)
    '''
    # 对应的存储方式为:
    filename = 'vocab.json'
    with open(filename,'w') as f:
       json.dump(vocab.idx_to_token,f)
    '''
    vocab.idx_to_token = json.load(open(os.path.join(data_dir,
        'vocab.json')))
    vocab.token_to_idx = {token:index for index,token in enumerate(vocab.idx_to_token)}  # 此时已经完成字典的加载
    bert = d2l.BERTModel(len(vocab), num_hiddens, norm_shape=[256], # 加载训练好的参数的bert模型
                         ffn_num_input=256, ffn_num_hiddens=ffn_num_hiddens,
                         num_heads=4, num_layers=2, dropout=0.2,
                         max_len=max_len, key_size=256, query_size=256,
                         value_size=256, hid_in_features=256,
                         mlm_in_features=256, nsp_in_features=256)
    # 加载模型这一句话和torch.save(bert_model.state_dict(),'pretrained.params')相对应，将训练好的参数以字典的形式存储到pretrained.params
    bert.load_state_dict(torch.load(os.path.join(data_dir,'pretrained.params')))  # 读取当前路径下的文件中参数，并且加载到模型中去
    # 模型必须和训练时的一模一样（按字典的键的名称一一赋值）
    return bert,vocab

# 处理数据集上的内容
def read_snli(data_dir, is_train=True):
    def deal_data(s):
        s = re.sub('\(',' ',s)
        s = re.sub('\)',' ',s)
        s = re.sub('\s{2,}',' ',s)
        return s.strip()
    filename = os.path.join(data_dir,'snli_1.0_train.txt' if is_train==True else 'snli_1.0_test.txt')
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    with open(filename,'r',encoding='utf-8') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [deal_data(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [deal_data(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises,hypotheses,labels  # 此时premises为:文章(列表1)->一行的字符串


# 创建输入的数据集
def tokenize(lines,mode='word'):
    if mode == 'word':
        return [line.split(' ') for line in lines]
    elif mode == 'char':
        return [list(line) for line in lines]

class SNLIBERTDataset(data.Dataset):
    def __init__(self,dataset,max_len,vocab=None):
        super(SNLIBERTDataset, self).__init__()
        # 对于[d2l.tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]]来说应该是[premises,hypotheses],之后premises,hypotheses
        # 进入d2l.tokenize([s.lower() for s in sentences]) , 此时将premises变为了一个 文章(列表1)->句子(列表2)->单词，因此总共有三个列表
        # zip代表取出 同时取出premises,hypotheses中的第i个元素（句子列表） 之后 [p_tokens , h_tokens]得到premises,hypotheses中每个句子组成一个列表
        # 此时为[p_tokens , h_tokens],因此仍然是三个列表文章(列表1)->句子([premises,hypotheses])(列表2)->premises,hypotheses单独的句子->单词
        # 将premises,hypotheses拼接在在一起，为作为bert的输入做准备
        # 对于zip来说执政对列表或者元组，对于一个矩阵a来说 for i in zip(a) 和 for i in a相同
        all_premise_hypothesis_tokens = [[p_tokens , h_tokens] for p_tokens, h_tokens in
            zip(*[d2l.tokenize([s.lower() for s in sentences]) for sentences in dataset[:2]])]
        print(all_premise_hypothesis_tokens[0])
        self.labels = torch.tensor(dataset[2])
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(all_premise_hypothesis_tokens)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_premise_hypothesis_tokens):
        #print('电脑并行处理器的个数:',multiprocessing.cpu_count())
        #pool = multiprocessing.Pool(4)
        # map的用法和zio相似，其参数为一个function + n个迭代器 ，其中元组，列表，字典，集合都是可迭代对象->iter(可迭代对象)变为一个迭代器,之后返回一个迭代器
        # 例：a = map(function,iter([1,2]),iter([3,4])) -> list(a)=[4,6]  (function用来实现两个数的相加)
        # 对于迭代器可以利用next来获取中的每一个值
        out = map(self._mp_worker, all_premise_hypothesis_tokens) # all_premise_hypothesis_tokens为一个列表的迭代器，传入premises,hypotheses对应的一组句子
        out = list(out)
        all_token_ids = [
            token_ids for token_ids, segments, valid_len in out] # 获得已经确认长度的索引的编码器的tokens输入
        all_segments = [segments for token_ids, segments, valid_len in out] # 获得已经确认长度的索引的编码器的segments输入
        valid_lens = [valid_len for token_ids, segments, valid_len in out] # 获得每个batch的实际长度
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, premise_hypothesis_tokens):
        p_tokens, h_tokens = premise_hypothesis_tokens # 此时分别得到了一个batch中的premises,hypotheses句子

        self._truncate_pair_of_tokens(p_tokens, h_tokens) # 对于过长的句子进行删减，使得两个句子的长度等于max_len-3

        tokens, segments = d2l.get_tokens_and_segments(p_tokens, h_tokens) # 将两个句子进行合并

        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] \
                    * (self.max_len - len(tokens))  # 如果两个句子太短，则添加pad符号至长度max_len-3（已经完成序列化）
        segments = segments + [0] * (self.max_len - len(segments)) # segments和tokens同理
        valid_len = len(tokens) # 获得实际长度
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, p_tokens, h_tokens):  # 如果超出了最大的长度，则会对两个句子进行适当的删减
        # 为BERT输入中的'<CLS>'、'<SEP>'和'<SEP>'词元保留位置
        while len(p_tokens) + len(h_tokens) > self.max_len - 3:
            if len(p_tokens) > len(h_tokens):
                p_tokens.pop()  # 删除列表中的最后一个元素
            else:
                h_tokens.pop()

    def __getitem__(self, item):
        return (self.all_token_ids[item],self.all_segments[item],\
               self.valid_lens[item]),self.labels[item]

    def __len__(self):
        return len(self.all_token_ids)


# 对于句子标签的预测一般是使用<cls>对应的特征向量来进行预测的
class BERTClassifier(nn.Module):
    def __init__(self,bert):
        super(BERTClassifier, self).__init__()
        self.encode = bert.encoder
        self.fc = nn.Sequential(bert.hidden,nn.Linear(256,3))
    def forward(self,input):
        tokens, segments, valid_lens = input
        out = self.encode(tokens,segments,valid_lens)
        cls_out = out[:,0,:]
        y_hat = self.fc(cls_out)
        return y_hat



def try_gpu():
    if torch.cuda.device_count()>=1:
        return torch.device('cuda')
    else:
        return torch.device('cpu')

class add_machine():
    def __init__(self,n):
        self.data = [0.0]*n
    def add(self,*args):
        self.data = [i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, item):
        return self.data[item]

def accrucy(y_hat,y):
    a = torch.argmax(y_hat,dim=1)
    cmp = (a.type(y.dtype) == y)
    return cmp.type(y.dtype).sum()

def evaluate_accuracy(net,data_iter,devices):
    net.eval()
    metric = add_machine(2)
    for inputs,labels in data_iter:
        inputs = [i.to(devices) for i in inputs]
        labels = labels.to(devices)
        y_hat = net(inputs)
        metric.add(accrucy(y_hat,labels),labels.numel())
    return metric[0]/metric[1]


def train_ch13(net, train_iter, test_iter, lr, num_epochs,devices):
    # 在此时微调时利用不同的学习率进行微调的效果并没有使用相同的学习率的效果好，(损失降不下去)
    # 原因可能是因为使用的梯度下降的方法是adam，其对学习率的变化并不敏感(一般固定于1e-3)，反而增大了输出层的学习率会出现问题
    pre_param = [param for name, param in net.named_parameters() if 'fc' not in name]
    trainer = torch.optim.Adam([{'params': pre_param}, {'params': net.fc.parameters(), 'lr': lr * 10}])
    # trainer = torch.optim.Adam(net.parameters(),lr=lr)
    loss = nn.CrossEntropyLoss()
    net.to(devices)
    for epoch in range(num_epochs):
        metric = add_machine(3)
        net.train()
        for inputs,labels in train_iter:
            inputs= [i.to(devices) for i in inputs]
            labels = labels.to(devices)
            y_hat = net(inputs)
            l = loss(y_hat,labels)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            metric.add(l*labels.numel(),accrucy(y_hat,labels),labels.numel())
        test_acc = evaluate_accuracy(net,test_iter,devices)
        print('当前迭代次数:',epoch + 1)
        print('loss:',metric[0]/metric[2])
        print('train_acc:',metric[1]/metric[2])
        print('test_acc:',test_acc)


devices = d2l.try_gpu()
model_data_dir = 'D:\\python\\pytorch作业\\序列模型\BERT\\BERT预训练模型的应用\\data\\bert.small.torch'
bert,vocab = load_pretrained_model(model_data_dir, num_hiddens=256, ffn_num_hiddens=512, num_heads=4
                      ,num_layers=2, dropout=0.1, max_len=512, devices=devices)
text_data_dir = 'D:\\python\\pytorch作业\\序列模型\\BERT\data\\snli_1.0\\snli_1.0'
train_dataset = read_snli(text_data_dir, is_train=True)
test_dataset = read_snli(text_data_dir,is_train=False)
#train_set = SNLIBERTDataset(train_dataset,128,vocab)
test_set = SNLIBERTDataset(test_dataset,128,vocab)
'''
train_iter = data.DataLoader(train_set,batch_size=128,shuffle=True)
test_iter = data.DataLoader(test_set,batch_size=128,shuffle=False)
net = BERTClassifier(bert)
lr, num_epochs = 1e-4, 5
train_ch13(net, train_iter, test_iter, lr,num_epochs, devices)
'''


'''
trainer = torch.optim.Adam(net.parameters(), lr=lr)
loss = nn.CrossEntropyLoss(reduction='none')
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs,
    devices)
'''
