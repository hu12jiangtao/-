from torch import nn
import torch

class Encode(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,X):
        raise NotImplementedError

class Decode(nn.Module):
    def __init__(self):
        super().__init__()
    def init_state(self,src_outputs):
        raise NotImplementedError
    def forward(self,X,state):
        raise NotImplementedError

class Seq2SeqEncode(Encode):
    def __init__(self,num_vocab,embed_size,num_layers,num_hiddens,dropout=0):
        super().__init__()
        self.embedding=nn.Embedding(num_vocab,embed_size) # 第一个参数为有多少个词元进行映射
        self.GRU=nn.GRU(embed_size,num_hiddens,num_layers,dropout=dropout)
    def forward(self,X): # shape=[batch_size,seq]
        X=self.embedding(X).permute(1,0,2) # shape=[seq,batch_size,embed_size]
        Y,state=self.GRU(X)
        return Y,state

class Seq2SeqDecode(Decode):
    def __init__(self,vocab_size,embed_size,num_layers,num_hiddens,dropout=0):
        super().__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.GRU=nn.GRU(num_hiddens+embed_size,num_hiddens,num_layers,dropout=dropout)
        self.dense=nn.Linear(num_hiddens,vocab_size)
    def init_state(self,src_outputs):
        return src_outputs[1] # 取出encode的状态输出
    def forward(self,X,state):
        X=self.embedding(X).permute(1,0,2) # shape=[seq,batch_size,embed_size]
        context=state[-1].repeat(X.shape[0],1,1)  # shape=[seq,batch_size,num_hiddens]
        X_to_context=torch.cat((X,context),dim=2)
        Y,state=self.GRU(X_to_context,state) # state=[num_layers,batch_size,num_hiddens], Y=[seq,batch_size,num_hiddens]
        Y=self.dense(Y).permute(1,0,2) # Y=[batch_size,seq,vocab_size]
        return Y,state

encode = Seq2SeqEncode(10, embed_size=8, num_hiddens=16,
                         num_layers=2)
encode.eval()
X = torch.zeros((4, 7), dtype=torch.long)
decode = Seq2SeqDecode(10, embed_size=8, num_hiddens=16,
                         num_layers=2)
decode.eval()
state = decode.init_state(encode(X))
output, state = decode(X, state)
print(output.shape, state.shape)

#创建整个网络
class EncodeDecode(nn.Module):
    def __init__(self,encode,decode):
        super().__init__()
        self.encode=encode
        self.decode=decode
    def forward(self,source_X,target_X):
        source_outputs=self.encode(source_X)
        target_state=self.decode.init_state(source_outputs)
        return self.decode(target_X,target_state)

net=EncodeDecode(encode,decode)
print(net(X,X)[0].shape,net(X,X)[1].shape)

# 设置网络的损失函数
def sequence_mask(X, valid_len, value=0):
    maxlen=X.shape[1]
    mask=torch.arange(maxlen,dtype=torch.float32,device=X.device)
    mask=(mask[None,:]<valid_len[:,None])
    X[~mask]=value
    return X

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    def forward(self,tgt_pred,labels,num_valid):
        weight=torch.ones_like(labels)
        weight=sequence_mask(weight, num_valid)
        self.reduction='none'
        no_weight_loss=super().forward(tgt_pred.permute(0,2,1),labels)
        weight_loss=(no_weight_loss*weight).mean(1)
        return weight_loss

loss = MaskedSoftmaxCELoss()
l=loss(torch.ones(3, 4, 10), torch.ones((3, 4), dtype=torch.long),
     torch.tensor([4, 2, 0]))
print(l)
