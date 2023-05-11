from torch import nn
import torch
from mingpt.utils import CfgNode as CN
import math
from torch.nn import functional as F
# GPT-2除了具有更多的变压器层和参数外，仅进行了一些架构修改:
# 1.并且在最后一个decode块之后添加了额外的层归一化。使用了一个修改后的初始化
# 2.残留层的权重最初按1 / sqrt {N}的比例进行缩放，其中N是残留层的数量。
# 3.层归一化被移动到每个子块的输入
def mask_prepare(x,valid_nums,fill=0.): # x为二维的矩阵,valid_nums为一维的向量
    a = x.shape[1]
    part = torch.arange(a,device=x.device)
    cmp = valid_nums[:,None] > part[None,:]
    x[~cmp] = fill
    return x

def mask_softmax(x,valid_nums): # x为三维的矩阵,valid_nums为一维或者二维的向量
    if valid_nums is None:
        return F.softmax(x,dim=-1)
    else:
        shape = x.shape
        if valid_nums.dim() == 1:
            valid_nums = torch.repeat_interleave(valid_nums,dim=0,repeats=x.shape[1])
        else:
            valid_nums = valid_nums.reshape(-1)
        out = mask_prepare(x.reshape(-1,x.shape[-1]),valid_nums,fill = float('-inf'))
        return F.softmax(out.reshape(shape),dim=-1)

class DotProductAttention(nn.Module):
    def __init__(self,dropout):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,valid_num):
        query_size = q.shape[-1]
        weight = torch.bmm(q,k.permute(0,2,1)) / math.sqrt(query_size)
        mask_weight = mask_softmax(weight,valid_num)
        return torch.bmm(self.dropout(mask_weight),v)

class CausalSelfAttention(nn.Module): # 此时其为带有掩码的自注意力机制
    def __init__(self,config):
        super(CausalSelfAttention, self).__init__()
        # 此时的config.n_embd相当于这个transformer解码器模块输入、输出的维度
        assert config.n_embd % config.n_head == 0
        self.w_o = nn.Linear(config.n_embd,config.n_embd)
        self.w_q = nn.Linear(config.n_embd, config.n_embd)
        self.w_k = nn.Linear(config.n_embd,config.n_embd)
        self.w_v = nn.Linear(config.n_embd,config.n_embd)
        self.n_head = config.n_head
        self.attention = DotProductAttention(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop) # 在案例中最后对out进行这个dropout

    def transpose_qkv(self,x,n_head):
        x = x.reshape(x.shape[0],x.shape[1],n_head,-1)
        x = x.permute(0,2,1,3)
        x = x.reshape(-1,x.shape[2],x.shape[3])
        return x

    def transpose_out(self,x,n_head):
        x = x.reshape(-1,n_head,x.shape[1],x.shape[2])
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0],x.shape[1],-1)
        return x

    def forward(self,x, valid_nums=None): # 此时输入x = [batch,seq,n_embd],其中seq为pad后的统一长度
        q = self.transpose_qkv(self.w_q(x),self.n_head)
        k = self.transpose_qkv(self.w_k(x),self.n_head)
        v = self.transpose_qkv(self.w_v(x),self.n_head)
        if valid_nums is not None:
            valid_nums = torch.repeat_interleave(valid_nums,dim=0,repeats=self.n_head)
        out = self.attention(q,k,v,valid_nums)
        out = self.transpose_out(out,self.n_head) # [batch,seq,n_embd]
        return self.w_o(out)

class PositionWiseFFN(nn.Module):
    def __init__(self,hidden_num,mid_hidden_num):
        super(PositionWiseFFN, self).__init__()
        self.linear1 = nn.Linear(hidden_num,mid_hidden_num)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(mid_hidden_num,hidden_num)

    def forward(self,x):
        return self.linear2(self.act(self.linear1(x)))

class Block(nn.Module): # 为一个transformer的decode模块
    # 其中一个模块由带有mask的多头注意力层、FFN层、2个layer norm层构成的
    def __init__(self,config):
        super(Block, self).__init__()
        self.FFN = PositionWiseFFN(config.n_embd,config.n_embd * 4)
        self.attn = CausalSelfAttention(config)
        # 在gpt2中层归一化被移到了输入上
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.dropout1 = nn.Dropout(config.resid_pdrop)
        self.dropout2 = nn.Dropout(config.resid_pdrop)

    def forward(self,x):
        # 带有掩码的注意力机制(此时对于第一个q来说应该只看到第一个k,即第i个q看到前i个k)
        # 此时对于x的第i个batch的X_i(shape=[seq,num_hidden])来说,valid_num应为torch.arange(X_i.shape[1])
        # 因此对于x的valid_num为torch.arange(1, x.shape[1] + 1).repeat((x.shape[0],1))
        # 原因s = torch.tensor([[[1,2],[3,4]],[[1,2],[3,4]]]).reshape(-1,s.shape[-1])=torch.tensor([[1,2],[3,4],[1,2],[3,4]])
        dec_valid_nums = torch.arange(1, x.shape[1] + 1,device=x.device).repeat((x.shape[0],1))
        # 带有mask的注意力机制
        x = x + self.dropout1(self.attn(self.ln_1(x),dec_valid_nums))
        x = x + self.dropout2(self.FFN(self.ln_2(x)))
        return x

class GPT(nn.Module):
    @ staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd = None # n_embd即是每个模块输入、输出的长度(相当于num_hiddens)
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self,config):
        super(GPT, self).__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        type_given = config.model_type is not None # 给定GPT的规模
        params_give = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        if type_given:
            config.merge_from_dict(
                {
                    'openai-gpt': dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                    'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    'gpt-mini': dict(n_layer=6, n_head=6, n_embd=192),
                }[config.model_type]
            )
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # GPT的输出层
        self.apply(self._init_weights)
        for pn, p in self.named_parameters(): # 在GPT2上面
            if pn.endswith('FFN.linear2.weight') or pn.endswith('attn.w_o.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        n_params = sum([i.numel() for i in self.transformer.parameters()])
        print("number of parameters: %.2fM" % (n_params / 1e6,))


    def _init_weights(self,m):
        if isinstance(m,nn.Linear):
            nn.init.normal_(m.weight,mean=0,std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m,nn.Embedding):
            nn.init.normal_(m.weight,mean=0,std=0.02)
        elif isinstance(m,nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self,idx,target=None):
        # 输出的idx.shape=[batch,seq],其中的target就是idx向后推一个序列的序列，就是train_dataset中的y=[batch,seq]
        # 当target=None的情况则是模型进行生成的时候，就不要进行损失的计算了
        device = idx.device
        batch,seq = idx.shape
        position = torch.arange(seq,dtype=torch.long,device=device).unsqueeze(0) # [1,seq]
        tok_embed = self.transformer.wte(idx) # [batch,seq,n_embd]
        pos_embed = self.transformer.wpe(position) # [1,seq,n_embd]
        x = self.transformer.drop(tok_embed + pos_embed) # [batch,seq,n_embd]
        for blk in self.transformer.h:
            x = blk(x)
        x = self.transformer.ln_f(x) # 在gpt2中最后一个decode块之后添加了额外的层归一化
        logits = self.lm_head(x) # [batch,seq,vocab_size]
        # 计算语言模型的损失
        loss = None
        if target is not None:
            loss = F.cross_entropy(logits.permute(0,2,1),target,ignore_index=-1) # 当label等于-1时这个样本不计入loss的计算
        return logits, loss

    def configure_optimizers(self,trainer_config):
        whitelist_weight_modules = (nn.Linear,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        decay = set()
        no_decay = set() #
        for mn,m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = f'{mn}.{pn}' if mn else pn # 对于最大的nn.Module类来说没有名字
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(mn,whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(mn,blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {n: p for n, p in self.named_parameters()}
        inter_params = no_decay & decay
        union_params = no_decay | decay
        print(len(union_params))
        print(len(param_dict.keys()))
        print(len(union_params - param_dict.keys()))
        print(param_dict.keys())
        print(union_params)
        assert len(inter_params) == 0 and len(union_params - param_dict.keys()) == 0 # 确保进行衰减的参数与不进行衰减的参数和为网络所有的参数并且没有交集
        optim_groups = [
            {'params':[param_dict[param_name] for param_name in sorted(list(decay))],'weight_decay':trainer_config.weight_decay},
            {'params': [param_dict[param_name] for param_name in sorted(list(no_decay))],"weight_decay": 0.0}
            ]
        optimizer = torch.optim.AdamW(optim_groups,lr=trainer_config.learning_rate,betas=trainer_config.betas)
        return optimizer

    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        # 此时的目标为每次输入一个序列长度小于等于self.block_size的1-t序列,得到预测的2-t+1的序列，取出第t+1个序列预测的值作为生成的一个序列,之后以此类推
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.block_size else idx[:,-self.block_size:]
            logits,_ = self(idx_cond) # [batch,seq,vocab_size],其中seq最长为block_size
            logits = logits[:,-1,:] / temperature
            if top_k is not None:
                value,_ = torch.topk(logits,top_k) # value=[batch,topk]
                logits[logits < value[:,-1].reshape(-1,1)] = -float('inf')
            probs = F.softmax(logits,dim=-1)
            if do_sample:
                choice_index = torch.multinomial(probs,num_samples=1)
            else:
                _,choice_index = torch.topk(probs,1)
            idx = torch.cat([idx,choice_index],dim=1)
        return idx


















if __name__ == '__main__':
    net = nn.Sequential(nn.Linear(10,7),nn.Sequential(nn.Linear(7,4),nn.Linear(4,3)),nn.Linear(3,1))
    part1 = set()
    part2 = set()
    for name,param in net.named_parameters():
        part1.add(name)

    # 在原先的configure_optimizers中使用的是net.named_modules
    for name,module in net.named_modules():
        # 列举出所有属于nn对象的名称，
        # 在此包含了nn.Sequential(nn.Linear(10,7),nn.Sequential(nn.Linear(7,4),nn.Linear(4,3)),nn.Linear(3,1)),但是没有名称
        # 包含了nn.Linear(10,7)、nn.Sequential(nn.Linear(7,4),nn.Linear(4,3))、nn.Linear(7,4)、nn.Linear(4,3)、nn.Linear(3,1)
        for pn,p in module.named_parameters():
            fpn = f'{name}.{pn}' if name else pn
            part2.add(fpn)

    print(part1)
    print(part2)

    a = set()
    a.add('a')
    a.add('b')
    a.add('c')

    b = set()
    b.add('a')
    b.add('c')
    print(a | b)
    print(a & b)
    print(a - b)

    logits = torch.tensor([[1,4,5],[2,1,6]],dtype=torch.float32)
    v,_ = torch.topk(logits,2)
    print(v[:, [-1]])
    print(v[:, -1])
    logits[logits < v[:, [-1]]] = -float('Inf')
    print(logits)






