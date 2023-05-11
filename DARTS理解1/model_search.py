# 构建整个搜索空间
import torch
from torch import nn
from operations import OPS
import operations
from torch.nn import functional as F
from collections import namedtuple
PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

class MixedLayer(nn.Module): # 针对于上一个特征图通过 所有的搜索架构以及对应的不同权重得到 下一个特征图
    def __init__(self,c,stride):
        super(MixedLayer, self).__init__()
        self.op = nn.ModuleList()
        for i in PRIMITIVES:
            layer = OPS[i](c, stride, affine=False)  # 此时输入的通道数是等于输出的通道数的
            if 'pool' in i:
                self.op.append(nn.Sequential(layer,nn.BatchNorm2d(c,affine=False)))
            else:
                self.op.append(layer)

    def forward(self,x,weights): # 此时的weight为一个一维的矩阵
        out = [layer(x) * weight for layer, weight in zip(self.op,weights)]
        return sum(out)

# 构建一个cell模块(已进行验证，前向传播是正确的)
class Cell(nn.Module):
    def __init__(self, steps, multiplier, cpp, cp, c, reduction, reduction_prev):
        super(Cell, self).__init__()
        # steps代表在这个cell中有几个特征图(节点),multiplier的值应当和steps的值相同，用于最后求解cell的输出
        # reduction代表这个模块的宽高是否需要减半
        # cpp是操作中的输入通道数(process0的输入通道)，cp代表输入的通道数(process1的输入通道)，c代表的是输出的通道数
        self.reduction = reduction # 当前的cell中是否需要进行宽高减半的标志
        self.reduction_prev = reduction_prev
        self.steps = steps
        if reduction_prev: # 当reduction_prev=True说明此时的宽高处理需要减半
            self.process0 = operations.FactorizedReduce(cpp, c, affine=False)
        else:
            self.process0 = operations.ReLUConvBN(cpp, c, 1, 1, 0, affine=False)
        self.process1 = operations.ReLUConvBN(cp, c, 1, 1, 0, affine=False)
        self.layers = nn.ModuleList()
        for step in range(steps):
            for j in range(step + 2):
                stride = 2 if self.reduction and j < 2 else 1
                self.layers.append(MixedLayer(c,stride))
        self.multiplier = multiplier

    def forward(self, x0, x1,weights):
        # 此时输入为[batch,48,32,32]且stride=1的情况下
        # 对于每个cell来说一共有14中特征图之间的连接，每个连接有8中可能的架构,因此weights.shape=[14,8]
        s0 = self.process0(x0) # [batch,16,32,32]
        s1 = self.process1(x1) # [batch,16,32,32]
        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum([self.layers[offset + j](state,weights[offset + j]) for j,state in enumerate(states)])
            offset += len(states)
            states.append(s)
        # 此时states为一个长度为 2 + steps ，形状为[batch,16,32,32]的列表
        return torch.cat(states[-self.multiplier:],dim=1) # [batch,64,32,32]

# 搭建整个super_net模型
class Network(nn.Module):
    def __init__(self,c, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        # steps=4, multiplier=4代表是在一个cell中除去输入的两个节点，包含step个节点
        # layer代表的这个网络中一共有多少个cell
        super(Network, self).__init__()
        self.layers = layers
        self.criterion = criterion
        self.c = c
        self.num_classes = num_classes
        self.steps = steps
        self.multiplier = multiplier

        # super_net的输入层
        c_curr = stem_multiplier * c
        self.stem = nn.Sequential(nn.Conv2d(3, c_curr,kernel_size=3,stride=1,padding=1,bias=False),
                                  nn.BatchNorm2d(c_curr))

        # 中间的cells层
        self.cells = nn.ModuleList()
        cpp, cp, c_curr = c_curr, c_curr, c
        reduction_prev = False
        for layer in range(layers):
            if layer in [layers // 3, 2 * layers // 3]:
                c_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(steps, multiplier, cpp, cp, c_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells.append(cell)
            cpp, cp = cp, multiplier * c_curr

        # 输出层
        self.global_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(cp, num_classes)

        # 定义cell中每个架构的权重参数
        k = sum([1 for i in range(steps) for _ in range(i + 2)])
        num_ops = len(PRIMITIVES)
        self.alpha_normal = nn.Parameter(torch.randn(size=(k,num_ops))) # 所有宽高不减半的cell共享权重
        self.alpha_reduce = nn.Parameter(torch.randn(size=(k,num_ops))) # 所有宽高减半的cell共享权重
        with torch.no_grad():
            self.alpha_normal.mul_(1e-3)
            self.alpha_reduce.mul_(1e-3)
        # 此时利用的是列表，因此self._arch_parameters并不在network.parameters()中
        self._arch_parameters = [self.alpha_normal,self.alpha_reduce]

    def forward(self,x):
        # 输入层
        s0 = s1 = self.stem(x)
        # 中间的cell层
        for i,cell in enumerate(self.cells):
            if cell.reduction is True:
                weights = F.softmax(self.alpha_reduce,dim=-1)
            else:
                weights = F.softmax(self.alpha_normal,dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        # 输出层
        out = self.global_pooling(s1)
        out = out.reshape(out.shape[0],-1)
        out = self.classifier(out)
        return out

    def loss(self,x,y):
        logits = self(x)
        return self.criterion(logits,y)

    def new(self):
        model_new = Network(self.c, self.num_classes, self.layers, self.criterion).cuda()
        # 将self.arch_parameters()中每个元素复制给对应索引的model_new.arch_parameters()中去
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def arch_parameters(self):
        return self._arch_parameters

    def genotype(self):

        def _parse(weights): # 用于获得在一个cell中 当前node 输入的路径的序列， 连接的操作是什么
            # weight = [14,8]
            gene = []
            start = 0
            n = 2
            for i in range(self.steps):
                end = start + n
                W = weights[start:end].copy()
                # 当前node 输入的路径的序列
                edge = sorted(range(i + 2),
                              key=lambda x: -max([W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')]))[:2]

                # 遍历当前node与相连的两个node，选择最大概率
                for j in edge:
                    k_best = None
                    for idx in range(len(W[j])):
                        if idx != PRIMITIVES.index('none'):
                            if k_best is None or W[j][idx] > W[j][k_best]:
                                k_best = idx
                    gene.append((PRIMITIVES[k_best],j))
                start = end
                n += 1
            return gene
        # 长度为4,第1个元素代表node1和输入的第j条路径相连，选择的操作为gene_normal[0][0]名称的操作
        gene_normal = _parse(F.softmax(self.alpha_normal,dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alpha_reduce,dim=-1).data.cpu().numpy())
        concat = range(2, 2 + self.steps)
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        return genotype

    # def genotype(self):
    #     def _parse(weights):
    #         gene = []
    #         n = 2
    #         start = 0
    #         for i in range(self.steps): # for each node
    #             end = start + n
    #             W = weights[start:end].copy() # [2, 8], [3, 8], ...
    #             edges = sorted(range(i + 2), # i+2 is the number of connection for node i
    #                         key=lambda x: -max(W[x][k] # by descending order
    #                                            for k in range(len(W[x])) # get strongest ops
    #                                            if k != PRIMITIVES.index('none')) # 'none'即是代表两个节点不连接的概率，若其最大说明两个节点不连接
    #                            )[:2] # only has two inputs # [:2]代表保留两个节点之间拥有最大概率的两个操作
    #             for j in edges: # for every input nodes j of current node i
    #                 k_best = None
    #                 for k in range(len(W[j])): # get strongest ops for current input j->i, 遍历当前两个节点的操作
    #                     if k != PRIMITIVES.index('none'):
    #                         if k_best is None or W[j][k] > W[j][k_best]:
    #                             k_best = k
    #                 gene.append((PRIMITIVES[k_best], j)) # save ops and input node 保存两个节点之间最大可能的操作，是当前节点的那一条连接操作
    #             start = end
    #             n += 1
    #         return gene
    #
    #     gene_normal = _parse(F.softmax(self.alpha_normal, dim=-1).data.cpu().numpy())
    #     gene_reduce = _parse(F.softmax(self.alpha_reduce, dim=-1).data.cpu().numpy())
    #
    #     concat = range(2 + self.steps - self.multiplier, self.steps + 2) # [2,6) 其用于表示当前节点的序号
    #     genotype = Genotype(
    #         normal=gene_normal, normal_concat=concat,
    #         reduce=gene_reduce, reduce_concat=concat
    #     )
    #
    #     return genotype


Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')



if __name__ == '__main__':
    torch.manual_seed(1)
    device = torch.device('cuda')
    loss = nn.CrossEntropyLoss()
    net = Network(c=16, num_classes=10, layers=8, criterion=loss).to(device)
    x = torch.randn(size=(1,3,32,32),device=device)
    print(net(x))
    print(sum([i.numel() for i in net.parameters()]))
