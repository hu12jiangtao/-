import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.alpha = nn.Parameter(torch.ones(size=(4,),dtype=torch.float32))
        with torch.no_grad():
            self.alpha.mul_(0.1)
        self.op = nn.Linear(5,4,bias=False)
        self.lst = [self.alpha]

    def forward(self,x):
        x = self.op(x) # [batch,4]
        return x * self.alpha

# torch.manual_seed(1)
# net = Net()
# print(net.lst)
# x = torch.randn(size=(1,5))
# y = torch.tensor([2],dtype=torch.long)
# loss = nn.CrossEntropyLoss()
# opt = torch.optim.Adam(net.parameters(),lr=0.001)
# l = loss(net(x),y)
# opt.zero_grad()
# l.backward()
# opt.step()
# print(net.lst)


# model = nn.Sequential(nn.Linear(5,4),nn.Linear(4,3))
# optimizer = torch.optim.SGD(model.parameters(), 0.1, momentum=0.9, weight_decay=1e-4)
# for v in model.parameters():
#     print(optimizer.state[v]['momentum_buffer'])

# a = [torch.tensor([[1,2],[3,4]]),torch.tensor([[5,6],[7,8]])]
# b = [torch.tensor([[9,10],[11,12]]),torch.tensor([[13,14],[15,16]])]
# for i,j in zip(a,b):
#     i.data.copy_(j.data)
#
# print(a)

# net = nn.Linear(4,3)
# opt = torch.optim.SGD(net.parameters(),lr=0.1)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,float(5),eta_min=0.001)
# for i in range(5):
#     lr_scheduler.step()
#     print(opt.param_groups[0]['lr'])

# net = nn.Linear(4,3)
# k = net.state_dict()
# print(k)
# param = {'weight':torch.ones(size=(3,4)),
#          'bias':torch.ones(size=(3,))}
# k.update(param)
# print(net.state_dict())
# print(k)

# from collections import namedtuple
# Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
# concat = range(2,4)
# gene_normal = [(nn.Linear(4,3),2),(nn.Linear(4,3),6),(nn.Linear(4,3),5),(nn.Linear(4,3),3)]
# gene_reduce = [(nn.Linear(4,3),2),(nn.Linear(4,3),6),(nn.Linear(4,3),5),(nn.Linear(4,3),3)]
# genotype = Genotype(normal=gene_normal, normal_concat=concat,
#                      reduce=gene_reduce, reduce_concat=concat)
# print(genotype)
a = torch.tensor([[1,2,3],[4,5,6],[4,2,7]])
b,c = torch.topk(a,2,1)
print(c)