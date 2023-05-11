import torch
from torch import nn
from torch.nn import functional as F
torch.random.manual_seed(1)   #可以将随机初始给固定住
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(10,5)
        self.linear2=nn.Linear(5,1)
    def forward(self,X):
        return self.linear2(F.relu(self.linear1(X)))
def xavier(m):
    if type(m)==nn.Linear:
        nn.init.xavier_uniform_(m.weight)

net=MyModel()
net.apply(xavier)
print(net)
print(net.state_dict())
torch.save(net.state_dict(),'mpl.params')


