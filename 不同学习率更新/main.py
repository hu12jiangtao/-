# 此时利用不同的学习率对网络的不同层进行更新
import torch
from torch import nn

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.linear1 = nn.Linear(5,4)
        self.linear2 = nn.Linear(4,3)
        self.linear3 = nn.Linear(3,2)

    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x

net = Net1()

for module in net.children():
    print(module._parameters['weight'])
detach_param_id = list(map(id,net.linear1.parameters()))
update_param = filter(lambda x:id(x) not in detach_param_id, net.parameters())
opt = torch.optim.Adam([{'params':update_param,'lr':0.001},{'params':net.linear1.parameters(),'lr':0}])
x = torch.randn(size=(2,5))
y = torch.tensor([1,0])
y_hat = net(x)
loss = nn.CrossEntropyLoss()
l = loss(y_hat,y)
opt.zero_grad()
l.backward()
opt.step()
print('*' * 50)
for module in net.children():
    print(module._parameters['weight'])
