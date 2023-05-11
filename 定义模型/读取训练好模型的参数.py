import torch
from torch import nn
from torch.nn import functional as F
#此时load数据时，必须将模型重新进行定义
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=nn.Linear(10,5)
        self.linear2=nn.Linear(5,1)
    def forward(self,X):
        return self.linear2(F.relu(self.linear1(X)))

clone=MyModel()
clone.load_state_dict(torch.load('mpl.params'))
clone.eval()  #eval和trainer之间并没有什么区别，只有在batch_norm或者drop_out上的测试时需要利用eval(),训练时trainer()
print(clone.state_dict())

