import torch
from torch import nn
from torch.utils import data
def synthetic_data(w,b,sample_num=1000):  #这一步是为生成随机的特征以及与之对应的标签
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,[sample_num,1])  # [sample_num,1],这一层只有一个神经元
    return X,Y
w=torch.tensor([[2],[3.4]])
b=torch.tensor([[-5]])
features,labels=synthetic_data(w,b)

#利用模板生成小批量的数据
def mini_batch(data_array,batch_size=10,is_train=True):
    dataset=data.TensorDataset(*data_array)   # dataset=NEWS.TensorDataset(data_array[0],data_array[1])
    # TensorDataset的作用生成一个元组，元组的长度为其中一个的shape[0]，元组a索引中的内容为一个元组，包含data_array[0]该索引下的tensor，和data_array[1]索引下的tensor
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
items= mini_batch([features,labels],batch_size=10,is_train=True)
#print(next(iter(items)))

#构建一个网络
net=nn.Sequential(nn.Linear(2,1))
net[0].weight.data.normal_(0,0.01) #初始化训练参数
net[0].bias.data.fill_(0)
loss=nn.MSELoss()
trainer=torch.optim.SGD(net.parameters(),lr=0.03)
epcho_num=3
for i in range(epcho_num):
    for X,Y in items:
        l=loss(net(X),Y)
        trainer.zero_grad()  # 反向传播步骤一
        l.backward() # 反向传播步骤二
        trainer.step() #参数更新
    with torch.no_grad():
        epcho_loss=loss(net(features),labels)
        print(f'第{i+1}次迭代的损失:{epcho_loss:1.6f}')
print(w)