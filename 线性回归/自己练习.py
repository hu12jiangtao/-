import torch
import random
#生成数据
def create_data():
    w=torch.tensor([[2],[3.4]])
    b=torch.tensor([[5]])
    sample_num=1000
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,Y.shape)
    return X,Y
features,labels=create_data()
print(features.shape)
print(labels.shape)

#每一次随机样本的抽取
def choice_sample(features,labels,batch_size=10):
    sample_num=features.shape[0]
    index=list(range(sample_num))
    random.shuffle(index)
    for i in range(0,sample_num,batch_size):
        batch_index=index[i:min(sample_num,i+batch_size)]
        yield features[batch_index],labels[batch_index]

#前向传播
def forward(w,b,X):
    return torch.matmul(X,w)+b
#损失函数
def compute_loss(y,y_hat):
    cost=torch.mean(0.5*(y-y_hat)**2)
    return cost
#参数的更新
def update_parameters(parameters,lr=0.03):
    for param in parameters:
        with torch.no_grad():
            param-=lr*param.grad
            param.grad.zero_()  #当一次梯度更新后应当把在梯度中存储的值给清零

#整一个线性回归
w=torch.normal(0,1,[2,1],requires_grad=True)
b=torch.zeros([1,1],requires_grad=True)
epcho_num=3
for i in range(epcho_num):
    for X,Y in choice_sample(features,labels,batch_size=10):
        l=compute_loss(Y,forward(w,b,X))
        l.backward()
        update_parameters([w,b], lr=0.03)
    with torch.no_grad():
        epcho_loss=compute_loss(labels,forward(w,b,features))
        print(f'第{i+1}次迭代的损失:{epcho_loss:1.6f}')
print('预测的w为:')
print(w)
print('真实的w为:')
print(torch.tensor([[2],[3.4]]).numpy())

