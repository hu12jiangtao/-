import torch
import random

from d2l import torch as d2l  #利用这句话安装pip install d2l --no-dependencies


#在吴恩达的版本中X（shape=（feature，num））样本的定义为一列是一个小样本（列数就是特征数量），w的shape为（连接的后一个神经元数，连接的前一个神经元数）
#b的shape为(后一个神经元的个数，1)，此时a=w*X+b，a的shape为[后一个神经元的个数，num]

#而在torch中使用的是X(shape=(num,feature))，每一行定义为一个小样本（行数就是特征数量），w的shape为（连接的前一个神经元数，连接的后一个神经元数）
#b的shape为(1,后一个神经元的个数)，此时a=X*w+b，a的shape为[num，后一个神经元的个数]

# print(*train_iters)中的*可以对yield创造的dataset进行解码或者对DataLoader创造dataset的进行解码

def synthetic_data(w,b,sample_num):  #这一步是为生成随机的特征以及与之对应的标签
    X=torch.normal(0,1,[sample_num,w.shape[0]])
    Y=torch.matmul(X,w)+b
    Y+=torch.normal(0,0.01,[sample_num,1])  # [sample_num,1],这一层只有一个神经元
    return X,Y


w=torch.tensor([[2],[-3.4]])
b=4.2
sample_num=1000
features,labels=synthetic_data(w,b,sample_num)  # features,labels都是tensor类型的

print(features.shape)
print(labels.shape)
d2l.set_figsize()  #创建画框后
d2l.plt.scatter(features[:,1].numpy(),labels.numpy(),1)     # 画点函数中应当是一个numpy类型的的列表,1是用来控制点的大小的
#d2l.plt.show()
def data_iter(batch_size,feature,lables):
    example_num=feature.shape[0]  #样本的个数
    index=list(range(example_num))
    random.seed(1)
    random.shuffle(index)
    for i in range(0,example_num,batch_size):
        batch_index=index[i:min(i+batch_size,example_num)]
        yield feature[batch_index,:],lables[batch_index,:]   # yield必须使用在子函数中的
batch_size=10

#定义初始化参数
w=torch.normal(0,0.01,[2,1],requires_grad=True)
b=torch.zeros([1,1],requires_grad=True)

def linear(w,b,X):
    return torch.matmul(X,w)+b
def squred_loss(y_head,y):
    cost=0.5*(y-y_head)**2
    return cost
def sgd(params,lr,batch_size):  #参数更新部分
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

lr=0.03
num_epochs=3
net=linear
loss=squred_loss

for i in range(num_epochs):
    for X,Y in data_iter(batch_size,features,labels):   #遍历了所有的样本
        l=loss(net(w,b,X),Y)
        l.sum().backward()   #相当于计算了w,b的梯度，可以通过w.grad,b.grad进行查看
        sgd([w,b],lr,batch_size)
    with torch.no_grad():   #
        train_1=loss(net(w,b,features),labels)
        print(f'epcho {i+1}  ,  loss {float(train_1.mean()):1.6f}')

print(w)




d2l.Animator

