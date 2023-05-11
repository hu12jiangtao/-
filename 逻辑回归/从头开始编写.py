
import torchvision
import torch
from torch.utils import data
from d2l import torch as d2l
from torchvision import transforms
import time
d2l.use_svg_display()

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
def get_dataloader_workers(num):
    return num


def load_data(batch_size):
    trans = transforms.ToTensor()
    mninst_train = torchvision.datasets.FashionMNIST(root='../NEWS', train=True, transform=trans,
                                                     download=True)  # train=True说明其参与训练
    mninst_test = torchvision.datasets.FashionMNIST(root='../NEWS', train=False, transform=trans,
                                                    download=True)  # transform=trans代表将图片转换为了tensor数据类型
    #print(mninst_train[0][0].shape)  # mninst_train[0][0]中包含了第一个样本的图片，(mninst_train[0][1]第一张图片的标签
    # 此时相当于已经进行过data.TensorDataset()了，其中的第一个量为[num,1,28,28]，另一个为标签列表[num],利用data.TensorDataset打包后该元组中有num个元素，每个元素为一个（图片的三维矩阵，label）的元组
    iters_train = data.DataLoader(mninst_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(4))
    iters_test = data.DataLoader(mninst_test, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(4))
    return iters_train,iters_test

iters_train,iters_test=load_data(batch_size=256)  # 每一张图片的尺寸为1*28*28 ，此时每一个小样本的label应当是一个向量（长度为256）,此时其中还是一个四维的矩阵
print(next(iter(iters_train))[0].shape)
#初始化权重（逻辑回归中每一个像素为一个特征，拉升成一条直线）
feature_input=784
label_num=10
batch_size=256
w=torch.normal(0,1,size=[784,10],requires_grad=True)
b=torch.zeros([1,label_num],requires_grad=True)  # b的形状应当是[1,连接的后一层神经元的个数]

def softmax(x):  #此时输出为num*10
    e_x=torch.exp(x)
    partition=torch.sum(e_x,dim=1,keepdim=True)
    return e_x/partition

def net(w,b,X):
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w)+b)

def loss(y,y_hat):  #此时取出的label应当为一个向量（利用交叉熵损失函数）
    #在python中我们可以得到a = torch.tensor([[1, 2, 3], [4, 5, 6]]),此时a[[0,1],[0,2]]代表的意思就是将a[0,0],a[1,2]取出来构成了一个行向量
    #而在交叉熵的求取过程中y_hat[range(y_hat.shape[0]),y]的意思就是取出每个样本lable对应的预测值，此时我们返回的是一个行向量
    return -torch.log(y_hat[range(y_hat.shape[0]),y])

def sgd(params,batch_size,lr):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

def updater(batch_size):
    return sgd([w,b],batch_size,lr=0.1)

def accuracy(y,y_hat):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)  #此时应该是一个行向量
    cmp=(y_hat.type(y.dtype)==y)  #得出一个值的类型为布尔的行向量（确保y,y_hat不会因为值的类型的不同而全部变为false）
    return float(cmp.type(y.dtype).sum())  #此时bool值转换为int或者float型的值都是True=1,Flase=0,y.dtype指的是y中元素的数据类型,此时返回的是tensor的值


class Accumulator():
    def __init__(self,n):
        self.data=[0.0]*n  # 此时的意思是变为了[0.0,0.0]
    def add(self,*args):   #此时定义了一个加法，*args代表着不知道传入参数的个数,但是传入的参数是以元组的形式传入的，args[0]就是acuuracy(y,forward_propogration(w,b,X))的值
        self.data=[a+float(b) for a,b in zip(self.data,args)]   #和TensorDataset指令相类似(一个大的元组中套着多个小的元组)
        #此时的a，b代表的是self.NEWS,args这两个列表每一个序列的相加
    def reset(self):  #相当于重置
        self.data=[0.0]*len(self.data)
    def __getitem__(self, idx):   # 实例对象通过[] 运算符取值时，会调用它的方法__getitem__(此时的data是一个列表)
        #print('这个方法被调用了')
        return self.data[idx]  #返回类内指定的列表中的相关的值（重要）

def evaluate_accuracy(net,iters_train):
    if isinstance(net,torch.nn.Module):  # isinstance是用来判断forward_propogration是否是torch.nn.Module的子类（forward_propogration为调用的torch模板时是成立的）
        net.eval()   #用于torch.nn.Module下的net的模型评估
    metric=Accumulator(2)  #创建了一个Accumulator的类名字为metric，其中存在一个参数要求，参数为n
    for X,y in iters_train:
        metric.add(accuracy(y,net(w,b,X)),y.numel())  #对应metric中的add(self,*args)的*args，y.numel()代表的是y中的元素
        #这句话调用了类函数中的add函数（60000/512+1次）,在没有metric.reset时保留前一次的值
    return metric[0]/metric[1]   #进入了类函数__getitem__两次，同时对应metric[0]对应类内的data[0]的值，metric[1]对应类内的data[0]的值


def train_epcho_ch3(net,iters_train):  # 存在两个模具，一个是网络net的模具，
    if isinstance(net,torch.nn.Module):  #判断forward_propogration是否属于torch.nn.Module模具中的
        net.train() #如果是调用模具的应当开启训练模式
    metric=Accumulator(3)  #给这个累加器传入三个参数
    for X,y in iters_train:
        y_hat=net(w,b,X)
        l=loss(y,y_hat)
        if isinstance(updater,torch.optim.Optimizer) == False:  #是否是类的从属关系
            l.sum().backward()  #对应SGD函数中的/batch_size
            updater(X.shape[0])
            metric.add(float(l.sum()),accuracy(y,y_hat),y.numel())
    return (metric[0]/metric[2],metric[1]/metric[2])

for i in range(10):
    train_matric=train_epcho_ch3(net,iters_train)
    test_acc=evaluate_accuracy(net, iters_train)
    train_loss,train_accuracy=train_matric
    print(train_loss)
    print(train_accuracy)
    print(test_acc)
    print('*'*50)





