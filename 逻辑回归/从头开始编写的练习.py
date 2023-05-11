import torchvision
import torch
from torch.utils import data
from d2l import torch as d2l
from torchvision import transforms
import matplotlib.pyplot as plt
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
    iters_train = data.DataLoader(mninst_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(1))
    iters_test = data.DataLoader(mninst_test, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers(1))
    return iters_train,iters_test

iters_train,iters_test=load_data(batch_size=256)

#构建网络参数
w=torch.normal(0,1,[784,10],requires_grad=True)  # 特征值为784，最后softmax分为了10类
b=torch.zeros([1,10],requires_grad=True)
#进行前项传播（net的构建）
def softmax(x):
    e_x=torch.exp(x)
    patition=torch.sum(e_x,dim=1,keepdim=True)
    return e_x/patition

def net(w,b,X):  #此时输入X为256*1*28*28，需要转换为256*784
    return softmax(torch.matmul(X.reshape(-1,w.shape[0]),w)+b)

#构建损失函数
def loss(y,y_hat):   #此时y是一个列表（256，）,y_hat为256*10
    return -torch.log(y_hat[list(range(y_hat.shape[0])),y])  #此时输出一个列表

def updater_params(params,batch_size,lr=0.1):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()

def updater(batch_size):
    return updater_params([w,b],batch_size,lr=0.1)

#模型的评估
def accuracy(y,y_hat):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=torch.argmax(y_hat,dim=1)
        judge=(y_hat.type(y.dtype)==y)
        return judge.type(y.dtype).sum()  #判断出了y,y_hat中相对应的个数

class Accumulator():
    def __init__(self,n):
        self.data=[0.0]*n  #其为一个列表
    def add(self,*args):
        self.data=[i+float(j) for i,j in zip(self.data,args)]
    def __getitem__(self, index):
        return self.data[index]

def evaluate_accuracy(net,deal_data):
    if isinstance(net,torch.nn.Module):
        net.eval()
    matric = Accumulator(2)
    for X,y in deal_data:
        matric.add(accuracy(y,net(w,b,X)),y.numel())
    return matric[0]/matric[1]   #此时为准确率

#构建一个整体的模型
def train_epcho_ch3(iters_train,net,updater,batch_size):
    if isinstance(net,torch.nn.Module):
        net.train()
    matric=Accumulator(3)
    for X,y in iters_train:
        print(X.is_leaf)
        y_hat=net(w,b,X)
        epcho_loss=loss(y,y_hat)
        if isinstance(updater,torch.nn.Module) == False:
            epcho_loss.sum().backward()
            updater(batch_size)
            #print(accuracy(y,y_hat))
            matric.add(float(epcho_loss.sum()),accuracy(y,y_hat),y.numel())
    return (matric[0]/matric[2],matric[1]/matric[2])


epcho_num=10
for i in range(epcho_num):
    train_matric=train_epcho_ch3(iters_train, net, updater, batch_size=256)
    #train_matric=train_epcho_ch3(net, iters_train)
    train_loss, train_accuracy=train_matric
    print('train loss:',train_loss)
    print('train accuracy:', train_accuracy)
    test_accuracy=evaluate_accuracy(net, iters_test)
    print('test accuracy:', test_accuracy)
    print('*'*50)

#进行预测
def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(d2l.argmax(net(w,b,X), axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    d2l.show_images(
        d2l.reshape(X[0:n], (n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()



predict_ch3(net, iters_test, n=6)

