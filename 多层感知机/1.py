import torch
'''
a=torch.tensor([[0],[0],[0]])
b=torch.tensor([[1],[-1],[2]])
print(torch.max(a,b))
'''

class machine():
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def add(self,c):
        return self.a+self.b+c
    def sub(self):
        return self.a-self.b

Machine=machine(4,1)
c=Machine.add(5)

class Person:

    def __init__(self):
        print('执行__init__方法')

    def __call__(self, *args, **kwargs):
        print('执行__call__方法')

p1 = Person()  # 返回：执行__init__方法  解读：实例化只执行__init__方法
p1()