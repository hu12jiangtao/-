from torch.utils import data
import os
import config
from PIL import Image
import torchvision.transforms as T
import numpy as np
import pandas as pd

def rename(root_test,root_csv):
    if os.listdir(root_test)[0].split('.')[0] not in ['dog','cat']:
        file = pd.read_csv(root_csv,sep=',',engine='python')
        label = list(file.label)
        for i in range(len(label)):
            if label[i] == 0:
                label[i] = 'cat'
            else:
                label[i] = 'dog'
        name = os.listdir(root_test)
        name.sort(key=lambda x:int(x.split('.')[0]))
        old_image_path = [os.path.join(root_test,i) for i in name]
        name = [i + '.' + j for i,j in zip(label,name)]
        rename_image_path = [os.path.join(root_test,i) for i in name]
        for i,j in zip(old_image_path,rename_image_path):
            os.rename(i,j)

class DogvsCatDataset(data.Dataset):
    def __init__(self,data_dir,train=True,gamma=1.):
        super(DogvsCatDataset, self).__init__()
        self.root = os.path.join(data_dir,'train' if train is True else 'test')
        self.name = os.listdir(self.root)
        self.image_path = [os.path.join(self.root,name) for name in self.name]
        self.train = train
        mean = [0.41651208, 0.45481538, 0.48827952]
        std = [0.22509782, 0.22477485, 0.22919002]
        self.train_transforms = T.Compose([T.Resize((int(224 * gamma),int(224 * gamma))),
                                           T.RandomHorizontalFlip(),T.RandomVerticalFlip(),
                                           T.ToTensor(),T.Normalize(mean=mean,std=std)])
        self.test_transforms = T.Compose([T.Resize((224,224)),
                                         T.ToTensor(),T.Normalize(mean=mean,std=std)])
        if train is False:
            self.root_csv = os.path.join(data_dir,'sample_submission.csv')
            rename(self.root, self.root_csv)




    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image_path = self.image_path[item]
        image_class = self.name[item].split('.')[0]
        if image_class == 'cat':
            label = 0 # 猫的标签为0
        elif image_class == 'dog':
            label = 1 # 狗的标签为1
        else:
            label = -1 # 在计算损失的时候ignore
        image = Image.open(image_path)
        if self.train is True:
            image = self.train_transforms(image)
        else:
            image = self.test_transforms(image)
        return image, label

if __name__ == '__main__':
    dataset = DogvsCatDataset(config.root_dir,train=True,gamma=0.714)
    c = dataset[0][0].shape
    print(c)
