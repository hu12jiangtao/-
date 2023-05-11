import torch

device = torch.device('cuda')

VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
num_classes = len(VOC_CLASSES)

voc_root = ''

train_batch_size = 4
test_batch_size = 1
lr = 1e-4
momentum = 0.9
weight_decay = 1e-4
epochs = 50
