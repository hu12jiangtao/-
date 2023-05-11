import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import LoadDataset
from Models import FCN
import cfg
from metrics import averageMeter, runningScore
import time

# 得到每张图片的预测时间以及频率
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

BATCH_SIZE = 4
miou_list = [0]

Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Load_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

net = FCN.FCN(num_class)
net.eval()
net.to(device)
net.load_state_dict(t.load("./Results/weights/xxx.pth"))
running_metrics_val = runningScore(12)
time_meter = averageMeter()

for i, sample in enumerate(test_data):
	time_start = time.time()
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	true_label = label.data.cpu().numpy()
	running_metrics_val.update(true_label, pre_label)  # 计算预测的像素图的所有的像素点的预测准确率和miu的值
	time_meter.update(time.time() - time_start, n=data.size(0))

metrics = running_metrics_val.get_scores()
for k, v in metrics[0].items():
	print(k, v)
print('inference time per image: ', time_meter.avg)
print('inference fps: ', 1 / time_meter.avg)