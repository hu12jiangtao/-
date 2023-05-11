import numpy as np


class runningScore(object):
    '''
        n_classes: database的类别,包括背景
        ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
    '''

    def __init__(self, n_classes, ignore_index=None):
        self.n_classes = n_classes
        # 混淆矩阵的形状为[num_classes,num_classes],其中confusion_matrix[i][j]代表着第i类的像素被错误的归类成j类的像素的点的个数
        self.confusion_matrix = np.zeros((n_classes, n_classes))

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class) # 得到一个元素值全为True，长度为h*w的行向量
        # np.bincount的作用类似于创建一个字典,键名为[0,max_value],键值为在输入向量中出现的键名出现的次数
        # 当在label_true[1] = 4代表着第1个像素点的真实类别为4，label_pred[1] = 3代表着第1个像素点的预测类别为3，
        # 此时对应的一维向量中的1 * num_classes + 1的键名的键值应当+1代表将第真实的第4类预测成错误的第三类的个数+1
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class) # 求解得到此时的混乱矩阵，minlength代表键值对个数为n_class ** 2
        return hist

    def update(self, label_trues, label_preds):
        # label_trues = [batch,h,w](真实的每个像素点的类别) , label_preds = [batch,h,w](预测的每个像素点的类别)
        for lt, lp in zip(label_trues, label_preds): # lt.shape=lp.shape=[h,w]
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes) # 混淆矩阵的更新

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - pixel_acc:
            - class_acc: class mean acc
            - mIou :     mean intersection over union
            - fwIou:     frequency weighted intersection union
        """

        hist = self.confusion_matrix

        # ignore unlabel
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        # 此时hist.sum(axis=1)[0]代表着真实的类别0的像素点的个数，hist.sum(axis=0)[0]代表着预测的类别0的像素点的个数
        # 此时hist[0][0]代表此时真实的类别0的像素点被成功的预测成类别0的个数
        # 因此第0类的并集的像素点的个数为hist.sum(axis=1)[0]+hist.sum(axis=0)[0]-hist[0][0]
        # 交集的个数为hist[0][0]
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pixel_acc: ": acc,
                "class_acc: ": acc_cls,
                "mIou: ": mean_iou,
                "fwIou: ": fw_iou,
            },
            cls_iu,
        )

    def reset(self): # 每一轮迭代开始时将混淆矩阵给清空
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
