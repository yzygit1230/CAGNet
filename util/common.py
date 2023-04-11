import os
import torch
import random
import cv2
import numpy as np
from pathlib import Path
import matplotlib
from tqdm import tqdm
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn
'''
def result_visual(img1, img2, label1, label2, out1, out2):
    # a, b, c, d, e, f 为[H,W,3]
    if len(img1.shape) < 3:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) < 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    if len(label1.shape) < 3:
        label1 = cv2.cvtColor(label1, cv2.COLOR_GRAY2BGR)
    if len(label2.shape) < 3:
        label2 = cv2.cvtColor(label2, cv2.COLOR_GRAY2BGR)
    if len(out1.shape) < 3:
        out1 = cv2.cvtColor(out1, cv2.COLOR_GRAY2BGR)
    if len(out2.shape) < 3:
        out2 = cv2.cvtColor(out2, cv2.COLOR_GRAY2BGR)
    row_white = np.ones((10, img1.shape[0], 3)) * 255
    column_white = np.ones((img1.shape[1] * 2 + 10, 10, 3)) * 255

    left_part = np.concatenate([img1, row_white, img2], axis=0)
    middle_part = np.concatenate([label1, row_white, label2], axis=0)
    right_part = np.concatenate([out1, row_white, out2], axis=0)

    out = np.concatenate([left_part, column_white, middle_part, column_white, right_part], axis=1)

    return out
'''

def plot_results(result_paths, save_dir=None, names=None):
    if not isinstance(result_paths, list):
        result_paths = [result_paths]

    fig, ax = plt.subplots(3, 3, figsize=(20, 20), tight_layout=True)

    for result_path in result_paths:
        assert result_path.endswith(".txt"), 'please check path: {}'.format(result_path)
        if save_dir is None:
            save_dir = result_path.replace(result_path.split(os.sep)[-1], '')

        ax = ax.ravel()
        s = ['lr', 'P', 'R', 'F1', 'mIOU', 'OA', 'best_metric', 'train_loss', 'val_loss']
        results = np.loadtxt(result_path, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9], skiprows=1, ndmin=2).T
        n = results.shape[1]  # number of rows
        x = range(n)
        for i in range(len(s)):
            y = results[i, x]
            if i == 6:
                y[y < 0.5] = 0.5
            ax[i].plot(x, y, marker='', label=s[i], linewidth=2, markersize=8)
            ax[i].set_title(s[i], fontsize=20)

    if names is None:
        names = result_paths
    ax[6].legend(names, loc='best')
    fig.savefig(Path(save_dir) / 'results.jpg', dpi=400)
    plt.close()
    del fig, ax


def init_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    from torch.backends import cudnn
    cudnn.benchmark = False
    cudnn.deterministic = True


def check_dirs():
    print("\n"+"-"*30+"Check Dirs"+"-"*30)
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
        os.mkdir('./runs/train')
        os.mkdir('./runs/eval')
    file_names = os.listdir('./runs/train')
    file_names = [int(i) for i in file_names] + [0]
    new_file_name = str(max(file_names) + 1)

    save_path = './runs/train/' + new_file_name
    every_ckp_save_path = os.path.join(save_path, 'every_ckp')
    best_ckp_save_path = os.path.join(save_path, 'best_ckp')
    os.mkdir(save_path)
    os.mkdir(every_ckp_save_path)
    os.mkdir(best_ckp_save_path)
    print("checkpoints & results are saved at: {}".format(save_path))

    result_save_path = os.path.join(save_path, "result.txt")

    best_ckp_file = None

    return save_path, best_ckp_save_path, best_ckp_file, result_save_path, every_ckp_save_path


def check_eval_dirs():
    print("\n"+"-"*30+"Check Dirs"+"-"*30)
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
        os.mkdir('./runs/train')
        os.mkdir('./runs/eval')
    file_names = os.listdir('./runs/eval')
    file_names = [int(i) for i in file_names] + [0]
    new_file_name = str(max(file_names) + 1)
    save_path = './runs/eval/' + new_file_name
    os.mkdir(save_path)

    result_save_path = os.path.join(save_path, "eval_result.txt")
    print("results are saved at: {}".format(save_path))

    return save_path, result_save_path


def compute_metrics(tn_fp_fn_tps):   # 计算各种指标
    p, r, f1, miou, iou_0, iou_1, oa, kappa = [], [], [], [], [], [], [], []
    for tn_fp_fn_tp in tn_fp_fn_tps:
        tn, fp, fn, tp = tn_fp_fn_tp

        p_tmp = tp / (tp + fp)
        r_tmp = tp / (tp + fn)
        f1_tmp = 2 * p_tmp * r_tmp / (r_tmp + p_tmp)
        iou_0_tmp = tn / (tn + fp + fn)
        iou_1_tmp = tp / (tp + fp + fn)
        miou_tmp = 0.5 * tp / (tp + fp + fn) + 0.5 * tn / (tn + fp + fn)
        oa_tmp = (tp + tn) / (tp + tn + fp + fn)
        p0 = oa_tmp
        pe = ((tp+fp)*(tp+fn)+(fp+tn)*(fn+tn))/(tp+fp+tn+fn)**2
        kappa_tmp = (p0-pe) / (1-pe)
        
        p.append(p_tmp)
        r.append(r_tmp)
        f1.append(f1_tmp)
        miou.append(miou_tmp)
        iou_0.append(iou_0_tmp)
        iou_1.append(iou_1_tmp)
        oa.append(oa_tmp)
        kappa.append(kappa_tmp)
        
        print('Precision: {}\nRecall: {}\nF1-Score: {} \nmIOU:{} \nIOU_0:{} \nIOU_1:{}'.format(p,r,f1,miou,iou_0,iou_1))
        print('OA: {}\nKappa: {}'.format(oa,kappa))

    return np.array(p), np.array(r), np.array(f1), np.array(miou), np.array(iou_0), np.array(iou_1), np.array(oa), np.array(kappa)


def gpu_info():
    print("\n" + "-" * 30 + "GPU Info" + "-" * 30)
    gpu_count = torch.cuda.device_count()
    x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
    s = 'Using CUDA '
    c = 1024 ** 2  # bytes to MB
    if gpu_count > 0:
        print("Using GPU count: {}".format(torch.cuda.device_count()))
        for i in range(0, gpu_count):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g name='%s', memory=%dMB" % (s, i, x[i].name, x[i].total_memory / c))
    else:
        print("Using CPU !!!")


class SaveResult:
    def __init__(self, result_save_path):
        self.result_save_path = result_save_path

    def prepare(self):
        # 为写日志做准备，先创建这个文件
        with open(self.result_save_path, "w")as f:
            f.write(('%-7s'+'%-12s' * 9) % (
                'epoch', 'lr', 'P', 'R', 'F1', 'mIOU', 'OA', 'best_metric', 'train_loss', 'val_loss') + "\n")

    def show(self, p, r, f1, miou, oa,
             refer_metric=np.array(0), best_metric=0, train_avg_loss=0, val_avg_loss=0, lr=0, epoch=0):
        print(
            "lr:{}  P:{}  R:{}  F1:{}  mIOU:{} OA:{}\nrefer_metric-mean: {} best_metric: {}".format(
                lr, p, r, f1, miou, oa, round(refer_metric.mean(), 5), round(best_metric, 5)))
        with open(self.result_save_path, "a")as f:
            f.write(
                ('%-7s'+'%-12s' * 9) % (str(epoch), str(round(lr, 8)),
                                        str(round(float(p.mean()), 6)),
                                        str(round(float(r.mean()), 6)),
                                        str(round(float(f1.mean()), 6)),
                                        str(round(float(miou.mean()), 6)),
                                        str(round(float(oa.mean()), 6)),
                                        str(round(float(best_metric), 6)),
                                        str(round(train_avg_loss, 6)),
                                        str(round(val_avg_loss, 6))) + "\n")

        plot_results(self.result_save_path)


class CosOneCycle:
    def __init__(self, optimizer, max_lr, epochs, min_lr=None, up_rate=0.3):  # max=0.0035, min=0.00035
        self.optimizer = optimizer

        self.max_lr = max_lr
        if min_lr is None:
            self.min_lr = max_lr / 10
        else:
            self.min_lr = min_lr
        self.final_lr = self.min_lr / 50

        self.new_lr = self.min_lr

        self.step_i = 0
        self.epochs = epochs
        self.up_rate = up_rate
        assert up_rate < 0.5, "up_rate should be smaller than 0.5"

    def step(self):
        self.step_i += 1
        if self.step_i < (self.epochs*self.up_rate):
            self.new_lr = 0.5 * (self.max_lr - self.min_lr) * (
                        np.cos((self.step_i/(self.epochs*self.up_rate) + 1) * np.pi) + 1) + self.min_lr
        else:
            self.new_lr = 0.5 * (self.max_lr - self.final_lr) * (np.cos(
                ((self.step_i - self.epochs * self.up_rate) / (
                            self.epochs * (1 - self.up_rate))) * np.pi) + 1) + self.final_lr

        if len(self.optimizer.state_dict()['param_groups']) == 1:
            self.optimizer.param_groups[0]["lr"] = self.new_lr
        elif len(self.optimizer.state_dict()['param_groups']) == 2:
            self.optimizer.param_groups[0]["lr"] = self.new_lr / 10
            self.optimizer.param_groups[1]["lr"] = self.new_lr
        else:
            raise Exception('Error. You need to add a new "elif". ')

    def plot_lr(self):
        all_lr = []
        for i in range(self.epochs):
            all_lr.append(self.new_lr)
            self.step()
        fig = seaborn.lineplot(x=range(self.epochs), y=all_lr)
        fig = fig.get_figure()
        fig.savefig('./lr_schedule.jpg', dpi=200)
        self.step_i = 0
        self.new_lr = self.min_lr


class ScaleInOutput:
    def __init__(self, input_size=512):
        if isinstance(input_size, int):
            input_size = (input_size, input_size)
        self.input_size = input_size
        self.output_size = None

    def scale_input(self, imgs: tuple):
        assert isinstance(imgs, tuple), "Please check the input type. It should be a 'tuple'."
        imgs = list(imgs)
        self.output_size = imgs[0].shape[2:]

        for i, img in enumerate(imgs):
            imgs[i] = F.interpolate(img, self.input_size, mode='bilinear', align_corners=True)

        return tuple(imgs)

    def scale_output(self, outs: tuple):
        if type(outs) in [torch.Tensor]:
            outs = (outs,)
        assert isinstance(outs, tuple), "Please check the input type. It should be a 'tuple'."
        outs = list(outs)

        assert self.output_size is not None, \
            "Please call 'scale_input' function firstly, to make sure 'output_size' is not None"

        for i, out in enumerate(outs):
            outs[i] = F.interpolate(out, self.output_size, mode='bilinear', align_corners=True)

        return tuple(outs)