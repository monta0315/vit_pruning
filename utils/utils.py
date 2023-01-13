# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import copy
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from sklearn.linear_model import LinearRegression

from models.attn_importance_split_slim import ViT
from models.select_split import ViT as select_base
from models.select_split import channel_selection
from utility import Utility


def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def add(self, nsum, n=1):
        self.val = nsum / n
        self.sum += nsum
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def txt_impotance_scores_convert_array(name,ind):
    importance_score_lists = []
    with open(f"importances/self-pruned-{name}/block_{ind}.txt",'r') as f:
        f.seek(0, os.SEEK_END)
        isempty = f.tell() == 0
        f.seek(0)
        ind_arr = []
        soft = []
        hard = []
        value = []
        method2_score = []
        if not isempty:
            for i in f:
                index,soft_score,hard_score,v,method2 = i[:-1].split(',')
                ind_arr.append(index)
                soft.append(float(soft_score))
                hard.append(float(hard_score))
                value.append(float(v))
                method2_score.append(float(method2))
        else:
            print(f"block_{i} is empty file")
        
        importance_score_lists.append(ind_arr)
        importance_score_lists.append(soft)
        importance_score_lists.append(hard)
        importance_score_lists.append(value)
        importance_score_lists.append(method2_score)

        #return [512,512,512,512,512]

    return importance_score_lists

def linear_regression(importance_score_lists):
    x = np.array(importance_score_lists[2])
    y = np.array(importance_score_lists[1])
    print("corrcoef",np.corrcoef(x,y)[0][1])
    # 転置
    x = np.array(importance_score_lists[2]).reshape((-1,1))
    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    # 決定係数を出す意味があるのかはわからん => あんま関係ない気もする
    print(f"coefficient of determination: {r_sq}")
    y_h_arr = model.predict(x)
    y_loss_arr = []
    for i in range(len(x)):
        # np.arrayで下の数式を計算するのとただのarrayで計算するのだと計算結果が微妙に違う
        y_loss = math.fabs(y_h_arr[i]-y[i])
        y_loss_arr.append(y_loss)

    return y_loss_arr

def test(model,device,name,checkpoint, pruned=False, cfg=None,method=1,strategy="all",cfg_mask=[]):
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        testset = torchvision.datasets.CIFAR10(
            root="data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=8
        )
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                progress_bar(
                    batch_idx,
                    len(testloader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        test_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

            print("Acc: %.3f%% (%d/%d)" % (100.0 * correct / total, correct, total))
        if pruned:
            rate = 1-sum(cfg)/(512*6)
            formated_rate = '{:.1f}'.format(rate)
            state = {
                "net": model.state_dict(),
                "acc": 100.0 * correct / total,
                "epoch": checkpoint["epoch"],
                "cfg": cfg,
                "rate":rate,
                "cfg_mask":cfg_mask
            }
            torch.save(state, f"./pruned{method}_checkpoints/self-pruned-{name}-{strategy}-{formated_rate}.pth".format(4))
            print("Saved model name",f"./pruned{method}_checkpoints/self-pruned-{name}-{strategy}-{formated_rate}.pth".format(4))

            if not os.path.isdir("results"):
                os.makedirs("results")
            
            with open(
                f"results/self-pruned-{method}-{name}-{strategy}-{formated_rate}.txt","w"
            )as f:
                f.write(f"accuracy:{100.0 * correct / total},cfg:{cfg},rate:{rate}")

            print("Complete!!!!!!!!!!!!!!!")

def get_select_values(ind):
    sele_val = []
    selection_index = [19, 41, 63, 85, 107, 129]
    model = select_base(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=512,  # 512
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True
    )
    model_path = "ch_sele_checkpoints/newest-CIFAR10-100epochs-256bs.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["net"])
    for k,m in enumerate(model.modules()):
        if isinstance(m,channel_selection):
            if k in selection_index:
                sele_val.append(m.indexes.data.abs().clone())

    return sele_val[ind]

def get_mask(name,layer_num=0):
    checkpoint = torch.load(name, map_location="cpu")
    
    return  checkpoint["cfg_mask"][layer_num]

def squeeze_channel(target,mask):
    idx = np.squeeze(np.argwhere(np.asarray(mask.numpy())))
    v = torch.Tensor(target).clone()

    # ここの絞り込みがlistだとうまくいかないからTensorにしゃーなしでかえる
    v = v[idx.tolist()].clone()
    return v.tolist()
