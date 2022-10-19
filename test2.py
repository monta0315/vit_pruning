"""Train CIFAR10 with PyTorch."""
from __future__ import print_function

import argparse
import csv
import logging
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast

from models.attn_importance_split_slim import ViT as attn_ViT
from models.slim_split import ViT_slim
from utility import Utility
from utils.utils import AverageMeter, ProgressMeter, accuracy

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--img_size", default=32, type=int, help="Resolution size")
parser.add_argument(
    "--batch_size", default=128, type=int, help="Total batch size for training."
)
parser.add_argument(
    "--block_ind", default=-1, type=int, help="Total batch size for training."
)

args = parser.parse_args()

u = Utility()

base_path,_ = u.get_model_path()


use_cuda = torch.cuda.is_available()
batch_size = args.batch_size
if use_cuda:
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu

print("==> Preparing data..")
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

print("==> Resuming from checkpoint..")
checkpoint = torch.load(base_path, map_location="cpu")
cfg = [512]*6
teacher_model = attn_ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    qkv_bias=True,
    cfg=cfg
)
model_dict = teacher_model.state_dict()
new_dict = {}
for k, v in checkpoint["net"].items():
    new_dict[k] = v
model_dict.update(new_dict)

teacher_model.load_state_dict(model_dict)


teacher_model.cuda()
teacher_model = torch.nn.DataParallel(teacher_model)
print("=> loaded teacher checkpoint")

checkpoint = torch.load(base_path, map_location="cpu")
candidate_index = range(cfg[args.block_ind])


importance = []
for delete_ind in candidate_index:
    net = attn_ViT(
        image_size=32,
        num_classes=10,
        qkv_bias=True,
        patch_size=4,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        reduce=delete_ind,
        ind=args.block_ind,
        cfg=cfg
    )

    net.cuda()
    net = torch.nn.DataParallel(net)

    model_dict = net.state_dict()


    """
        ここまでattn層勝手に追加されてない
    """

    new_dict = {}
    for k, v in checkpoint["net"].items():
        # biasは一次元512要素だからv.size(0)==512からdelete_indをひく
        if str(args.block_ind) + ".0.fn" + ".attn_to_q.bias" in k:
            new_index = [torch.arange(v.size(0)) != delete_ind]
            new_v = v[new_index]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif str(args.block_ind) + ".0.fn" + ".attn_to_k.bias" in k:
            new_index = [torch.arange(v.size(0)) != delete_ind]
            new_v = v[new_index]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif str(args.block_ind) + ".0.fn" + ".attn_to_v.bias" in k:
            new_index = [torch.arange(v.size(0)) != delete_ind]
            new_v = v[new_index]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif str(args.block_ind) + ".0.fn" + ".attn_to_q.weight" in k:
            new_v = v[torch.arange(v.size(0)) != delete_ind, :]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif str(args.block_ind) + ".0.fn" + ".attn_to_k.weight" in k:
            new_v = v[torch.arange(v.size(0)) != delete_ind, :]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif str(args.block_ind) + ".0.fn" + ".attn_to_v.weight" in k:
            new_v = v[torch.arange(v.size(0)) != delete_ind, :]
            # print(new_v.shape)
            new_dict["module." + k] = new_v
        elif str(args.block_ind) + ".0.fn" + ".attn_to_out.0.weight" in k:
            new_v = v[:, torch.arange(v.size(1)) != delete_ind]
            new_dict["module." + k] = new_v
        else:
            # print(v.shape)
            new_dict["module." + k] = v

    model_dict.update(new_dict)

    net.load_state_dict(model_dict)

    batch_time = AverageMeter("Time", ":6.3f")

    progress = ProgressMeter(len(testloader), [batch_time], prefix="Test: ")

    evaluate = True
    if evaluate:
        with torch.no_grad():
            net.eval()
            sample_correct = 0
            teacher_correct = 0
            total = 0

            end = time.time()
            kldiv = nn.KLDivLoss(reduction="sum")
            kldivloss = 0
            for i, (images, target) in enumerate(testloader):
                with autocast():

                    images = images.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                    # compute output
                    total += target.size(0)
                    teacher_output = teacher_model(images)
                    teacher_score, teacher_predicted = teacher_output.max(1)
                    teacher_correct += teacher_predicted.eq(target).sum().item()
                    with torch.no_grad():
                        output = net(images)
                        score, predicted = output.max(1)
                        sample_correct += predicted.eq(target).sum().item()

                    logsoftmax = nn.LogSoftmax(dim=1).cuda()
                    softmax = nn.Softmax(dim=1).cuda()
                    kldivloss += kldiv(logsoftmax(output),softmax(teacher_output))
            sample_acc = 100.0 * sample_correct / total
            teacher_acc = 100.0 * teacher_correct / total
            print("kldivloss",float(kldivloss))
            print("DeleteIndex", delete_ind)
            print(
                "SampleAcc: %.3f%% (%d/%d)"
                % (100.0 * sample_correct / total, sample_correct, total)
            )
            print(
                "TeacherAcc: %.3f%% (%d/%d)"
                % (100.0 * teacher_correct / total, teacher_correct, total)
            )

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        importance.append([float(kldivloss), delete_ind])
if not os.path.isdir("importances"):
    os.makedirs("importances")

name = u.get_name()
if not os.path.isdir(f"importances/self-pruned-{name}"):
    os.makedirs(f"importances/self-pruned-{name}")

with open(
    f"importances/self-pruned-{name}/block_{args.block_ind}.txt", "w"
) as f:
    for l, ind in importance:
        f.write(str(l) +","+ str(ind) + "\n")

