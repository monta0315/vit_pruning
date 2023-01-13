import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.attn_importance_split_slim import ViT
from models.split_vit import ViT as base
from utility import Utility
from utils.utils import (linear_regression, progress_bar, test,
                         txt_impotance_scores_convert_array)

parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--rate", default=0.3, type=float, help="Resolution size")
args = parser.parse_args()
threshould = args.rate

u = Utility("base")
# name = u.get_name()
name = "base-CIFAR10-100epochs-256bs-each"
_,model_path = u.get_model_path()

block_ind = 0

importance_score_lists = []
threshould = args.rate
pruned_target_index = []

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0
start_epoch = 0

def threshoud_importance_score(score_list):
    for i in range(6):
        target = score_list[i]
        step = []
        if len(target) != 0:
            target.sort()
            thre_index = int(len(target)*threshould)
            for i in range(thre_index):
                step.append(target[i][1])
        pruned_target_index.append(step)


def make_mask(score_lists,all_importance_score):
    cfg_mask = []
    cfg = []
    threshoud_importance_score = culc_threshoud_importance_score(all_importance_score)
    for list in score_lists:
        list = torch.FloatTensor(list)
        threshoud_importance_score_ = copy.copy(threshoud_importance_score)
        while sum(i > threshoud_importance_score_ for i in list) % 8 != 0:
            threshoud_importance_score_ -= 0.001
        #step = list.lt(threshoud_importance_score_)
        step = list.gt(threshoud_importance_score_)
        cfg_mask.append(step)
        cfg.append(int(step.sum()))

    print(100-(sum(cfg)/(512*6))*100,"%")
    
    return cfg,cfg_mask

def culc_threshoud_importance_score(all_importance_score) -> float:
    all_importance_score.sort()
    #all_importance_score = all_importance_score[::-1]
    #all_importance_score.sort()
    thre_index = int(len(all_importance_score)*threshould)
    return all_importance_score[thre_index - 1]



all_importance_score = []

for i in range(6):
    step_ = txt_impotance_scores_convert_array(name,i)
    step = linear_regression(step_)
    #value = step_[4].copy()
    hard = np.array(step_[1])
    soft = np.array(step_[2])
    #value = hard*0.4+soft
    value = soft
    for i in range(len(value)):
        value[i] -= step[i]

    importance_score_lists.append(value)

all_importance_score = np.array(importance_score_lists).flatten()

# threshoud_importance_score(importance_score)

# if apply only method2 
#model_path = f"ch_sele_checkpoints/{name}.pth"
model_path = "pruned1_checkpoints/self-pruned-newest-CIFAR10-100epochs-256bs-each-1.0.pth"

checkpoint = torch.load(model_path,map_location="cpu")
base_model = base(    
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
    qkv_bias=True)

base_model.to(device)

base_model.load_state_dict(checkpoint['net'])

print("Base_model_accuracy")
#test(base_model,device,name,checkpoint)

cfg,cfg_mask = make_mask(importance_score_lists,all_importance_score)


print("pruned",cfg)
# print("before",checkpoint['cfg'])

new_model = ViT(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=512,
    dropout=0.1,
    emb_dropout=0.1,
    cfg=cfg,
    qkv_bias=True
)

new_model.to(device)


new_model_dict = new_model.state_dict().copy()

new_dict = {}

for k, v in checkpoint['net'].items():
    new_dict[k] = v


""" u.debag(new_dict) """



for dim in range(6):
    target_layers = [
        f"transformer.layers.{dim}.0.fn.attn_to_q.bias",
        f"transformer.layers.{dim}.0.fn.attn_to_k.bias",
        f"transformer.layers.{dim}.0.fn.attn_to_v.bias",
        f"transformer.layers.{dim}.0.fn.attn_to_q.weight",
        f"transformer.layers.{dim}.0.fn.attn_to_k.weight",
        f"transformer.layers.{dim}.0.fn.attn_to_v.weight",
    ]
    for l in target_layers:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[dim].cpu().numpy())))
        v = new_dict[l].clone()
        new_dict[l] = v[idx.tolist()].clone()

    l = f"transformer.layers.{dim}.0.fn.attn_to_out.0.weight"
    idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[dim].cpu().numpy())))
    v = new_dict[l].clone()
    new_dict[l] = v[:, idx.tolist()].clone()


new_model_dict.update(new_dict)

#u.debag(new_model_dict)

new_model.load_state_dict(new_model_dict)


test(new_model,device,name,checkpoint,True,cfg,3,"test-last",cfg_mask)
