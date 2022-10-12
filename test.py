import os

import numpy as np
import torch

from utility import Utility

u = Utility()
name = u.get_name()
_,model_path = u.get_model_path()

block_ind = 0

importance_score = []
threshould = 0.2
pruned_target_index = []

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

for i in range(6):
    with open(f"importances/self-pruned-{name}/block_{i}.txt",'r') as f:
        f.seek(0, os.SEEK_END)
        isempty = f.tell() == 0
        f.seek(0)
        step = []
        if not isempty:
            for i in f:
                score,index = i[:-1].split(',')
                step.append([score,index])
        else:
            print(f"block_{i} is empty file")
        
        importance_score.append(step)

threshoud_importance_score(importance_score)


checkpoint = torch.load(model_path)
new_dict = {}
for k, v in checkpoint['net'].items():
    new_dict[k] = v

checkpoint = new_dict


for dim in range(6):
    target_layers = [
        f"transformer.layers.{i}.0.fn.attn_to_q.bias",
        f"transformer.layers.{i}.0.fn.attn_to_k.bias",
        f"transformer.layers.{i}.0.fn.attn_to_v.bias",
        f"transformer.layers.{i}.0.fn.attn_to_q.weight",
        f"transformer.layers.{i}.0.fn.attn_to_k.weight",
        f"transformer.layers.{i}.0.fn.attn_to_v.weight",
    ]
    for l in target_layers:
        target = new_dict[l]
        new_index = [i not in pruned_target_index[0][dim] for i in torch.arange(v.size(0))]
        new_target = target[new_index]
        new_dict[l] = new_target

    l = f"{i}.0.fn.attn_to_out.0.weight"
    target = new_dict[f"{i}.0.fn.attn_to_out.0.weight"]
    new_index = [i not in pruned_target_index[0][dim] for i in torch.arange(v.size(1))]
    new_target = target[new_index]
    new_dict[target] = new_target

if os.isdir("pruned2_checkpoints"):
    os.makedirs("pruned2_checkpoints")

torch.save(new_dict,f"pruned2_checkpoints/second_pruned_{name}.pth")
