import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from models.attn_importance_split_slim import ViT
from utility import Utility
from utils.utils import progress_bar

u = Utility("base")
name = u.get_name()
_,model_path = u.get_model_path()

block_ind = 0

importance_score_lists = []
threshould = 0.40
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
        threshoud_importance_score_ = threshoud_importance_score
        while sum(i > threshoud_importance_score_ for i in list) % 8 != 0:
            threshoud_importance_score_ -= 0.001
        step = list.gt(threshoud_importance_score_)
        cfg_mask.append(step)
        cfg.append(int(step.sum()))
    
    return cfg,cfg_mask

def culc_threshoud_importance_score(all_importance_score) -> float:
    all_importance_score.sort()
    thre_index = int(len(all_importance_score)*threshould)
    return all_importance_score[thre_index - 1]



all_importance_score = []

for i in range(6):
    with open(f"importances/self-pruned-{name}/block_{i}.txt",'r') as f:
        f.seek(0, os.SEEK_END)
        isempty = f.tell() == 0
        f.seek(0)
        step = []
        if not isempty:
            for i in f:
                score,index = i[:-1].split(',')
                step.append(float(score))
                all_importance_score.append(float(score))
        else:
            print(f"block_{i} is empty file")
        
        importance_score_lists.append(step)

# threshoud_importance_score(importance_score)

# if apply only method2 
model_path = f"ch_sele_checkpoints/{name}.pth"

checkpoint = torch.load(model_path,map_location="cpu")

cfg,cfg_mask = make_mask(importance_score_lists,all_importance_score)


print("pruned",cfg)
#print("before",checkpoint['cfg'])

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


def test(model, pruned=False, cfg=None):
    global name
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

        print("Acc: %.3f%% (%d/%d)" % (100.0 * correct / total, correct, total))
    if pruned:
        state = {
            "net": model.state_dict(),
            "acc": 100.0 * correct / total,
            "epoch": checkpoint["epoch"],
            "cfg": cfg,
        }
        torch.save(state, f"./pruned2_checkpoints/self-pruned-{name}-all.pth".format(4))
        print("Complete!!!!!!!!!!!!!!!")


test(new_model,True,cfg)
