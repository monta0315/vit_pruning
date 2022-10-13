import os

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utility import Utility
from utils.utils import progress_bar

u = Utility()
name = u.get_name()
_,model_path = u.get_model_path()

block_ind = 0

importance_score = []
threshould = 0.2
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

for i in range(6):
    with open(f"importances/self-pruned-{name}test/block_{i}.txt",'r') as f:
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


checkpoint = torch.load(model_path,map_location="cpu")
new_dict = {}

for k, v in checkpoint['net'].items():
    new_dict[k] = v


""" u.debag(new_dict) """


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
        new_index = [i for i in torch.arange(target.size(0))]
        new_target = target[new_index]
        new_dict[l] = new_target

    l = f"transformer.layers.{i}.0.fn.attn_to_out.0.weight"
    target = new_dict[l]
    new_index = [i not in pruned_target_index[dim] for i in torch.arange(target.size(1))]
    new_target = target[:,new_index]
    new_dict[target] = new_target




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



def test(epoch):
    global best_acc
    criterion = nn.CrossEntropyLoss()
    net = new_dict.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
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


    acc = 100.0 * correct / total
    if acc > best_acc:
        print("Saving")
        state = {"net": net.state_dict(), "acc": acc, "epoch": epoch}

        if not os.path.isdir("pruned2_checkpoints"):
            os.makedirs("pruned2_checkpoints")

        torch.save(state,f"pruned2_checkpoints/second_pruned_{name}.pth")
        best_acc = acc




for epoch in range(start_epoch, 1):
    test(epoch)
