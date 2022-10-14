import argparse
from random import shuffle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description="which model use")
args = parser.parse_args()


from models.attn_importance_split_slim import ViT as attn_ViT
from models.select_split import ViT, channel_selection
from models.slim_split import ViT_slim
from utility import Utility

"""
    channel selection layerのinputのindex
    multi head attentionのために8の倍数に設定するから指定してあげる必要がある
"""
selection_index = [19, 41, 63, 85, 107, 129]

device = "cuda" if torch.cuda.is_available() else "cpu"
cudnn.benchmark = True


model = ViT(
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

u = Utility()

name = u.get_name()
model_path = f"ch_sele_checkpoints/{name}.pth"


model = model.to(device)
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
start_epoch = checkpoint["epoch"]
best_prec1 = checkpoint["acc"]
model.load_state_dict(checkpoint["net"])


print(
    "=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(
        model_path, checkpoint["epoch"], best_prec1
    )
)


"""
    ここから下がわけわからん
"""
total = 0
index = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        total += m.indexes.data.shape[0]

bn = torch.zeros(total) # total = 512 * 6(multi)
index = 0
for m in model.modules():
    if isinstance(m, channel_selection):
        size = m.indexes.data.shape[0]
        bn[index : (index + size)] = m.indexes.data.abs().clone()
        index += size

"""
    bn = tensor([0.9922, 0.9848, 1.0028,  ..., 0.9977, 1.0119, 0.9991]) torch.Size([3072])
    刈り込む層の各チャンネルの重みを６層分まとめてる
"""

# 重みが小さいものの下から3割のindexを判明させている
percent = 0.3
y, i = torch.sort(bn)
"""
    y tensor([0.9551, 0.9560, 0.9560,  ..., 1.0368, 1.0370, 1.0408]) torch.Size([3072]) 
    i tensor([ 425,  441,  130,  ..., 1728, 1847, 1763]) torch.Size([3072])
"""
thre_index = int(total * percent)
thre = y[thre_index]


pruned = 0
cfg = []
cfg_mask = []
"""
    cfg_maskはtorch.Size([512])が6つ格納されたlistで各層のどのチャンネルを刈り込むかを適用するためのmask
"""


for k, m in enumerate(model.modules()):
    if isinstance(m, channel_selection):
        # print("m",m)
        if k in selection_index:
            """
                weight_copy_tensorに対して、threよりも大きいweightを真偽値で出力する
                >>> weight.gt(torch.tensor([[1, 2], [3, 4]]), weight = torch.tensor([[1, 1], [4, 4]]))
                tensor([[False, True], [False, False]])
            """
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            thre_ = thre.clone()
            """
                 pruning後のチャンネル数が8の倍数じゃないとheadの数と合わなくなるため、閾値の値を少しずつ下げ、チャンネル数が8nになるようにしている
            """
            while torch.sum(mask) % 8 != 0:
                thre_ = thre_ - 0.0001
                mask = weight_copy.gt(thre_).float().cuda()
        else:
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()

        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.indexes.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        """

        """
        print(
            "layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}".format(
                k, mask.shape[0], int(torch.sum(mask))
            )
        )


pruned_ratio = pruned / total
print("prunded_ratio",f"{float(pruned_ratio)*100}%")
print("Pre-processing Successful!")


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
        torch.save(state, f"./pruned1_checkpoints/self-pruned-{name}.pth".format(4))
        print("Complete!!!!!!!!!!!!!!!")


test(model)


print("cfg_prune", cfg)
newmodel = attn_ViT(
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


newmodel.to(device)
newmodel_dict = newmodel.state_dict().copy()

i = 0
newdict = {}

#u.debag(model.state_dict())

def target_layer(i:int):
    target_layer_list = [
        f"transformer.layers.{i}.0.fn.attn_to_q.bias",
        f"transformer.layers.{i}.0.fn.attn_to_k.bias",
        f"transformer.layers.{i}.0.fn.attn_to_v.bias",
        f"transformer.layers.{i}.0.fn.attn_to_q.weight",
        f"transformer.layers.{i}.0.fn.attn_to_k.weight",
        f"transformer.layers.{i}.0.fn.attn_to_v.weight",
    ]
    return target_layer_list

for k, v in model.state_dict().items():
    target = target_layer(i)
    if k in target:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif f"{i}.0.fn.attn_to_out.0.weight"  in k:
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:, idx.tolist()].clone()
        i = i+1
    elif k in newmodel.state_dict():
        newdict[k] = v


newmodel_dict.update(newdict)


newmodel.load_state_dict(newmodel_dict)


# torch.save(newmodel.state_dict(), 'pruned.pth')
print("after pruning: ", end=" ")
test(newmodel, True, cfg)
