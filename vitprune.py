import argparse
from random import shuffle

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='which model use')
args = parser.parse_args()


from models.vit_select import ViT, channel_selection
from models.vit_slim import ViT_slim

"""
    channel selection layerのinputのindex
    multi head attentionのために8の倍数に設定するから指定してあげる必要がある
"""
selection_index = [20,46,72,98,124,150]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True


model = ViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,                  # 512
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
    )

name = "vit-CIFAR10-100epochs-256bs"
model_path = f"checkpoint/{name}.pth"



model = model.to(device)
print("=> loading checkpoint '{}'".format(model_path))
checkpoint = torch.load(model_path)
start_epoch = checkpoint['epoch']
best_prec1 = checkpoint["acc"]
model.load_state_dict(checkpoint['net'])
print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}".format(model_path, checkpoint['epoch'], best_prec1))


'''
    ここから下がわけわからん
'''
total = 0
index = 0
for m in model.modules():
    if isinstance(m,channel_selection):
        total += m.indexes.data.shape[0]



bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m,channel_selection):
        size = m.indexes.data.shape[0]
        bn[index:(index+size)] = m.indexes.data.abs().clone()
        index += size

# 0.3未満の何かを見つけてる？？
percent = 0.3
y,i = torch.sort(bn)
thre_index = int(total*percent)
thre = y[thre_index]


pruned = 0
cfg = []
cfg_mask = []


for k,m in enumerate(model.modules()):
    if isinstance(m,channel_selection):
        #print("m",m)
        if k in selection_index:
            '''
                weight_copy_tensorに対して、threよりも大きいweightを真偽値で出力する
                >>> weight.gt(torch.tensor([[1, 2], [3, 4]]), weight = torch.tensor([[1, 1], [4, 4]]))
                tensor([[False, True], [False, False]])
            '''
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            thre_ = thre.clone()
            # kが特定の値の時だけ閾値を下げてる？
            while (torch.sum(mask)%8 != 0):
                thre_ = thre_ - 0.0001
                mask = weight_copy.gt(thre_).float().cuda()
        else:
            weight_copy = m.indexes.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
        
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.indexes.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        '''
            kはLayerIndexでその層にある元々のチャンネル数がmask.shape[0]で与えられる？？
            maskの状態はまだ刈り込みは行っていない？
            Layerってなんやねん
        '''
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, mask.shape[0], int(torch.sum(mask))))


pruned_ratio = pruned/total
print('Pre-processing Successful!')
print("cfg",cfg)


def test(model,pruned=False,cfg=None):
    global name
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset = torchvision.datasets.CIFAR10(root="data",train=False,download=True,transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=8)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx,(inputs,targets) in enumerate(testloader):
            inputs,targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            _,predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        print('Acc: %.3f%% (%d/%d)' % (100.*correct/total, correct, total))
    if pruned:
        state = {
            'net':model.state_dict(),
            'acc':100.*correct/total,
            'epoch':checkpoint['epoch'],
            'cfg':cfg
        }
        torch.save(state, f'./checkpoint/self-pruned-{name}.pth'.format(4))
        print("Complete!!!!!!!!!!!!!!!")

test(model)
cfg_prune = []
for i in range(len(cfg)):
    if i%2 !=0:
        cfg_prune.append([cfg[i-1],cfg[i]])


print("cfg_prune",cfg_prune)
newmodel = ViT_slim(image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 512,
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    cfg=cfg_prune)


newmodel.to(device)
newmodel_dict = newmodel.state_dict().copy()

i = 0
newdict = {}
for k,v in model.state_dict().items():
    if 'net1.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'net1.0.bias' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'to_q' in k or 'to_k' in k or 'to_v' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[idx.tolist()].clone()
    elif 'net2.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1
    elif 'to_out.0.weight' in k:
        # print(k)
        # print(v.size())
        # print('----------')
        idx = np.squeeze(np.argwhere(np.asarray(cfg_mask[i].cpu().numpy())))
        newdict[k] = v[:,idx.tolist()].clone()
        i = i + 1

    elif k in newmodel.state_dict():
        newdict[k] = v

newmodel_dict.update(newdict)
newmodel.load_state_dict(newmodel_dict)

#torch.save(newmodel.state_dict(), 'pruned.pth')
print('after pruning: ', end=' ')
test(newmodel,True,cfg_prune)
