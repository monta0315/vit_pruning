from random import shuffle

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

from models.vit import ViT, channel_selection
from models.vit_slim import ViT_slim

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
model = model.to(device)

model_path = "checkpoint/vit-4-ckpt.t7"
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

print("thre",thre)

pruned = 0
cfg = []
cfg_mask = []
for k,m in enumerate(model.modules()):
    if isinstance(m,channel_selection):
        print("what's k",k)
        print("m",m)
        if k in [16,40,64,88,112,136]:
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
print(cfg)


def test(model):
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
