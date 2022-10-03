import torch

from models.attn_importance_score_model_test import ViT as attn_ViT


def get_dict():
    base_path = "checkpoint/add_attn-CIFAR10-2epochs-256bs.pth"

    checkpoint = torch.load(base_path, map_location='cpu')

    dict = checkpoint['net']

    return dict

def get_dict2():
        net = attn_ViT(
        image_size = 32,num_classes=10,qkv_bias=True,
        patch_size=4, dim=512, depth=6, heads=8, mlp_dim=512, reduce = 0, ind=0)

        dict = net.state_dict()

        return dict

for k,v in get_dict2().items():
    print(k)
    print(v.shape)

