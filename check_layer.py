import torch

# from models.vit_select import ViT
from models.attn_importance_split import ViT

# from models.attn_importance_score_model_test import ViT
# from models.vit import ViT


def get_dict():
    base_path = "checkpoint/self-pruned-vit-CIFAR10-5epochs-64bs.pth"

    checkpoint = torch.load(base_path, map_location="cpu")

    dict = checkpoint["net"]

    return dict


def get_dict2():
    cfg = [376, 392, 360, 368, 336, 400]
    net = ViT(
        image_size=32,
        patch_size = 4,
        num_classes=10,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=512,
        dropout=0.1,
        emb_dropout=0.1,
        qkv_bias=True
    )

    dict = net.state_dict()

    return dict


for k, v in get_dict().items():
    print(k)
    print(v.shape)
