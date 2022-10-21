import torch

#from models.slim_split import ViT_slim
from models.attn_importance_split_slim import ViT as ViT_slim
from models.select_split import ViT


class Utility():
    def __init__(self,name="newest",name2=None,strategy="each"):
        self.cfg = f"CIFAR10-100epochs-256bs-{strategy}"
        self.first_cfg = f"CIFAR10-100epochs-256bs"
        self.name = f"{name}-{self.cfg}"
        self.first_name = f"{name}-{self.first_cfg}"
        self.model_path = f"ch_sele_checkpoints/{name}-{self.cfg}.pth"
        self.pruned_model_path = f"pruned1_checkpoints/self-pruned-{name}-{self.cfg}.pth"
        self.pruned_2_1_model_path = f"pruned2_checkpoints/self-pruned-{name}-{self.cfg}.pth"
        self.pruned_2_2_model_path = f"pruned2_checkpoints/self-pruned-{name}-{self.cfg}-all.pth"


    def get_model(self):
        model = ViT(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,                  # 512
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            qkv_bias=True
            )

        model_path = self.model_path
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])


        pruned_model_path = self.pruned_model_path
        pruned_checkpoint = torch.load(pruned_model_path)
        pruned_model = ViT_slim(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,                  # 512
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            cfg=pruned_checkpoint['cfg'],
            qkv_bias=True
            )
        pruned_model.load_state_dict(pruned_checkpoint['net'])

        return model,pruned_model
    
    def get_model_for_comparing_two_pruned(self):
        pruned_2_1_checkpoint = torch.load(self.pruned_2_1_model_path)
        pruned_2_2_checkpoint = torch.load(self.pruned_2_2_model_path)
        pruned_2_1_model = ViT_slim(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,                  # 512
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            cfg=pruned_2_1_checkpoint['cfg'],
            qkv_bias=True
            )
        pruned_2_2_model = ViT_slim(
            image_size = 32,
            patch_size = 4,
            num_classes = 10,
            dim = 512,                  # 512
            depth = 6,
            heads = 8,
            mlp_dim = 512,
            dropout = 0.1,
            emb_dropout = 0.1,
            cfg=pruned_2_2_checkpoint['cfg'],
            qkv_bias=True
        )
        pruned_2_1_model.load_state_dict(pruned_2_1_checkpoint['net'])
        pruned_2_2_model.load_state_dict(pruned_2_2_checkpoint['net'])

        return pruned_2_1_model,pruned_2_2_model

    
    def get_model_path(self):
        return self.model_path,self.pruned_model_path
    
    def get_name(self):
        return self.name
        
    def get_first_name(self):
        return self.first_name
    
    def get_two_pruned_model_path(self):
            return self.pruned_2_1_model_path,self.pruned_2_2_model_path

    def debag(self,model):
        for k,v in model.items():
            print(k)
            print(v.shape)
