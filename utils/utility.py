import torch
from models.vit_select import ViT
from models.vit_slim import ViT_slim


class Utility():
    def __init__(self,name= "vit-CIFAR10-100epochs-256bs"):
        self.name = name
        self.model_path = f"checkpoint/{name}.pth"
        self.pruned_model_path = f"checkpoint/self-pruned-{name}.pth"


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
            emb_dropout = 0.1
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
            cfg=pruned_checkpoint['cfg']
            )
        pruned_model.load_state_dict(pruned_checkpoint['net'])

        return model,pruned_model
    
    def get_model_path(self):
        return self.model_path,self.pruned_model_path
    
    def get_name(self):
        return self.name
