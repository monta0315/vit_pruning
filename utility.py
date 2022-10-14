import torch

#from models.slim_split import ViT_slim
from models.attn_importance_split_slim import ViT as ViT_slim
from models.select_split import ViT


class Utility():
    def __init__(self,name= "newest-CIFAR10-100epochs-256bs"):
        self.name = name
        self.model_path = f"ch_sele_checkpoints/{name}.pth"
        self.pruned_model_path = f"pruned1_checkpoints/self-pruned-{name}.pth"


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
    
    def get_model_path(self):
        return self.model_path,self.pruned_model_path
    
    def get_name(self):
        return self.name

    def debag(self,model):
        for k,v in model.items():
            print(k)
            print(v.shape)
