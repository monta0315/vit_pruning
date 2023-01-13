import torch

#from models.slim_split import ViT_slim
from models.attn_importance_split_slim import ViT as ViT_slim
#from models.select_split import ViT
from models.split_vit import ViT


class Utility():
    def __init__(self,name="newest",name2=None,strategy="each",rate="0.4"):
        self.cfg = f"CIFAR10-100epochs-256bs"
        self.first_cfg = f"CIFAR10-100epochs-256bs"
        self.name = f"{name}-{self.cfg}"
        self.first_name = f"{name}-{self.first_cfg}"
        #self.model_path = f"ch_sele_checkpoints/{name}-{self.cfg}.pth"
        self.model_path = f"pruned1_checkpoints/self-pruned-newest-CIFAR10-100epochs-256bs-each-1.0.pth"
        #self.pruned_model_path = f"pruned1_checkpoints/self-pruned-{name}-{self.cfg}.pth"
        self.pruned_model_path = f"pruned1_checkpoints/self-pruned-newest-CIFAR10-100epochs-256bs-all-0.3.pth"
        #self.pruned_2_1_model_path = f"pruned2_checkpoints/self-pruned-{name}-{self.cfg}.pth"
        self.pruned_2_1_model_path = f"pruned3_checkpoints/self-pruned-base-CIFAR10-100epochs-256bs-each-test-last-{rate}.pth"
        #self.pruned_2_1_model_path = f"pruned2_checkpoints/self-pruned-base-CIFAR10-100epochs-256bs-each-test-each-{rate}.pth"
        #self.pruned_2_2_model_path = f"pruned2_checkpoints/self-pruned-{name}-{self.cfg}-all.pth"
        self.pruned_2_2_model_path = f"pruned1_checkpoints/self-pruned-newest-CIFAR10-100epochs-256bs-all-{rate}.pth"
        #self.pruned_2_2_model_path = f"pruned3_checkpoints/self-pruned-base-CIFAR10-100epochs-256bs-each-test-all-{rate}.pth"
        self.strategy = strategy


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
        print("each",pruned_2_1_checkpoint['cfg'],pruned_2_1_checkpoint['rate'])
        print("all",pruned_2_2_checkpoint['cfg'],pruned_2_2_checkpoint['rate'])
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
    def strategy(self):
        return self.strategy
    
    def txt_impotance_scores_convert_array(self,name,ind):
        importance_score_lists = []
        with open(f"importances/self-pruned-{name}/block_{ind}.txt",'r') as f:
            f.seek(0, os.SEEK_END)
            isempty = f.tell() == 0
            f.seek(0)
            step = []
            if not isempty:
                for i in f:
                    index,soft_score,hard_score,q,k,v = i[:-1].split(',')
                    step.append(index,float(soft_score),float(hard_score))
            else:
                print(f"block_{i} is empty file")
            
            importance_score_lists.append(step)
        
        return importance_score_lists
