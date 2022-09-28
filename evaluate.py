import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from utils.utility import Utility
from utils.utils import progress_bar


class Evaluate:
    def __init__(self) -> None:
        u = Utility()
        self.model,self.pruned_model = u.get_model()
        self.model_path,self.pruned_model_path = u.get_model_path()

    def get_img(self):
        img_path = "data/inference_img.png"
        if os.path.isfile(img_path):
            img = Image.open(img_path)
        else:
            img = Image.open(requests.get("https://raw.githubusercontent.com/pytorch/ios-demo-app/master/HelloWorld/HelloWorld/HelloWorld/image.png", stream=True).raw)
            img.save(img_path)

        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])

        return transform(img)[None,]



    def comparing_inference_speed(self,inference_times:int = 100):
        model,pruned_model = self.model,self.pruned_model
        img = self.get_img()

        with torch.autograd.profiler.profile(use_cuda=False) as prof1:
            for _ in range(inference_times):
                out = model(img)   
        
        with torch.autograd.profiler.profile(use_cuda=False) as prof2:
            for _ in range(inference_times):
                out = pruned_model(img)
    

        #print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
        #print("pruned model: {:.2f}ms".format(prof2.self_cpu_time_total/1000)) 

        df = pd.DataFrame({'Model': ['original model','pruned model']})
        df = pd.concat([df, pd.DataFrame([
            ["{:.2f}ms".format((prof1.self_cpu_time_total)/1000),"{:.2f}ms".format((prof1.self_cpu_time_total)/inference_times/1000), "0%"],
            ["{:.2f}ms".format((prof1.self_cpu_time_total)/1000),"{:.2f}ms".format((prof2.self_cpu_time_total)/inference_times/1000),
            "{:.2f}%".format((prof1.self_cpu_time_total-prof2.self_cpu_time_total)/prof1.self_cpu_time_total*100)]],
            columns=[f'{inference_times} Inference','Ave Inference', 'Reduction'])], axis=1)
        
        print(df)
        print("\n", end="") 
    
    def check_model_size(self):
        model_path,pruned_model_path = self.model_path,self.pruned_model_path

        with zipfile.ZipFile(f'{model_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(model_path) 

        original_model_size = os.path.getsize(f'{model_path}.zip')

        # torch.save(pruned_model.state_dict(), "pruned_cifar10_vgg16.pth.tar")

        with zipfile.ZipFile(f'{pruned_model_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(pruned_model_path) 

        pruned_model_size = os.path.getsize(f'{pruned_model_path}.zip')

        print(f'Size of the the Original Model: {original_model_size:,} bytes')
        print(f'Size of the the Pruned Model: {pruned_model_size:,} bytes')
        print(f'Compressed Percentage: {(100 - (pruned_model_size / original_model_size) * 100):.2f}%')
        print("\n", end="") 
    
    def total_params_count(self):
        model,pruned_model = self.model,self.pruned_model
        total_params_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
        pruned_model_params_count = sum(param.numel() for param in pruned_model.parameters() if param.requires_grad)
        print(f'Original Model parameter count: {total_params_count:,}')
        print(f'Pruned Model parameter count: {pruned_model_params_count:,}')
        print(f'Compressed Percentage: {(100 - (pruned_model_params_count / total_params_count) * 100):.2f}%')
        print("\n", end="") 



e = Evaluate()
print("==> Comparing inference speed..")
e.comparing_inference_speed(1000)
print("==> Comparing model size..")
e.check_model_size()
e.total_params_count()
