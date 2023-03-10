import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torchvision.transforms as transforms
from PIL import Image
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from utility import Utility
from utils.utils import progress_bar


class Evaluate:
    def __init__(self) -> None:
        u = Utility("newest","base",rate=0.7)
        self.first_pruned_model,self.second_pruned_model = u.get_model_for_comparing_two_pruned()
        self.first_pruned_model_path,self.second_pruned_model_path = u.get_two_pruned_model_path()
        self.base_model,_ = u.get_model()
        self.base_model_path,_ = u.get_model_path()
        print(self.first_pruned_model_path,self.second_pruned_model_path)


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



    def comparing_inference_speed(self,inference_times:int = 1000):
        base_model = self.base_model
        first_pruned_model,second_pruned_model = self.first_pruned_model,self.second_pruned_model
        img = self.get_img()

        with torch.autograd.profiler.profile(use_cuda=False) as prof1:
            for _ in range(inference_times):
                out = base_model(img)

        with torch.autograd.profiler.profile(use_cuda=True) as prof2:
            for _ in range(inference_times):
                out = first_pruned_model(img)
        
        with torch.autograd.profiler.profile(use_cuda=True) as prof3:
            for _ in range(inference_times):
                out = second_pruned_model(img)
    

        #print("original model: {:.2f}ms".format(prof1.self_cpu_time_total/1000))
        #print("pruned model: {:.2f}ms".format(prof2.self_cpu_time_total/1000))

        df = pd.DataFrame({'Model': ['original model','pruned only second method model','pruned combine model']})
        df = pd.concat([df, pd.DataFrame([
                ["{:.2f}ms".format((prof1.self_cpu_time_total)/1000),"{:.2f}ms".format((prof1.self_cpu_time_total)/inference_times/1000), "0%"],
                ["{:.2f}ms".format((prof2.self_cpu_time_total)/1000),"{:.2f}ms".format((prof2.self_cpu_time_total)/inference_times/1000),
                "{:.2f}%".format((prof1.self_cpu_time_total-prof2.self_cpu_time_total)/prof1.self_cpu_time_total*100)],
                ["{:.2f}ms".format((prof3.self_cpu_time_total)/1000),"{:.2f}ms".format((prof3.self_cpu_time_total)/inference_times/1000),
                "{:.2f}%".format((prof1.self_cpu_time_total-prof3.self_cpu_time_total)/prof1.self_cpu_time_total*100)]            
            ],
            columns=[f'{inference_times} Inference','Ave Inference', 'Reduction'])], axis=1)
        
        print(df)
        print("\n", end="")


    def comparing_accuracy(self):
        first_pruned_model_path,second_pruned_model_path = self.first_pruned_model_path,self.second_pruned_model_path
        base_model_path = self.base_model_path

        checkpoint = torch.load(base_model_path)
        first_pruned_model_checkpoint = torch.load(first_pruned_model_path)
        second_pruned_model_checkpoint = torch.load(second_pruned_model_path)

        original_acc = checkpoint["acc"]
        first_pruned_model_acc = first_pruned_model_checkpoint["acc"]
        second_pruned_model_acc = second_pruned_model_checkpoint["acc"]

        df = pd.DataFrame({'Model':["base model","pruned only second method model","pruned combine model"]})
        df = pd.concat([df,pd.DataFrame([
            [original_acc,"0%"],
            [first_pruned_model_acc,"{:.4f}%".format(first_pruned_model_acc-original_acc)],
            [second_pruned_model_acc,"{:.4f}%".format(second_pruned_model_acc-original_acc)]
        ],
        columns=["Accuracy", "Reduction"])],axis=1)

        print(df)
        print("\n", end="")
    
    def check_model_size(self):
        first_pruned_model_path,second_pruned_model_path = self.first_pruned_model_path,self.second_pruned_model_path
        first_pruned_model,second_pruned_model = self.first_pruned_model,self.second_pruned_model
        base_model = self.base_model
        base_model_path = self.base_model_path

        with zipfile.ZipFile(f'{base_model_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(base_model_path) 

        base_model_size = os.path.getsize(f'{base_model_path}.zip')
        os.remove(f'{base_model_path}.zip')

        with zipfile.ZipFile(f'{first_pruned_model_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(first_pruned_model_path) 

        first_pruned_model_size = os.path.getsize(f'{first_pruned_model_path}.zip')
        os.remove(f'{first_pruned_model_path}.zip')

        with zipfile.ZipFile(f'{second_pruned_model_path}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.write(second_pruned_model_path) 

        second_pruned_model_size = os.path.getsize(f'{second_pruned_model_path}.zip')
        os.remove(f'{second_pruned_model_path}.zip')

        total_params_count = sum(param.numel() for param in base_model.parameters() if param.requires_grad)
        first_pruned_model_params_count = sum(param.numel() for param in first_pruned_model.parameters() if param.requires_grad)
        second_pruned_model_params_count = sum(param.numel() for param in second_pruned_model.parameters() if param.requires_grad)

        df = pd.DataFrame({'Model':["original model","pruned only second method model","pruned combine method model"]})
        df = pd.concat([df,pd.DataFrame([
            [f'{base_model_size:,} bytes',"0%",total_params_count,"0%"],
            [f'{first_pruned_model_size/1000000:,} KB',f"{(100 - (first_pruned_model_size/1000000 / base_model_size/1000000) * 100):.2f}%",
             first_pruned_model_params_count,f'{(100 - (first_pruned_model_params_count / total_params_count) * 100):.2f}%'],
            [f'{second_pruned_model_size/1000000:,} KB',f"{(100 - (second_pruned_model_size/1000000 / base_model_size/1000000) * 100):.2f}%",
             second_pruned_model_params_count,f'{(100 - (second_pruned_model_params_count / total_params_count) * 100):.2f}%']
        ],
        columns=["Size","SizeReduction", "Params","ParamsReduction"])],axis=1)

        print(df)
        print("\n", end="")




e = Evaluate()
print("==> Comparing inference speed..")
e.comparing_inference_speed(50)
print("==> Comparing accuracy..")
e.comparing_accuracy()
print("==> Comparing model size..")
e.check_model_size()
