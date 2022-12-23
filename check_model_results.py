import argparse
import csv
import os
from random import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#from models.select_split import ViT, channel_selection
#from models.slim_split import ViT_slim as ViT
from models.attn_importance_split_slim import ViT
from models.slim_split import channel_selection
from utils.utils import progress_bar

# parsers
parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
parser.add_argument("--model_path",default="pruned3_checkpoints/self-pruned-base-CIFAR10-100epochs-256bs-each-test-each.pth")

args = parser.parse_args()

checkpoint = torch.load(args.model_path,map_location="cpu")

print("acc:",checkpoint["acc"],"epoch:",checkpoint["epoch"],"rate:",checkpoint["rate"])