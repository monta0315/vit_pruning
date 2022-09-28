# Prune Vision Transformer

# Dataset

CIFAR10

# Create Evironment
- Use Venv

 ```
    python3 -m venv --system-site-packages ./ENVNAME
 ```
- Install Package By pip

 ```
    pip3 install -r requirements.txt
 ```

# Train ViT Model
- Train Base Model

 ```
    python3 train_cifar10.py 
 ```
- Prune Base Model
 ```
    python3 vitprune.py
 ```

- Compare Both Model

 ```
    python3 evaluate.py
 ```
  
  
 
