# Prune Vision Transformer

# Dataset

CIFAR10

# Create Environment
- Use venv

 ```
    python3 -m venv --system-site-packages ./ENVNAME
 ```
- Install packages by pip

 ```
    pip3 install -r requirements.txt
 ```

# Train 
- Train Base Model

 ```
    python3 train_cifar10.py 
 ```
 
# Prune

- Prune BaseModel and Create PrunedModel
 ```
    python3 vitprune.py
 ```

# Evaluate

- Compare Both Model 
 ```
    python3 evaluate.py
 ```
  
  
 
