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
 
# Prune 1

- Prune BaseModel and Create PrunedModel
 ```
    python3 vitprune.py
 ```

# Prune 2

- Culc importance scores
 ```
    python3 split_culc_attn_importance_scores.py
 ```
- Prune by impotance scores
 ```
    python3 pruned_by_importance_score_all.py
 ``` 

# Evaluate

- Compare both base_model and pruned_model
 ```
    python3 evaluate.py
 ```
- Compare both pruned_model
 ```
    python3 evaluate_two_pruned_model.py
 ```
  
  
 
