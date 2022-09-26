from utility import Utility

u = Utility()
m,pm = u.get_model()

def total_params_count(model,pruned_model):
    total_params_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    pruned_model_params_count = sum(param.numel() for param in pruned_model.parameters() if param.requires_grad)
    print(f'Original Model parameter count: {total_params_count:,}')
    print(f'Pruned Model parameter count: {pruned_model_params_count:,}')
    print(f'Compressed Percentage: {(100 - (pruned_model_params_count / total_params_count) * 100):.2f}%')

total_params_count(m,pm)
