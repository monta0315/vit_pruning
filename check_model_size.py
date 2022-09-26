import os
import zipfile

from utility import Utility

u = Utility()
model_path,pruned_model_path = u.get_model_path()

def check_model_size(mp,pmp):
    with zipfile.ZipFile(f'{mp}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(mp) 

    original_model_size = os.path.getsize(f'{mp}.zip')

    # torch.save(pruned_model.state_dict(), "pruned_cifar10_vgg16.pth.tar")

    with zipfile.ZipFile(f'{pmp}.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(pmp) 

    pruned_model_size = os.path.getsize(f'{pmp}.zip')

    print(f'Size of the the Original Model: {original_model_size:,} bytes')
    print(f'Size of the the Pruned Model: {pruned_model_size:,} bytes')
    print(f'Compressed Percentage: {(100 - (pruned_model_size / original_model_size) * 100):.2f}%')

check_model_size(model_path,pruned_model_path)
