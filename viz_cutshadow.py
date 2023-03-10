#%%
import os
import argparse
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils

from dataset import DataLoaderTrainOfficialWarped
import options
from utils.loader import get_training_data, get_validation_data

opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()

row, col = 1, 4
num = row * col

train_dir = 'datasets/official_warped/train'
train_dataset = DataLoaderTrainOfficialWarped(
    rgb_dir = train_dir, 
    img_options = {'patch_size': 640},
    color_space='rgb',
    mask_dir='mask_v_mtmt',
    opt=opt
)
train_loader = DataLoader(dataset=train_dataset, batch_size=num, shuffle=False, 
        num_workers=1, pin_memory=True, drop_last=False)

from torchvision.transforms.functional import to_pil_image, to_tensor

# aug = CutShadow()
output_dir = Path('viz_cutshadow')
os.makedirs(output_dir, exist_ok=True)

for i, data in enumerate(train_loader):
    clean, noisy, mask, _, filenames, _ = data
    # filename = filenames[0]

    # img = torchvision.utils.make_grid(noisy, nrow=col)
    # img = transforms.functional.to_pil_image(img)
    # img.save(output_dir / f'cut_shadow_img{filenames[0]}')
    # img = torchvision.utils.make_grid(mask, nrow=col)
    # img = transforms.functional.to_pil_image(img)
    # img.save(output_dir / f'cut_shadow_mask{filenames[0]}')
    img = torchvision.utils.make_grid(torch.concat((noisy, mask.expand(num, 3, -1, -1)), 0), nrow=col)
    img = transforms.functional.to_pil_image(img)
    img.save(output_dir / f'cut_shadow_{filenames[0]}')

    # clean = clean.squeeze(dim = 0)
    # noisy = noisy.squeeze(dim = 0)
    # mask = mask.squeeze(dim = 0)

    # mix, mask = aug(clean, noisy, mask)
    
    # clean = to_pil_image(clean)
    # noisy = to_pil_image(noisy)
    # # mix = to_pil_image(mix)
    # mask = to_pil_image(mask)
    
    # noisy.save(output_dir / f'input_{filename}')
    # mask.save(output_dir / f'mask_{filename}')
    
    if i > 10:
        break
# %%
