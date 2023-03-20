import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.utils

from dataset import DataLoaderTrainOfficialWarped
import options
from utils.dataset_utils import CutBlur
from utils.loader import get_training_data, get_validation_data

opt = options.Options().init(argparse.ArgumentParser(description='image denoising')).parse_args()
opt.cut_shadow_ratio = 1
opt.cut_shadow_ns_s_ratio = 0
opt.visualize = True
# opt.color_aug = True

row, col = 1, 6
batch_num = row * col
batch_num = 1

train_dir = 'datasets/official_warped/train'
train_dataset = DataLoaderTrainOfficialWarped(
    rgb_dir = train_dir, 
    img_options = {'patch_size': 1280},
    color_space='rgb',
    mask_dir='mask_v',
    opt=opt
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_num, shuffle=False, 
        num_workers=1, pin_memory=True, drop_last=False)

from torchvision.transforms.functional import to_pil_image, to_tensor

# aug = CutShadow()
output_dir = Path('viz_dataloader')
os.makedirs(output_dir, exist_ok=True)

cut_blur = CutBlur(p = opt.cut_shadow_ratio, 
                ns_s_ratio = opt.cut_shadow_ns_s_ratio, 
                sample_from_s = opt.sample_from_s,
                visualize = opt.visualize)

for i, data in enumerate(train_loader):
    clean, noisy_, mask, _, filenames, noisy = data

    # CutBlur再現
    if True:
        # 画像を縮小する
        scale_factor = 32.0
        tensor = F.interpolate(clean, scale_factor=1 / scale_factor, mode="nearest")
        # 画像を拡大する
        lr = F.interpolate(tensor, scale_factor=scale_factor, mode="nearest")

        noisy_, _ = cut_blur(clean.squeeze(0), lr.squeeze(0), mask.squeeze(0)) # cut shadow
        noisy_ = noisy_.unsqueeze(0)
        mask = torch.ones_like(mask)

        img = torchvision.utils.make_grid(
            torch.concat((clean, lr, noisy_, mask.expand(batch_num, 3, -1, -1)), 0), 
            nrow=col, 
            pad_value=1, 
            padding=30,
            mode="bilinear", 
            align_corners=False
        )
    
    # CutShadow
    elif True:
        img = torchvision.utils.make_grid(
            torch.concat((clean, noisy, noisy_, mask.expand(batch_num, 3, -1, -1)), 0), 
            nrow=col, 
            pad_value=1, 
            padding=30,
            mode="bilinear", 
            align_corners=False
        )

    # img = torchvision.utils.make_grid(torch.concat((noisy, clean, mask.expand(batch_num, 3, -1, -1)), 0), nrow=col)
    # img = torchvision.utils.make_grid(torch.concat((noisy, clean), 0), nrow=col)
    img = transforms.functional.to_pil_image(img)
    img.save(output_dir / f'cut_shadow_{filenames[0]}')
    print(f"save cut_shadow_{filenames[0]}")

    if i > 10:
        break
