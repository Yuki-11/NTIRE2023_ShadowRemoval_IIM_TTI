import os
import sys
import shutil
import argparse
import numpy as np
from pathlib import Path

import cv2
import torch

parser = argparse.ArgumentParser(description='RGB denoising evaluation on the validation set of SIDD')
parser.add_argument('--first_dir', default='results/test_m_diff2',
    type=str, help='Directory to compare(first)')
parser.add_argument('--second_dir', default='results/test_warped_model_finish',
    type=str, help='Directory to compare(second)')
parser.add_argument('--output_dir', default='',
    type=str, help='Output directory(default: {--first_dir}/diff_{--second_dir})')
args = parser.parse_args()

first_dir = Path(args.first_dir)
second_dir = Path(args.second_dir)
if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    output_dir = first_dir / f'diff_{second_dir.name}'
os.makedirs(output_dir, exist_ok=True)

# 差分画像出して，PSNR計算

psnr_list = []
print('PSNR')
for first_path, second_path in zip(first_dir.iterdir(), second_dir.iterdir()):
    img_first = cv2.imread(str(first_path))
    img_second = cv2.imread(str(second_path))

    # 差分とPSNR
    psnr = cv2.PSNR(img_first, img_second)
    psnr_list.append(psnr)
    print(first_path.name, ":", psnr)
    img_diff = cv2.absdiff(img_first, img_second)
    cv2.imwrite(str(output_dir / first_path.name), img_diff)
print(np.mean(psnr_list))