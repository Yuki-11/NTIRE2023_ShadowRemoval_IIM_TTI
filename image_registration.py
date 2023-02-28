#%%
import os
import sys
import shutil
import numpy as np
from pathlib import Path

import cv2
import torch
#%%

args = sys.argv

result_dir = Path('image_registration')
split_set = args[1]

#%%

gt_dir = Path(f'datasets/official/{split_set}/gt')
input_dir = Path(f'datasets/official/{split_set}/input')

gt_paths = list(gt_dir.iterdir())
input_paths = list(input_dir.iterdir())

#%%
# 差分画像出して，PSNR計算
# & reflect padding 実行

os.makedirs(result_dir / split_set / 'gt_pad', exist_ok=True)
os.makedirs(result_dir / split_set / 'diff', exist_ok=True)
psnr_list = []
print('PSNR')
for gt_path, input_path in zip(gt_paths, input_paths):
    img_gt = cv2.imread(str(gt_path))
    img_in = cv2.imread(str(input_path))

    # reflect padding
    img1_pad = np.pad(img_gt, ((100, ), (100, ), (0, )), 'reflect')
    cv2.imwrite(str(result_dir / split_set / 'gt_pad' / gt_path.name), img1_pad)

    # 差分とPSNR
    psnr = cv2.PSNR(img_gt, img_in)
    psnr_list.append(psnr)
    print(gt_path.name, ":", psnr)
    img_diff = cv2.absdiff(img_gt, img_in)
    cv2.imwrite(str(result_dir / split_set / 'diff' / gt_path.name), img_diff)
print(np.mean(psnr_list))

# %%
# 特徴点マッチング

gt_pad_dir = result_dir / split_set / 'gt_pad'
gt_pad_paths = list(gt_pad_dir.iterdir())

os.makedirs(result_dir / split_set / 'matches', exist_ok=True)
os.makedirs(result_dir / split_set / 'warped', exist_ok=True)
for gt_path, input_path in zip(gt_pad_paths, input_paths):
    if os.path.exists(result_dir / split_set / 'warped' / gt_path.name): continue
    img_gt = cv2.imread(str(gt_path))
    img_in = cv2.imread(str(input_path))

    akaze = cv2.AKAZE_create()
    kp_gt, desc_gt = akaze.detectAndCompute(img_gt, None)
    kp_in, desc_in = akaze.detectAndCompute(img_in, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_gt, desc_in, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append([m])

    matches_img = cv2.drawMatchesKnn(
        img_gt, kp_gt,
        img_in, kp_in,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(result_dir / split_set / 'matches' / gt_path.name), matches_img)

    if len(good_matches) >= 4:
        # 適切なキーポイントを選択
        ref_matched_kpts = np.float32(
            [kp_gt[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        sensed_matched_kpts = np.float32(
            [kp_in[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # ホモグラフィを計算
        H, status = cv2.findHomography(
            ref_matched_kpts, sensed_matched_kpts, cv2.RANSAC, 5.0)

        # 画像を変換
        warped_image = cv2.warpPerspective(
            img_gt, H, (img_gt.shape[1], img_gt.shape[0]))
        h, w, _ = img_in.shape
        warped_image = warped_image[:h, :w]
        cv2.imwrite(str(result_dir / split_set / 'warped' / gt_path.name), warped_image)
    else:
        shutil.copyfile(gt_dir / gt_path.name, result_dir / split_set / 'warped' / gt_path.name)

# %%
# 差分画像出して，PSNR計算

warped_dir = result_dir / split_set / 'warped'
warped_paths = list(warped_dir.iterdir())
output_dir = Path(f'datasets/official_warped/{split_set}') / 'gt'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(result_dir / split_set / 'diff_warped', exist_ok=True)

psnr_list = []
print('PSNR')
for warped_path, input_path in zip(warped_paths, input_paths):
    img_gt = cv2.imread(str(warped_path))
    img_in = cv2.imread(str(input_path))

    # 差分とPSNR
    psnr = cv2.PSNR(img_gt, img_in)
    psnr_list.append(psnr)
    print(warped_path.name, ":", psnr)
    img_diff = cv2.absdiff(img_gt, img_in)
    cv2.imwrite(str(result_dir / split_set / 'diff_warped' / warped_path.name), img_diff)
print(np.mean(psnr_list))


# %%
