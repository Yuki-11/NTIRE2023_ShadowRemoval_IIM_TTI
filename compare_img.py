import os
from pathlib import Path

import cv2
from tqdm import tqdm

dataset_path = Path('datasets/official_warped')
results_dir = Path('results')
compare_paths = [
    dataset_path / 'test' / 'input',
    results_dir / 'test_dino1e6_latest',
    results_dir / 'test_dino1e6_maskvmtmt',
]

# コピペ用
model_list = [
    dataset_path / 'test' / 'input',
    results_dir / 'test_vanilla',
    results_dir / 'test_warped_model_finish',
    results_dir / 'test_m_shadow1',
    results_dir / 'test_m_diff2',
    results_dir / 'test_m_shadow_m_diff1',
    results_dir / 'test_dino1e6_latest',
    results_dir / 'test_dino1e7',
    results_dir / 'test_hsv',
    results_dir / 'test_dino1e6_maskvmtmt',
    results_dir / 'test_dino1e6_self_rep1',
    results_dir / 'test_self_rep1',
    # results_dir / 'test_dino1e6_cut_shadow',
    # results_dir / 'test_maskvmtmt2',
]

# compare_img_list = [
#     0,
#     5,
#     13,
#     16,
#     19,
#     28,
# ]

output_dir = Path('compare/dino1e6-dino1e6_maskvmtmt')
os.makedirs(output_dir, exist_ok=True)
assert len(list(output_dir.iterdir())) <= 2, "上書き防止assertion"
with open(output_dir / 'memo.txt', 'w') as f:
    f.write('compare list\n')
    for path in compare_paths:
        f.write(f"* {path}\n")

compare_files = []
for path in compare_paths:
    compare_files.append(list(path.iterdir()))

for i in tqdm(range(len(compare_files[0]))):
    img_list = []
    for j in range(len(compare_paths)):
        img = cv2.imread(str(compare_files[j][i]))
        # img = cv2.resize(img, None, None, 0.5, 0.5)
        img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img_list.append(img)
    result_img = cv2.hconcat(img_list)
    cv2.imwrite(str(output_dir / compare_files[0][i].name), result_img)
