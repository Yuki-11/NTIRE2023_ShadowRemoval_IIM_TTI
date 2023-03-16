import os
from pathlib import Path
import matplotlib.pyplot as plt 
import cv2
from tqdm import tqdm

dataset_path = Path('datasets/official_warped')
results_dir = Path('results')
compare_paths = [
    dataset_path / 'test' / 'input',
    results_dir / 'test_seam_1',
    results_dir / 'test_dino1e6_cut_shadow0.5_maskvmtmt',
    results_dir / 'test_dino1e6_cut_shadow0.5_maskvmtmt_nomixup_fixed', 
    results_dir / 'test_set1_cut_shadow_ns_s_ratio0.5',
    results_dir / 'test_set2',
    # dataset_path / 'test' / 'mask_v_mtmt',    # mask
]

output_dir = Path('compare/set1-base-2')

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
    results_dir / 'test_maskvmtmt',
    results_dir / 'test_dino1e5',
    results_dir / 'test_seam_1',
    results_dir / 'test_set1_w_hsv',
    results_dir / 'test_dino1e6_cut_shadow0.5_maskvmtmt', 
    results_dir / 'test_dino1e6_cut_shadow0.5_maskvmtmt_nomixup_fixed', 
    results_dir / 'test_set1_cut_shadow_ns_s_ratio0.5',
    results_dir / 'test_set2',
    # results_dir / 'test_dino1e6_cut_shadow',
]

# compare_img_list = [
#     0,
#     5,
#     13,
#     16,
#     19,
#     28,
# ]

w, h = 3, 1
fs = 18
while len(compare_paths) > w * h:
    if (w + h)%2 == 0:
        h += 1
    else:
        w += 1


os.makedirs(output_dir, exist_ok=True)
assert len(list(output_dir.iterdir())) <= 2, "上書き防止assertion"
with open(output_dir / 'memo.txt', 'w') as f:
    f.write('compare list\n')
    for path in compare_paths:
        f.write(f"* {path}\n")

compare_files = []
for path in compare_paths:
    compare_files.append(sorted(list(path.iterdir())))

# for i in tqdm(range(len(compare_files[0]))):
#     img_list = []
#     for j in range(len(compare_paths)):
#         img = cv2.imread(str(compare_files[j][i]))
#         img = cv2.resize(img, None, None, 0.5, 0.5)
#         img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))

#         # # ここでテキストを追加
#         # font = cv2.FONT_HERSHEY_SIMPLEX
#         # text = compare_paths[j].name
#         # org = (50, 50) # テキストの左下座標
#         # fontScale = 1
#         # color = (255, 0, 0) # BGR形式で色を指定
#         # thickness = 2 # テキストの太さ
#         # img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

#         img_list.append(img)
#     result_img = cv2.hconcat(img_list)
#     cv2.imwrite(str(output_dir / compare_files[0][i].name), result_img)


for i in tqdm(range(len(compare_files[0]))):
    img_list = []
    plt.figure(figsize=(12*w, 9*h), dpi=120)
    plt.title(str(i).zfill(4))
    for j in range(len(compare_paths)):
        img = cv2.imread(str(compare_files[j][i]))
        img = cv2.resize(img, None, None, 0.5, 0.5)
        # img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.subplot(h, w, j + 1)
        plt.title(compare_paths[j],  fontsize=fs)
        plt.axis('off')
        plt.imshow(img)
    plt.savefig(str(output_dir / compare_files[0][i].name), bbox_inches='tight')
    plt.clf()
    plt.close()
