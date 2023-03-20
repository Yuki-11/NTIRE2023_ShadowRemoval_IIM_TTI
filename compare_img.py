import os
from pathlib import Path
import matplotlib.pyplot as plt 
import cv2
from tqdm import tqdm

dataset_path = Path('datasets/official_warped')
results_dir = Path('results')

# mask比較
compare_paths = [
    dataset_path / 'val' / 'input',
    dataset_path / 'val' / 'mask',
    dataset_path / 'val' / 'mask_v',
    # dataset_path / 'val' / 'mask_v_mtmt',
    # results_dir / 'val_set2_joint_learning1e-2_nomixup',
    # results_dir / 'val_set2_joint_learning1e-4_coloraug2_nomixup',
]
mask_or_not = [
    False, False, False,
    False, True, True,
]

# compare_paths = [
#     dataset_path / 'test_final' / 'input',
#     results_dir / 'test_final_set2_nomixup',
#     results_dir / 'test_final_set2_joint_learning1e-2_nomixup',
#     results_dir / 'test_final_set2_joint_learning1e-4_coloraug2_nomixup',
#     results_dir / 'test_final_set2_joint_learning1e-2_coloraug2_nomixup',
# ]
# # compare_paths = [
# #     dataset_path / 'test_final' / 'input',
# #     results_dir / 'test_final_set2_joint_learning_nomixup2',
# #     results_dir / 'test_final_w_val_set2_joint_learning_nomixup2',
# #     results_dir / 'test_final_set2_joint_learning_coloraug2_nomixup',
# #     results_dir / 'test_final_w_val_set2_joint_learning_coloraug2_nomixup',
# #     results_dir / 'test_final_set2_nomixup',
# # ]
# mask_or_not = [
#     False, False, False,
#     False, False, False,
# ]


output_dir = Path('compare/mask_line1')

# コピペ用
model_list = [
    dataset_path / 'test' / 'input',
    dataset_path / 'test_final' / 'input',
    dataset_path / 'test_final' / 'mask_v_mtmt',
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
    results_dir / 'test_final_dino1e6',
    results_dir / 'test_final_set1',
    results_dir / 'test_final_set1_nomixup',
    results_dir / 'test_final_set2',
    results_dir / 'test_final_set2_coloraug',
    results_dir / 'test_final_set2_nomixup',
    results_dir / 'test_final_set2_coloraug_continuous',
    results_dir / 'test_final_set2_coloraug2',
    results_dir / 'test_final_set2_joint_learning_nomixup',
    results_dir / 'test_final_dino1e7_cutshadow0.5_joint_learning_nomixup',
    results_dir / 'test_final_set2_joint_learning1e-2_nomixup',
    results_dir / 'test_final_set2_joint_learning_nomixup2',
    results_dir / 'test_final_set2_joint_learning1e-4_nomixup',
    results_dir / 'test_final_set2_joint_learning1e-5_nomixup',
    results_dir / 'test_final_set2_joint_learning1e-2_coloraug2_nomixup',
    results_dir / 'test_final_set2_joint_learning_coloraug2_nomixup',
    results_dir / 'test_final_set2_joint_learning1e-4_coloraug2_nomixup',
    results_dir / 'test_final_set2_joint_learning1e-5_coloraug2_nomixup',
    results_dir / 'test_final_w_val_set2_joint_learning_coloraug2_nomixup',
    results_dir / 'test_final_w_val_set2_joint_learning_nomixup2',
    results_dir / 'test_final_w_val_set2_nomixup',
    # results_dir / 'test_final_w_val_set2_joint_learning1e-4_nomixup',
    # results_dir / 'test_final_w_val_set2_joint_learning1e-2_nomixup',
    # results_dir / 'test_final_w_val_set2_joint_learning1e-4_coloraug2_nomixup',
    results_dir / '',
]


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

for i in tqdm(range(len(compare_files[0]))):
    img_list = []
    for j in range(len(compare_paths)):
        img = cv2.imread(str(compare_files[j][i]))
        # img = cv2.resize(img, None, None, 0.5, 0.5)
        img = cv2.copyMakeBorder(img, 20, 20, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255))

        img_list.append(img)
    result_img = cv2.hconcat(img_list)
    cv2.imwrite(str(output_dir / compare_files[0][i].name), result_img)


# for i in tqdm(range(len(compare_files[0]))):
#     img_list = []
#     plt.figure(figsize=(12*w, 9*h), dpi=120)
#     plt.title(str(i).zfill(4))
#     for j in range(len(compare_paths)):
#         # img = cv2.imread(str(compare_files[j][i]))
#         if mask_or_not[j]:
#             img = cv2.imread(str(compare_paths[j] / f"{compare_files[0][i].name}-mask_pred.png"))
#         else:
#             img = cv2.imread(str(compare_paths[j] / compare_files[0][i].name))
#         img = cv2.resize(img, None, None, 0.5, 0.5)
#         # img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#         plt.subplot(h, w, j + 1)
#         plt.title(compare_paths[j],  fontsize=fs)
#         plt.axis('off')
#         plt.imshow(img)
#     plt.savefig(str(output_dir / compare_files[0][i].name), bbox_inches='tight')
#     plt.clf()
#     plt.close()
