# NTIRE2023 Image Shadow Removal Challenge Team IIM_TTI

This repository is a participation code for the Image Shadow Removal Challenge of the [NTIRE 2023 challenges](https://cvlai.net/ntire/2023/).


## Datasets
datasets were distributed to those who participated in the competition. [Competition page](https://codalab.lisn.upsaclay.fr/competitions/10253)

## Requirement
* Python>=3.7

```bash
pip install -r requirements.txt
```

## Inference

### 1. Arrange the final test data as follows:

```
datasets
└── official
    └── test_final
        └── input
            ├── 0000.png
            ├── 0001.png
            ├── ...
            └── 0099.png
```

### 2. Download weights.

[Download Link](https://drive.google.com/file/d/1I3oGi_ZlMoz5Zc4Mfm5tQY8N50Dh1VR7/view?usp=share_link)

Place the files in the 'weights' directory as follows:
```
weights/shadow_former+.pth
```

### 3. Run the inference.

```
python inference.py \
  --weights weights/shadow_former+.pth \
  --result_dir results/shadow_former+ \
  --joint_learning_alpha 1 \
  --gpus 0
```

The results will be output to results/shadow_former+.

## References
* [ShadowFormer](https://github.com/GuoLanqing/ShadowFormer)
* [MTMT](https://github.com/eraserNut/MTMT)