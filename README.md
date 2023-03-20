# NTIRE2023 Image Shadow Removal Challenge Team IIM_TTI

This repository is a participation code for the Image Shadow Removal Challenge of the [NTIRE 2023 challenges](https://cvlai.net/ntire/2023/).

## ToDo

* weightsのダウンロードリンク
* 動作確認
  * 環境構築(requirements.txt, requirements2.txt)
* referencesのライセンス記述

## Datasets
datasets were distributed to those who participated in the competition. [Competition page](https://codalab.lisn.upsaclay.fr/competitions/10253)

## Requirement
* Python>=3.7

[To Do 環境構築確認]

```bash
pip install -r requirements.txt
```

## Inference

### 1. Arrange the final test data as follows:

```
datasets
└── official
    └── test_final
        ├── 0000.png
        ├── 0001.png
        ├── ...
        └── 0099.png
```

### 2. Download weights.

[Download Link]()

Place the files in the 'weights' directory as follows:
```
weights/model_best.pth
```

### 3. Run the inference.

```
python inference.py \
  --weights weights/model_best.pth \
  --result_dir results/model_best \
  --joint_learning_alpha 1 \
  --gpus 0
```

The results will be output to results/model_best.

## References
* ShadowFormer
* MTMT
* DiNO