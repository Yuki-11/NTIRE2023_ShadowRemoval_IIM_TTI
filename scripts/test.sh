# val
python test.py --cal_metrics \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --result_dir results/val_[] \
  --save_images \
  --gpus 

python test.py --cal_metrics \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --gpus 


# test
python inference.py \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --result_dir results/test_[] \
  --gpus 