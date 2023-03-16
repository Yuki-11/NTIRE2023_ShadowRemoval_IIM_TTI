# val
python test.py --cal_metrics \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --result_dir results/val_[] \
  --save_images \
  --gpus 

python test.py --cal_metrics \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --gpus 

# val mask変更
python test.py --cal_metrics \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --result_dir results/val_[] \
  --save_images \
  --mask_dir mask_v_mtmt \
  --gpus 

python test.py --cal_metrics \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --mask_dir mask_v_mtmt \
  --gpus 
# test
python inference.py \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --result_dir results/test_[] \
  --gpus 

# test mask変更
python inference.py \
  --weights log/ShadowFormer_[]/models/model_best.pth \
  --result_dir results/test_[] \
  --mask_dir mask_v_mtmt \
  --gpus 
