# dino
python train.py --warmup \
  --env _[] \
  --dino_lambda [] \
  --gpu 

# mask
python train.py --warmup \
  --env _set1_cut_shadow_ns_s_ratio0.5 \
  --cut_shadow_ratio 0.5 \
  --cut_shadow_ns_s_ratio 0.5
  --gpu 4,5

python train.py --warmup \
  --env _set1_whsv \
  --w_hsv \
  --gpu 1,2

python train.py --warmup \
  --env _test3 \
  --dino_lambda 1e6 \
  --cut_shadow
  --pretrain_weights log/ShadowFormer_dino1e6_cut_shadow0.5_maskvmtmt/models/model_latest.pth
  --resume
  --gpu 0,1

python train.py --warmup \
  --env _dino1e6_cut_shadow0.5_maskvmtmt \
  --dino_lambda 1e6 \
  --mask mask_v_mtmt \
  --cut_shadow
  --pretrain_weights log/ShadowFormer_dino1e6_cut_shadow0.5_maskvmtmt/models/model_250.pth
  --resume
  --gpu 0