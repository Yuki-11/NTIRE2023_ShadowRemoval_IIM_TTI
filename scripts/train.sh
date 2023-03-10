# dino
python train.py --warmup \
  --env _[] \
  --dino_lambda [] \
  --gpu 

# mask
python train.py --warmup \
  --env _dino1e6_maskvmtmt2 \
  --dino_lambda 1e6 \
  --mask mask_v_mtmt
  --gpu 2,3