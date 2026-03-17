CUDA_VISIBLE_DEVICES=0 python infer.py \
  --batch_size 24 \
  --data_dir_2d ./data/test_img \
  --data_dir_2d_dev ./data/test_img \
  --img_length_read 6 \
  --ckpt_dir ./ckpts/model1 \
  --input_csv ./data/test_total.csv \
  --output_dir ./output_csv/model1 \
  --crop_size 512