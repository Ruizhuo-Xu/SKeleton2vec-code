CUDA_VISIBLE_DEVICES=3 \
python dist_train.py \
--config configs/train_skt.yaml \
--name ntu120_xsub_resume --tag init_smooth0.1 --port '12358'
