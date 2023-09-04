CUDA_VISIBLE_DEVICES=3 \
python dist_train.py \
--config configs/train_skt.yaml \
--name nturgbd60 --tag lrcos-drp0.0-nodp-adamw-noinit-smooth0.1-posRandn_ --port '12358'
