CUDA_VISIBLE_DEVICES=1 \
python dist_train.py \
--config configs/train_skt.yaml \
--name nturgbd60 --tag lrcos-drp0.0-nodp-adamw-init-smooth0.1-posRandn --port '12357'
