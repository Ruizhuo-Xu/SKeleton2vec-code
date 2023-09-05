CUDA_VISIBLE_DEVICES=0,1 \
python dist_pretrain.py \
--config configs/pretrain_skt.yaml \
--name nturgbd60 --tag pretrain_test_bz64 --port '12355'
