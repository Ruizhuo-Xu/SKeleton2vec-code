TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=0,1 \
python dist_pretrain2.py \
--config configs/pretrain_skt2vec2.yaml \
--name nturgbd60 --tag pretrain_skt2vec2_mask0.90_largeEMA_tube5_randomcrop --port '12355' --enable_amp --compile
