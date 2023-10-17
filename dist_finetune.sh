TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=0,1 \
python dist_train.py \
--config configs/fine_tune.yaml \
--name ntu60_xsub --tag fine_tune_800EP_1e-4minLR_2e-3baseLR --port '12351' --enable_amp --compile
# TORCH_DISTRIBUTED_DEBUG=DETAIL \
# CUDA_VISIBLE_DEVICES=0,1 \
# python dist_train.py \
# --config configs/fine_tune.yaml \
# --name ntu60_xsub --tag fine_tune_800EP_0.65ld_0.15dp_3e-4lr_5wp_0.05wd_xavier_ --port '12351' --enable_amp --compile \
# --layer_decay 0.65