TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=0 \
python dist_train.py \
--config configs/fine_tune.yaml \
--name ntu60_xsub --tag fine_tune_0.20motion_0.75ld_0.3dp_0.2smooth --port '12351' --enable_amp
