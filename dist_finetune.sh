TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=0 \
python dist_train.py \
--config configs/fine_tune.yaml \
--name ntu60_xsub --tag fine_tune_800EP_0.60ld_0.3dp_1e-4lr_5wp_0.08wd --port '12351' --enable_amp --compile
