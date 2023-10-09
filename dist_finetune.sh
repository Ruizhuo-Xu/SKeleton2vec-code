TORCH_DISTRIBUTED_DEBUG=DETAIL \
CUDA_VISIBLE_DEVICES=3 \
python dist_train.py \
--config configs/fine_tune.yaml \
--name ntu60_xsub --tag fine_tune_test --port '12351' --enable_amp
