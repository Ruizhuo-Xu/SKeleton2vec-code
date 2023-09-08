CUDA_VISIBLE_DEVICES=1,2 \
python dist_train.py \
--config configs/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_v2_test --port '12353'
