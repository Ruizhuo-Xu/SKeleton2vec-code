CUDA_VISIBLE_DEVICES=3 \
python dist_train.py \
--config configs/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_v2_test-ep160 --port '12350'
