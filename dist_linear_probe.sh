CUDA_VISIBLE_DEVICES=0,1 \
python dist_train.py \
--config configs/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_20 --port '12351'
