CUDA_VISIBLE_DEVICES=0,1 \
python dist_train.py \
--config configs/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_800EP_1e-4minLR_2e-3baseLR --port '12350' --enable_amp
