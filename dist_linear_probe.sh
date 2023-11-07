#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

# python dist_train.py \
# --config configs/ntu60_xsub/linear_probe_.yaml \
# --name ntu60_xsub --tag linear_probe_randomClip --port $MASTER_PORT --enable_amp --compile

python dist_train.py \
--config configs/ntu60_xsub/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_teacher --port $MASTER_PORT --enable_amp --compile