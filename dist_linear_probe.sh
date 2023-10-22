#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=3

python dist_train.py \
--config configs/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_test_ --port $MASTER_PORT --enable_amp
