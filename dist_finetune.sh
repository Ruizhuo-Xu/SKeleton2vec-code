#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_train.py \
--config configs/fine_tune.yaml \
--name ntu60_xsub --tag fine_tune_800EP_1e-5minLR_3e-4baseLR --port $MASTER_PORT --enable_amp --compile