#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0

python dist_train.py \
    --config configs/train_skt.yaml \
    --name ntu120_xsub --tag init_smooth0.1 --port $MASTER_PORT