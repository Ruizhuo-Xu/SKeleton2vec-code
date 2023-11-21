#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


python dist_train.py \
--config configs/pkuv2_xsub/fine_tune.yaml \
--name pkuv2_xsub --tag transfer_ntu120_xsub --port $MASTER_PORT --enable_amp \
--save_path ./save/experiments/pkuv2_xsub/