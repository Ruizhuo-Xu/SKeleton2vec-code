#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


python dist_train.py \
--config configs/ntu120_xset/fine_tune.yaml \
--name ntu120_xset --tag fine_tune_600EP_1e-4lr --port $MASTER_PORT --enable_amp --compile \
--save_path ./save/experiments/ntu120_xset/