#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024


# python dist_train.py \
# --config configs/ntu60_xsub/fine_tune.yaml \
# --name ntu60_xsub --tag fine_tune_bug_fix_teacher --port $MASTER_PORT --enable_amp --compile

python dist_train.py \
--config configs/ntu120_xsub/fine_tune.yaml \
--name ntu120_xsub --tag fine_tune_bug_fix_teacher --port $MASTER_PORT --enable_amp --compile