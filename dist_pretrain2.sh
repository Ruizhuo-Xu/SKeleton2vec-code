#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_pretrain2.py \
    --config configs/ntu60_xsub/pretrain_skt2vec2.yaml \
    --name ntu60_xsub --tag pretrain_randomMasking --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0
