#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_pretrain2.py \
    --config configs/pkuv1_xsub/pretrain_skt2vec2.yaml \
    --name pkuv1_xsub --tag pretrain_1e-3lr_1e-4minlr --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0

