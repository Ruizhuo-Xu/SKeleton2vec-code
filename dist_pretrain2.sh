#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_pretrain2.py \
    --config configs/pretrain_skt2vec2.yaml \
    --name nturgbd60 --tag pretrain_skt2vec2_64BSZ_3D_1e-3baseLR_1e-4minLR_tube5_tau0.2_ema990_125K --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0

