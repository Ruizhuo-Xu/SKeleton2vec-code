#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_pretrain2.py \
    --config configs/pretrain_skt2vec2.yaml \
    --name nturgbd60 --tag pretrain_skt2vec2_64BSZ_3D_2e-3baseLR_1e-4minLR_tube5_motion_aware_tau0.2 --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0

