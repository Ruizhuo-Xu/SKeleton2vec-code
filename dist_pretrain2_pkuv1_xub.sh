#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_pretrain2.py \
    --config configs/pkuv1_xsub/pretrain_skt2vec2.yaml \
    --name pkuv1_xsub --tag pretrain_skt2vec2_64BSZ_3D_1e-3baseLR_1e-5minLR_tube5_tau0.4_ema999_800EP --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0

