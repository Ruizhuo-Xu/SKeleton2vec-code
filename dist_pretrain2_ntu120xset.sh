#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1
# wandb offline

python dist_pretrain2.py \
    --config configs/ntu120xset_pretrain_skt2vec2.yaml \
    --name nturgbd120_xset --tag pretrain_skt2vec2_128BSZ_3D_1e-3baseLR_1e-5minLR_tube5_tau0.2_ema9999_800EP_${MASTER_PORT} --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0

