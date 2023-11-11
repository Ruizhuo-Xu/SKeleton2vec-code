#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1
# wandb offline
# wandb online

python dist_pretrain2.py \
    --config configs/ntu120xsub_pretrain_skt2vec2.yaml \
    --name nturgbd120_xsub --tag pretrain_skt2vec2_128BSZ_3D_spati_${MASTER_PORT} --port $MASTER_PORT \
    --compile \
    --nodes 1 --node_rank 0

