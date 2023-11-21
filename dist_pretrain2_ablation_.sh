#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

export MASTER_PORT=$(($RANDOM % 20000 + 12000))
python dist_pretrain2.py \
    --config configs/ntu60_xsub/pretrain_skt2vec2.yaml \
    --name ntu60_xsub --tag pretrain_mask0.80_${MASTER_PORT} --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0 \
    --beta 0.1 --alpha 5 --mask_ratio 0.8 \
    --save_path ./save/ablation

export MASTER_PORT=$(($RANDOM % 20000 + 12000))
python dist_pretrain2.py \
    --config configs/ntu60_xsub/pretrain_skt2vec2.yaml \
    --name ntu60_xsub --tag pretrain_mask0.85_${MASTER_PORT} --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0 \
    --beta 0.1 --alpha 5 --mask_ratio 0.85 \
    --save_path ./save/ablation