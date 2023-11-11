#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# python dist_pretrain2.py \
#     --config configs/ntu60_xsub/pretrain_skt2vec2.yaml \
#     --name ntu60_xsub --tag pretrain_1e-3lr_1e-5minlr_0.1tau_8layers_dim2_${MASTER_PORT} --port $MASTER_PORT \
#     --compile \
#     --nodes 1 --node_rank 0

python dist_pretrain2.py \
    --config configs/pkuv2_xsub/pretrain_skt2vec2.yaml \
    --name pkuv2_xsub_9999ema --tag pretrain_1e-3lr_5e-4minlr_0.1tau_8layers_dim2_${MASTER_PORT} --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0

python dist_pretrain2.py \
    --config configs/pkuv2_xsub/pretrain_skt2vec2.yaml \
    --name pkuv2_xsub_999ema --tag pretrain_1e-3lr_5e-4minlr_0.1tau_8layers_dim2_${MASTER_PORT} --port $MASTER_PORT \
    --enable_amp --compile \
    --nodes 1 --node_rank 0
