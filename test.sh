#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python test.py \
--config configs/ntu120_xset/test.yaml \
--port $MASTER_PORT --enable_amp