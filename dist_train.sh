#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_train_.py \
    --config configs/ntu60_xsub/fine_tune.yaml \
    --name ntu60_xsub --tag swa_test --port $MASTER_PORT --enable_amp --compile