#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_train.py \
--config configs/ntu120_xsub/fine_tune.yaml \
--name ntu120_xsub --tag fine_tune_0.3dpr_1e-4lr --port $MASTER_PORT --enable_amp --compile