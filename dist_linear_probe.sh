#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_train.py \
--config configs/ntu60_xsub/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe --port $MASTER_PORT --enable_amp

export MASTER_PORT=$(($RANDOM % 20000 + 12000))
python dist_train.py \
--config configs/ntu60_xsub/fine_tune.yaml \
--name ntu60_xsub --tag fine_tune_800EP_1e-5minLR_3e-4baseLR_0.1LS_0.8LD_0.3dr_0.3DP_new --port $MASTER_PORT --enable_amp --compile