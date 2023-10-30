#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

for((i=1;i<=5;i++)); 
do
python dist_train.py \
--config configs/ntu60_xsub/semi0.1_fine_tune.yaml \
--name ntu60_xsub_semi0.1 --tag fine_tune_800EP_1e-5minLR_3e-4baseLR_0.1LS_0.8LD_0.3dr_0.3DP_0.2rt_{$i} \
--port $MASTER_PORT --enable_amp --compile
done