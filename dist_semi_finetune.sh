#!/usr/bin/env bash
set -x
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

# for((i=1;i<=5;i++)); 
# do
# python dist_train.py \
# --config configs/ntu60_xview/semi0.01_fine_tune.yaml \
# --name ntu60_xview_semi0.01 --tag fine_tune_{$i} \
# --port $MASTER_PORT --enable_amp --compile \
# --save_path save/experiments/ntu60_xview
# done

for((i=4;i<=5;i++)); 
do
python dist_train.py \
--config configs/ntu60_xview/semi0.1_fine_tune.yaml \
--name ntu60_xview_semi0.1 --tag fine_tune_{$i} \
--port $MASTER_PORT --enable_amp --compile \
--save_path save/experiments/ntu60_xview
done
