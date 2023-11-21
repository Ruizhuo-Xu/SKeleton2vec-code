#!/usr/bin/env bash
export MASTER_PORT=$(($RANDOM % 20000 + 12000))
set -x
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export CUDA_VISIBLE_DEVICES=0,1

python dist_train.py \
--config configs/ntu60_xsub/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_EMA99999_400EP_${MASTER_PORT} --port $MASTER_PORT --enable_amp --compile \
--save_path ./save/experiments/ntu60_xsub/EMA \
--ckp checkpoints/ntu60_xsub/ntu60_xsub_pretrain_1e-3lr_1e-5minlr_0.1tau_8layers_dim2_99999EMA_600EP_15915/epoch-400.pth

python dist_train.py \
--config configs/ntu60_xsub/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_EMA99999_500EP_${MASTER_PORT} --port $MASTER_PORT --enable_amp --compile \
--save_path ./save/experiments/ntu60_xsub/EMA \
--ckp checkpoints/ntu60_xsub/ntu60_xsub_pretrain_1e-3lr_1e-5minlr_0.1tau_8layers_dim2_99999EMA_600EP_15915/epoch-500.pth

python dist_train.py \
--config configs/ntu60_xsub/linear_probe.yaml \
--name ntu60_xsub --tag linear_probe_EMA99_500EP_${MASTER_PORT} --port $MASTER_PORT --enable_amp --compile \
--save_path ./save/experiments/ntu60_xsub/EMA \
--ckp checkpoints/ntu60_xsub/ntu60_xsub_pretrain_1e-3lr_1e-5minlr_1.0tau_8layers_dim2_99ema_600EP_24242/epoch-500.pth