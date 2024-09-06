#!/usr/bin/env bash

# can be trained on two 24G 3090 GPUs

# chairs
CHECKPOINT_DIR=checkpoints/chairs-meflow && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 12 \
--val_dataset chairs sintel kitti \
--val_iters 12 \
--lr 4e-4 \
--image_size 352 480 \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

# things
CHECKPOINT_DIR=checkpoints/things-meflow && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--stage things \
--resume checkpoints/chairs-meflow/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 6 \
--val_dataset sintel kitti \
--val_iters 12 \
--lr 1.25e-4 \
--image_size 416 736 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

# sintel
CHECKPOINT_DIR=checkpoints/sintel-meflow && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--stage sintel \
--resume checkpoints/things-meflow/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 6 \
--val_dataset sintel kitti \
--val_iters 12 \
--lr 1.25e-4 \
--weight_decay 1e-5 \
--gamma 0.85 \
--image_size 368 960 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 100000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

# kitti
CHECKPOINT_DIR=checkpoints/kitti-meflow && \
mkdir -p ${CHECKPOINT_DIR} && \
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--stage kitti \
--resume checkpoints/sintel-meflow/step_100000.pth \
--no_resume_optimizer \
--checkpoint_dir ${CHECKPOINT_DIR} \
--batch_size 6 \
--val_dataset kitti \
--val_iters 12 \
--lr 1e-4 \
--weight_decay 1e-5 \
--gamma 0.85 \
--image_size 320 1024 \
--freeze_bn \
--summary_freq 100 \
--val_freq 10000 \
--save_ckpt_freq 5000 \
--save_latest_ckpt_freq 1000 \
--num_steps 50000 \
2>&1 | tee ${CHECKPOINT_DIR}/train.log

