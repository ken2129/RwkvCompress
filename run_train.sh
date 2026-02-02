#!/bin/bash

# Training script for LALIC using flickr2k_lalic dataset

# Parameters:
# Batch size: 8
# Learning rate: 1e-4
# Epochs: 150
# Patch size: 256 256
# Seed: 100
# Lambda: 0.0018
# Loss type: mse
# Dataset: /workspace/uchishiba_data/flickr2k_lalic
# Save path: /workspace/uchishiba_data/experiments
# Scheduling: None

WANDB_PROJECT="LALIC"
WANDB_RUN_NAME="test_run"

CUDA_VISIBLE_DEVICES=0 python train.py \
    -d /workspace/uchishiba_data/flickr2k_lalic \
    --lambda 0.0018 \
    --epochs 150 \
    --learning-rate 1e-4 \
    --batch-size 8 \
    --patch-size 256 256 \
    --seed 100 \
    --type mse \
    --save_path /workspace/uchishiba_data/experiments \
    --cuda \
    --clip_max_norm 1.0 \
    --save \
    --project "${WANDB_PROJECT}" \
    --name "${WANDB_RUN_NAME}"
