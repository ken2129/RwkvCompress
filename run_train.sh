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

WANDB_PROJECT="lalic-original"
WANDB_RUN_NAME="0.0018_ste"

# Optimization Settings
# Set to "true" to enable, "false" to disable
USE_AMP="true"
USE_BENCHMARK="true"
USE_DETERMINISTIC="false"
USE_TF32="true"

ARGS=""

if [ "$USE_AMP" = "true" ]; then
    ARGS="$ARGS --amp"
fi

if [ "$USE_BENCHMARK" = "true" ]; then
    ARGS="$ARGS --benchmark"
fi

if [ "$USE_DETERMINISTIC" = "false" ]; then
    ARGS="$ARGS --no-deterministic"
fi

if [ "$USE_TF32" = "true" ]; then
    ARGS="$ARGS --tf32"
fi

CUDA_VISIBLE_DEVICES=0 python train.py \
    -d /workspace/uchishiba_data/flickr2k_lalic \
    --lambda 0.0018 \
    --epochs 150 \
    --learning-rate 1e-4 \
    --lr_epoch 1000 \
    --batch-size 8 \
    --patch-size 256 256 \
    --seed 100 \
    --type mse \
    --save_path /workspace/uchishiba_data/experiments \
    --cuda \
    --clip_max_norm 1.0 \
    --save \
    --project "${WANDB_PROJECT}" \
    --name "${WANDB_RUN_NAME}" \
    $ARGS
