#!/bin/bash

# Training script for LALIC using flickr2k_lalic dataset (NOISE quantization mode)
# Lambda: 0.067

# Parameters:
# Batch size: 8
# Learning rate: 1e-4
# Epochs: 150
# Patch size: 256 256
# Seed: 100
# Lambda: 0.067
# Loss type: mse
# Quantizer: noise
# Dataset: /workspace/uchishiba_data/flickr2k_lalic
# Save path: /workspace/uchishiba_data/experiments/noise
# Scheduling: None (1000)

WANDB_PROJECT="lalic-original"
WANDB_RUN_NAME="0.067_noise"

# Optimization Settings
USE_AMP="true"
USE_BENCHMARK="true"
USE_DETERMINISTIC="false"
USE_TF32="true"

ARGS="--quantizer noise"

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

# CUDA_VISIBLE_DEVICES=0 to replace the STE run
CUDA_VISIBLE_DEVICES=0 python train.py \
    -d /workspace/uchishiba_data/flickr2k_lalic \
    --lambda 0.067 \
    --epochs 150 \
    --learning-rate 1e-4 \
    --lr_epoch 1000 \
    --batch-size 8 \
    --patch-size 256 256 \
    --seed 100 \
    --type mse \
    --save_path /workspace/uchishiba_data/experiments/noise \
    --cuda \
    --clip_max_norm 1.0 \
    --save \
    --project "${WANDB_PROJECT}" \
    --name "${WANDB_RUN_NAME}" \
    $ARGS
