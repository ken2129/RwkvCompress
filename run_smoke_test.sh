#!/bin/bash

# Smoke test script for LALIC training
# Runs a single epoch with small batch size and disabled logging to verify functionality.

# Disable WandB for smoke test
export WANDB_MODE=disabled

echo "Running Smoke Test..."

CUDA_VISIBLE_DEVICES=0 python train.py \
    -d /workspace/uchishiba_data/flickr2k_lalic \
    --lambda 0.0018 \
    --epochs 1 \
    --learning-rate 1e-4 \
    --batch-size 2 \
    --test-batch-size 2 \
    --patch-size 256 256 \
    --seed 100 \
    --type mse \
    --save_path /workspace/uchishiba_data/experiments/smoke_test \
    --cuda \
    --clip_max_norm 1.0 \
    --save \
    --project "SMOKE_TEST" \
    --name "smoke_test_run"

echo "Smoke Test Completed."
