#!/bin/bash
# =============================================================================
# demo.sh — CC_discover Quick-Start Demo
#
# Runs a full causal discovery experiment on the bundled demo dataset.
# Results are saved to:  ./test_results/<YYYYMMDD>/demo_dataset/
#
# Usage:
#   bash demo.sh                    # run with the default settings below
#   bash demo.sh --seed_iter 3      # override any argument on the command line
#   bash demo.sh --d_in 8 --seq_len 20 --batch_size 64
#
# To add a 4th (verification) split and enable Phase-2 ablation study:
#   bash demo.sh --ratios 0.6,0.1,0.2,0.1
#
# To evaluate against a ground-truth causal graph:
#   bash demo.sh --gt_path gt_causal.csv --gt_with_lag True
# =============================================================================

set -euo pipefail   # Exit immediately on error; treat unset variables as errors

# Any extra arguments passed to this script are forwarded to run_main.py unchanged
EXTRA_ARGS="$@"

# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------
IS_TRAINING=1           # 1 = train + test;  0 = test only (loads checkpoint)
MODEL=CC_discover
MODEL_ID=demo_dataset

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA=wdxcjnb1
ROOT_PATH=./dataset/demo/
DATA_PATH=demo_dataset.csv
RATIOS=0.6,0.1,0.3      # train / val / test split ratios (must sum to 1)
                         # Use 4 parts (e.g. 0.6,0.1,0.2,0.1) to enable Phase-2

# ---------------------------------------------------------------------------
# Sequence lengths
# ---------------------------------------------------------------------------
SEQ_LEN=10    # look-back window (number of samples)
PRED_LEN=1    # forecast horizon (number of samples)

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
D_IN=4          # number of input channels (variables); auto-detected if omitted
D_MODEL=256     # hidden (d_model) dimension
D_FF=512        # feed-forward network dimension
N_HEADS=16      # number of attention heads
D_LAYERS=3      # number of decoder layers
KERNEL_SIZE=3   # causal-conv kernel size

# ---------------------------------------------------------------------------
# Regularization / causal loss
# ---------------------------------------------------------------------------
LAMBDA_CH=0.05        # channel-gate sparsity weight
LAMBDA_AGC=0.05       # AGC (Attention-Gradient Consistency) loss weight
CUMULATIVE_RATIO=0.95 # keep the top edges covering this fraction of total strength

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE=128
TRAIN_EPOCHS=200
PATIENCE=10           # early-stopping patience (epochs without improvement)
LEARNING_RATE=0.001
LRADJ=TST             # learning-rate schedule: TST (OneCycleLR) | type1 | fixed
SEED_ITER=10          # number of independent random seeds to average over

# ---------------------------------------------------------------------------
# Output / visualization
# ---------------------------------------------------------------------------
SHOW_LAYER_IDX=-1     # decoder layer to visualize (-1 = last layer)
# Add --save_seed_plots below to also keep per-seed plots (default: SeedAvg only)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
python -u ./run_main.py \
    --is_training       "$IS_TRAINING"    \
    --model             "$MODEL"          \
    --model_id          "$MODEL_ID"       \
    --data              "$DATA"           \
    --root_path         "$ROOT_PATH"      \
    --data_path         "$DATA_PATH"      \
    --ratios            "$RATIOS"         \
    --scale             True              \
    --seq_len           "$SEQ_LEN"        \
    --pred_len          "$PRED_LEN"       \
    --d_in              "$D_IN"           \
    --d_model           "$D_MODEL"        \
    --d_ff              "$D_FF"           \
    --n_heads           "$N_HEADS"        \
    --d_layers          "$D_LAYERS"       \
    --kernel_size       "$KERNEL_SIZE"    \
    --lambda_ch         "$LAMBDA_CH"      \
    --lambda_agc        "$LAMBDA_AGC"     \
    --cumulative_ratio  "$CUMULATIVE_RATIO" \
    --batch_size        "$BATCH_SIZE"     \
    --train_epochs      "$TRAIN_EPOCHS"   \
    --patience          "$PATIENCE"       \
    --learning_rate     "$LEARNING_RATE"  \
    --lradj             "$LRADJ"          \
    --seed_iter         "$SEED_ITER"      \
    --show_layer_idx    "$SHOW_LAYER_IDX" \
    $EXTRA_ARGS
