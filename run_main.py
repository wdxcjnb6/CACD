"""
Main Training Script for CC_discover
Handles multi-seed training, testing, and optional Phase-2 validation (ablation study).

Usage example:
    python run_main.py --data_path demo.csv --seq_len 10 --pred_len 1 --seed_iter 3
"""

import os
import argparse
import datetime
import random
import shutil
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_discover import Exp_Main
from data.data_factory import data_provider
from utils.tools import (
    extract_channel_mask_from_model,
    plot_all_channels_R2,
    plot_causal_attention,
    export_seedavg_delay_causal_results,
    plot_pred_causal_matrix,
    plot_causal_graph_comparison,
)
from utils.metrics import (
    build_pred_matrix_from_triplets,
    binarize_by_cumulative_ratio,
    evaluate_causal_graph,
    evaluate_causal_graph_with_lag,
    print_metrics,
    save_three_causal_matrices,
    evaluate_sign_modulation,
)


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Define and return the argument parser for CC_discover experiments."""
    parser = argparse.ArgumentParser(description='CC_discover - Time Series Causal Discovery')

    # Basic settings
    parser.add_argument('--is_training',  type=int,   default=1,
                        help='1 = train then test; 0 = test only')
    parser.add_argument('--model_id',     type=str,   default='test',
                        help='Model identifier (used in checkpoint and output paths)')
    parser.add_argument('--model',        type=str,   default='CC_discover',
                        help='Model class name')
    parser.add_argument('--seed_iter',    type=int,   default=1,
                        help='Number of independent random seeds to run')
    parser.add_argument('--samplerate',  type=float, default=1000.,
                        help='Sampling rate (Hz) used when generating synthetic timestamps')

    # Data settings
    parser.add_argument('--data',       type=str,  default='wdxcjnb1',
                        help='Dataset type identifier (registered in data_factory)')
    parser.add_argument('--sample',     type=int,  default=1000,
                        help='Legacy sampling-frequency flag (dataset-dependent)')
    parser.add_argument('--root_path',  type=str,  default='./dataset/demo',
                        help='Root directory containing data files')
    parser.add_argument('--data_path',  type=str,  nargs='+',
                        default=['demo_dataset.csv'],
                        help='One or more CSV filenames under root_path, e.g. '
                             '--data_path TS1.csv TS2.csv')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='Directory to save model checkpoints')
    parser.add_argument('--inverse',    type=bool, default=False,
                        help='Apply inverse transform to predictions before evaluation')
    parser.add_argument('--ratios',     type=str,  default='0.6,0.1,0.3',
                        help='Comma-separated train/val/test[/veri] split ratios '
                             '(must sum to 1). Use 4 parts to enable Phase-2 validation.')

    # Ground-truth causal graph (optional — required only for metric evaluation)
    parser.add_argument('--gt_path',    type=str, nargs='*', default=None,
                        help='Ground-truth causal graph file(s). Provide one file to '
                             'share across all time-series, or one per time-series '
                             '(matching --data_path order).')
    parser.add_argument('--gt_with_lag', type=lambda x: x.lower() == 'true', default=True,
                        help='True: evaluate with lag dimension [C, C, max_lag+1]; '
                             'False: evaluate without lag [C, C]')

    # Forecasting task
    parser.add_argument('--seq_len',    type=int, default=10,
                        help='Input sequence (look-back) length')
    parser.add_argument('--pred_len',   type=int, default=1,
                        help='Prediction horizon length')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Causal convolution kernel size')

    # Model architecture
    parser.add_argument('--d_in',       type=int,   default=None,
                        help='Number of input channels (variables). '
                             'Auto-detected from CSV header when omitted.')
    parser.add_argument('--t_in',       type=int,   default=1,
                        help='Number of input time features')
    parser.add_argument('--time_emb_dim', type=int, default=1,
                        help='Time2Vec embedding dimension per time feature')
    parser.add_argument('--d_model',    type=int,   default=128,
                        help='Model hidden (d_model) dimension')
    parser.add_argument('--n_heads',    type=int,   default=16,
                        help='Number of attention heads')
    parser.add_argument('--d_layers',   type=int,   default=3,
                        help='Number of decoder layers')
    parser.add_argument('--d_ff',       type=int,   default=256,
                        help='Feed-forward network hidden dimension')
    parser.add_argument('--dropout',    type=float, default=0.0,
                        help='Dropout rate')
    parser.add_argument('--attn_dropout', type=float, default=0.0,
                        help='Attention dropout rate')

    # Normalization
    parser.add_argument('--scale',      type=bool, default=True,
                        help='Standardize input data (zero mean, unit variance)')
    parser.add_argument('--revin_flag', type=bool, default=False,
                        help='Use RevIN instance normalization')
    parser.add_argument('--norm_flag',  type=bool, default=False,
                        help='Use standard layer normalization')

    # Model features
    parser.add_argument('--pct_start',       type=float, default=0.3,
                        help='OneCycleLR warm-up percentage')
    parser.add_argument('--res_attention',   type=bool,  default=False,
                        help='Accumulate attention logits across layers (residual attention)')
    parser.add_argument('--pruning_ratio',   type=float, default=0.7,
                        help='Token pruning ratio (currently unused)')
    parser.add_argument('--pruning_enabled', type=bool,  default=True,
                        help='Enable token pruning (currently unused)')
    parser.add_argument('--show_layer_idx',  type=int,   default=-1,
                        help='Layer index to visualize (-1 = last layer)')
    parser.add_argument('--save_seed_plots', action='store_true', default=False,
                        help='Save per-seed plots in addition to the seed-averaged output')

    # Training hyperparameters
    parser.add_argument('--num_workers',   type=int,   default=10,
                        help='DataLoader worker processes')
    parser.add_argument('--itr',           type=int,   default=1,
                        help='Number of repeated runs per seed')
    parser.add_argument('--train_epochs',  type=int,   default=200,
                        help='Maximum training epochs')
    parser.add_argument('--batch_size',    type=int,   default=32,
                        help='Training batch size')
    parser.add_argument('--patience',      type=int,   default=10,
                        help='Early-stopping patience (epochs without improvement)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--loss',          type=str,   default='MSE',
                        help='Loss function identifier')
    parser.add_argument('--lradj',         type=str,   default='TST',
                        help='Learning-rate adjustment strategy')

    # Regularization / causal loss
    parser.add_argument('--lambda_ch',  type=float, default=0.05,
                        help='Channel-gate regularization weight (entropy + alignment)')
    parser.add_argument('--lambda_agc', type=float, default=0.05,
                        help='AGC (Attention-Gradient Consistency) auxiliary loss weight')
    parser.add_argument('--grad_thresh', type=float, default=0.0,
                        help='Absolute gradient threshold; edges below this are discarded')
    parser.add_argument('--strength_ratio_thresh', type=float, default=0.05,
                        help='Relative strength threshold: discard edges whose '
                             'causal_strength < max_strength * ratio (0 = no filter)')
    parser.add_argument('--cumulative_ratio', type=float, default=0.95,
                        help='Cumulative-strength threshold: keep the top edges that '
                             'together account for this fraction of total strength')
    parser.add_argument('--share_ch_mask', type=bool, default=True,
                        help='(Unused; kept for backward compatibility)')

    # Device settings
    parser.add_argument('--use_gpu',       type=bool,        default=True,
                        help='Use GPU if available')
    parser.add_argument('--gpu',           type=int,         default=0,
                        help='Primary GPU device ID')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='Use multiple GPUs via DataParallel')
    parser.add_argument('--devices',       type=str,         default='0,1,2,3',
                        help='Comma-separated GPU device IDs for multi-GPU mode')
    parser.add_argument('--test_flop',     action='store_true', default=False,
                        help='Measure FLOPs and exit immediately after one forward pass')

    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, use_gpu: bool) -> None:
    """Fix all random seeds for reproducible results."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def autodetect_d_in(csv_path: str) -> int:
    """
    Read the CSV header and return the number of numeric columns.
    Non-numeric columns (e.g. timestamp) are excluded.
    """
    sample = pd.read_csv(csv_path, nrows=2)
    return int(sample.select_dtypes(include='number').shape[1])


def get_gt_for_ts(gt_list, ts_idx: int):
    """
    Return the ground-truth file for a given time-series index.

    If only one GT file is provided, it is shared across all time-series.
    If multiple GT files are provided, they must match data_path one-to-one.
    """
    if gt_list is None:
        return None
    return gt_list[0] if len(gt_list) == 1 else gt_list[ts_idx]


def delete_checkpoint(ckpt_path: str) -> None:
    """Remove a checkpoint directory, ignoring errors."""
    if os.path.exists(ckpt_path):
        try:
            shutil.rmtree(ckpt_path)
            print(f"[INFO] Checkpoint deleted: {ckpt_path}")
        except Exception as exc:
            print(f"[WARNING] Could not delete checkpoint: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = build_parser()
    args   = parser.parse_args()

    # ---- Device configuration ----
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices   = args.devices.replace(' ', '')
        device_ids     = args.devices.split(',')
        args.device_ids = [int(d) for d in device_ids]
        args.gpu        = args.device_ids[0]

    print('=' * 60)
    print('Arguments:')
    print(args)
    print(f'Using GPU: {args.use_gpu}')
    print('=' * 60)

    # ---- Determine whether a 4th (verification) split exists ----
    # 3-part ratios (train,val,test): skip Phase-2 validation.
    # 4-part ratios (train,val,test,veri): run Phase-2 ablation study.
    ratios_parts = [p.strip() for p in str(args.ratios).split(',') if p.strip()]
    has_veri     = (len(ratios_parts) == 4)
    print(f"[INFO] {'4' if has_veri else '3'}-part ratios detected "
          f"-> Phase-2 validation will {'run' if has_veri else 'be skipped'}.")

    # ---- Output directory ----
    date_str        = datetime.date.today().strftime("%Y%m%d")
    base_root       = os.path.join('./test_results', date_str)
    base_result_dir = os.path.join(base_root, args.model_id)

    # Avoid overwriting an existing non-empty results directory
    if os.path.exists(base_result_dir) and os.listdir(base_result_dir):
        idx = 1
        while True:
            candidate = os.path.join(base_root, f'{args.model_id}_{idx}')
            if not os.path.exists(candidate) or not os.listdir(candidate):
                base_result_dir = candidate
                break
            idx += 1

    os.makedirs(base_result_dir, exist_ok=True)
    print(f"[INFO] Results will be saved to: {base_result_dir}")

    # ---- Validate GT / data_path pairing ----
    data_paths = args.data_path  # Always a list (argparse nargs='+')
    gt_list    = args.gt_path
    if gt_list is not None and len(gt_list) > 1:
        assert len(gt_list) == len(data_paths), (
            f"--gt_path count ({len(gt_list)}) must equal --data_path count "
            f"({len(data_paths)}), or supply a single shared GT file."
        )

    # ---- Per-time-series accumulators (reset per TS) ----
    orig_model_id = args.model_id  # Preserve original model_id before per-TS suffixing
    all_metrics   = {}             # {ts_name: metrics_dict} for the final summary CSV

    # ===========================================================================
    # OUTER LOOP: Iterate over every time-series file
    # ===========================================================================
    for ts_idx, ts_data_path in enumerate(data_paths):
        args.data_path = ts_data_path
        ts_name        = os.path.splitext(ts_data_path)[0]
        ts_result_dir  = os.path.join(base_result_dir, ts_name)
        os.makedirs(ts_result_dir, exist_ok=True)
        args.gt_path   = get_gt_for_ts(gt_list, ts_idx)

        # Auto-detect channel count (or use the manually specified value)
        data_file  = os.path.join(args.root_path, ts_data_path)
        if not os.path.exists(data_file):
            print(f"[SKIP] {ts_data_path} not found, skipping.")
            continue

        orig_d_in = args.d_in  # Preserve original CLI value (None = auto)
        if orig_d_in is None:
            args.d_in = autodetect_d_in(data_file)
            print(f"[INFO] Auto-detected d_in={args.d_in} from {ts_data_path}")
        else:
            print(f"[INFO] Using fixed d_in={args.d_in} (manually specified)")

        # Append ts_name to model_id so checkpoints for different TS files don't collide
        base_model_id = f'{orig_model_id}_{ts_name}'
        print("\n" + "=" * 60)
        print(f">>> Processing: {ts_data_path}  ->  {ts_result_dir}")
        print("=" * 60)

        # ---- Seed-averaged accumulators ----
        attn_sums_accum = None   # Accumulated cross-attention (list of per-layer tensors)
        n_samples_accum = 0      # Total number of test samples across seeds
        input_grad_sum  = None   # Accumulated input-gradient sensitivity (C, C, T)
        input_grad_n    = 0      # Number of seeds contributing to input_grad_sum
        gate_sum        = None   # Accumulated channel-gate values (C, C)
        gate_n          = 0      # Number of seeds contributing to gate_sum

        # =======================================================================
        # PHASE 1: Multi-Seed Training & Testing
        # =======================================================================
        if args.is_training:
            print("\n" + "=" * 60)
            print(">>> PHASE 1: MULTI-SEED TRAINING & TESTING <<<")
            print("=" * 60)

            seeds = list(range(args.seed_iter))
            for s in seeds:
                fix_seed = 2020 + s
                set_seed(fix_seed, args.use_gpu)
                args.model_id = f'{base_model_id}_seed{s}'
                print(f"\n[Seed {s}] Starting training with seed={fix_seed}")

                for ii in range(args.itr):
                    exp = Exp_Main(args)

                    # Checkpoint / output path string
                    setting = (
                        f'{args.model}_{args.data}_{args.model_id}_'
                        f'sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_'
                        f'nh{args.n_heads}_dl{args.d_layers}_df{args.d_ff}_'
                        f'ks{args.kernel_size}_ch{args.lambda_ch}_'
                        f'agc{args.lambda_agc}_{ii}'
                    )

                    # Train
                    print(f'[Seed {s}] Training: {setting}')
                    exp.train(setting)

                    # Test
                    print(f'[Seed {s}] Testing: {setting}')
                    seed_dir    = os.path.join(ts_result_dir, f'seed{s}')
                    test_result = exp.test(
                        setting,
                        test=1,
                        folder_path=seed_dir,
                        seed=s,
                        itr=ii,
                    )

                    # Remove checkpoint to free disk space
                    delete_checkpoint(os.path.join(args.checkpoints, setting))

                    # Unpack test results
                    mse        = test_result['mse']
                    preds      = test_result['preds']
                    trues      = test_result['trues']
                    inputx     = test_result['inputx']
                    attn_sums  = test_result['attn_sums']
                    n_samples  = test_result['n_samples']
                    input_grad = test_result['input_grad']  # (C_tgt, C_src, T)

                    core_model = (
                        exp.model.module
                        if isinstance(exp.model, nn.DataParallel)
                        else exp.model
                    )

                    # Optional: save per-seed visualizations
                    if args.save_seed_plots:
                        plot_all_channels_R2(
                            trues_all=trues,
                            preds_all=preds,
                            folder_path=seed_dir,
                            csv_path=os.path.join(ts_result_dir, 'all_seeds_r2.csv'),
                            meta={'seed': s, 'itr': ii},
                        )
                        plot_causal_attention(
                            attn_sums=attn_sums,
                            n_samples=n_samples,
                            d_in=args.d_in,
                            folder_path=seed_dir,
                            prefix=f'seed{s}_layer{args.show_layer_idx}',
                            avg_layers=False,
                            layer_idx=args.show_layer_idx,
                            do_norm=True,
                            gate_ch=extract_channel_mask_from_model(core_model, args.d_in),
                            input_grad=input_grad,
                            inputx=inputx,
                            preds=preds,
                        )

                    # Accumulate cross-attention across seeds (sample-weighted)
                    if attn_sums is not None:
                        if attn_sums_accum is None:
                            attn_sums_accum = list(attn_sums)
                        else:
                            for l in range(len(attn_sums)):
                                attn_sums_accum[l] += attn_sums[l]
                        n_samples_accum += n_samples

                    # Accumulate input-gradient sensitivity (sample-weighted)
                    if input_grad is not None:
                        if input_grad_sum is None:
                            input_grad_sum = input_grad * n_samples
                        else:
                            input_grad_sum += input_grad * n_samples

                    # Accumulate channel-gate masks
                    gate = extract_channel_mask_from_model(core_model, args.d_in)
                    if gate is not None:
                        gate_sum  = gate if gate_sum is None else gate_sum + gate
                        gate_n   += 1

                    torch.cuda.empty_cache()

            # ===================================================================
            # Seed-Averaged Results
            # ===================================================================
            print("\n" + "=" * 60)
            print(">>> SEED-AVERAGED RESULTS <<<")
            print("=" * 60)

            causal_out_dir = os.path.join(ts_result_dir, 'SeedAvg_CausalAnalysis')
            os.makedirs(causal_out_dir, exist_ok=True)

            input_grad_avg = (
                input_grad_sum / n_samples_accum
                if input_grad_sum is not None
                else None
            )
            gate_avg = gate_sum / gate_n if gate_n > 0 else None

            if attn_sums_accum is not None and n_samples_accum > 0:
                print("[INFO] Generating seed-averaged causal analysis...")

                # Visualize seed-averaged cross-attention
                plot_causal_attention(
                    attn_sums=attn_sums_accum,
                    n_samples=n_samples_accum,
                    d_in=args.d_in,
                    folder_path=causal_out_dir,
                    prefix=f'SeedAvg_layer{args.show_layer_idx}',
                    avg_layers=False,
                    do_norm=True,
                    gate_ch=gate_avg,
                    layer_idx=args.show_layer_idx,
                    input_grad=input_grad_avg,
                )

                # Export causal triplets (src, tgt, lag, direction, strength)
                triplets, causal_rownorm_gated, attn_lag_norm = (
                    export_seedavg_delay_causal_results(
                        global_attn_sums=attn_sums_accum,
                        global_n_total_samples=n_samples_accum,
                        d_in=args.d_in,
                        gate_avg=gate_avg,
                        input_grad_avg=input_grad_avg,
                        layer_idx=args.show_layer_idx,
                        avg_layers=False,
                        tau_gate=1 / args.d_in,
                        grad_thresh=args.grad_thresh,
                        strength_ratio_thresh=getattr(args, 'strength_ratio_thresh', 0.0),
                        cumulative_ratio=args.cumulative_ratio,
                        drop_self=True,
                        out_dir=causal_out_dir,
                        out_name="SeedAvg_delay_causal_triplets.csv",
                    )
                )

                # Binarize edges using cumulative-strength threshold
                pred_bin, strength_threshold_eff, lag_map, direction_map = (
                    binarize_by_cumulative_ratio(triplets, args.d_in, args.cumulative_ratio)
                )

                # Build signed modulation matrix: +1 = excitatory, -1 = inhibitory, 0 = absent
                delta_net = np.zeros((args.d_in, args.d_in), dtype=np.int8)
                for (src, tgt), d in direction_map.items():
                    delta_net[src, tgt] = d

                n_pred_edges = int(pred_bin.sum())
                print(
                    f"[INFO] cumulative_ratio={args.cumulative_ratio}: "
                    f"{n_pred_edges} edges cover >= {args.cumulative_ratio * 100:.0f}% "
                    f"of total strength."
                )

                # Visualize predicted causal graph (no GT required)
                plot_pred_causal_matrix(
                    causal_rownorm_gated=causal_rownorm_gated,
                    pred_bin=pred_bin,
                    lag_map=lag_map,
                    d_in=args.d_in,
                    out_dir=causal_out_dir,
                )

                # ---------------------------------------------------------------
                # Save three causal matrices: adjacency / lag / modulation
                # ROI labels are read from the CSV header when available.
                # ---------------------------------------------------------------
                try:
                    roi_df     = pd.read_csv(os.path.join(args.root_path, ts_data_path), nrows=0)
                    roi_labels = list(roi_df.columns)
                    if len(roi_labels) != args.d_in:
                        roi_labels = None  # Column count mismatch — fall back to ch0/ch1/...
                except Exception:
                    roi_labels = None

                save_three_causal_matrices(
                    triplets=triplets,
                    pred_bin=pred_bin,
                    lag_map=lag_map,
                    d_in=args.d_in,
                    out_dir=causal_out_dir,
                    roi_labels=roi_labels,
                )

                # ===============================================================
                # PHASE 2: Validation (Ablation Study)
                # Only runs when 4-part ratios are specified (has_veri=True)
                # ===============================================================
                if triplets is not None and len(triplets) > 0 and has_veri:
                    print("\n" + "=" * 60)
                    print(">>> PHASE 2: VALIDATION (ABLATION STUDY) <<<")
                    print("=" * 60)

                    print(f"\n[Input] Teacher proposed {len(triplets)} candidate edges (src->tgt@lag):")
                    for row in triplets:
                        src, tgt, lag = int(row[0]), int(row[1]), int(row[2])
                        print(f"  {src} -> {tgt} @ lag={lag}")

                    print("\n[Data] Loading verification set (flag='veri')...")
                    validator = Exp_Validation(args, triplets)

                    print("\n[Process] Training student model & running ablation...")
                    validator.train(epochs=500)

                    # delta = AblationLoss - BaseLoss; positive delta means the edge matters
                    validated_edges = validator.run_ablation(mode="mean")
                    kept_edges      = [e for e in validated_edges if e[3] >  1e-6]
                    pruned_edges    = [e for e in validated_edges if e[3] <= 1e-6]

                    print("\n[Output] Validation Result (delta = AblLoss - BaseLoss):")
                    print(
                        f"Total: {len(validated_edges)} | "
                        f"Kept (delta>0): {len(kept_edges)} | "
                        f"Pruned (delta<=0): {len(pruned_edges)}"
                    )
                    print("\nAll edges sorted by delta (descending):")
                    print(f"{'Src':<5} {'->':<4} {'Tgt':<5} {'Lag':<5} | {'DeltaLoss':<12}")
                    print("-" * 44)
                    for src, tgt, lag, delta in validated_edges:
                        print(f"{int(src):<5} {'->':<4} {int(tgt):<5} {int(lag):<5} | {delta:+.6f}")

                    print("\nKept edges (delta > 0):")
                    for src, tgt, lag, delta in kept_edges:
                        print(f"  KEEP  {int(src)} -> {int(tgt)} @ lag={int(lag)} | delta={delta:+.6f}")

                    print("\nPruned edges (delta <= 0):")
                    for src, tgt, lag, delta in pruned_edges:
                        print(f"  PRUNE {int(src)} -> {int(tgt)} @ lag={int(lag)} | delta={delta:+.6f}")

                    # Save validation results
                    val_out_csv = os.path.join(causal_out_dir, "Validated_Causal_Edges.csv")
                    with open(val_out_csv, 'w') as f:
                        f.write("src,tgt,lag,delta_loss\n")
                        for src, tgt, lag, delta in validated_edges:
                            f.write(f"{int(src)},{int(tgt)},{int(lag)},{float(delta):.8f}\n")
                    print(f"\n[File] Saved to: {val_out_csv}")
                    print("=" * 60 + "\n")

                else:
                    if not has_veri:
                        print("[Skip] 3-part ratios -> no independent 'veri' split, Phase-2 skipped.")
                    else:
                        print("[Skip] No edges found in Phase-1, Phase-2 skipped.")

                # ===============================================================
                # PHASE 3: Causal Discovery Metrics
                # Compares seed-averaged triplets against the ground-truth graph.
                # Only runs when --gt_path is provided.
                # ===============================================================
                if triplets is not None and len(triplets) > 0:
                    tmp_ds, _  = data_provider(args, flag='test')
                    causal_gt      = tmp_ds.causal_gt
                    causal_gt_sign = getattr(tmp_ds, 'causal_gt_sign', None)
                    gt_with_lag    = getattr(args, 'gt_with_lag', False)
                    del tmp_ds

                    if causal_gt is not None:
                        print("\n" + "=" * 60)
                        print(">>> PHASE 3: CAUSAL DISCOVERY METRICS <<<")
                        print("=" * 60)
                        try:
                            if gt_with_lag:
                                # Align GT and prediction matrices along the lag dimension
                                max_lag_gt   = causal_gt.shape[-1] - 1
                                max_lag_pred = max((int(row[2]) for row in triplets), default=0)
                                max_lag      = max(max_lag_gt, max_lag_pred)

                                pred_mat = build_pred_matrix_from_triplets(
                                    triplets, args.d_in, max_lag=max_lag
                                )

                                # Zero-pad GT if prediction covers more lags
                                if max_lag > max_lag_gt:
                                    pad = np.zeros(
                                        (args.d_in, args.d_in, max_lag - max_lag_gt),
                                        dtype=causal_gt.dtype,
                                    )
                                    causal_gt_eval = np.concatenate([causal_gt, pad], axis=-1)
                                else:
                                    causal_gt_eval = causal_gt

                                metrics = evaluate_causal_graph_with_lag(
                                    pred_mat, causal_gt_eval,
                                    threshold=strength_threshold_eff,
                                    pred_bin_ext=pred_bin,
                                    lag_map=lag_map,
                                )
                                print_metrics(metrics, title="Causal Discovery Metrics (with Lag)")

                                sign_metrics = evaluate_sign_modulation(
                                    triplets, pred_bin, lag_map, causal_gt_sign, args.d_in
                                )
                                metrics.update(sign_metrics)
                                print_metrics(sign_metrics, title="Sign Modulation Metrics")

                                plot_causal_graph_comparison(
                                    gt_bin=(causal_gt.sum(axis=-1) > 0).astype(np.float32),
                                    pred_bin=pred_bin,
                                    d_in=args.d_in,
                                    out_dir=causal_out_dir,
                                    metrics=metrics,
                                    filename="CausalGraph_GT_vs_Pred_collapsed.png",
                                    gt_matrix_lag=causal_gt,
                                    lag_map=lag_map,
                                )
                            else:
                                # Lag-free evaluation: use collapsed [C, C] matrices
                                pred_mat = build_pred_matrix_from_triplets(
                                    triplets, args.d_in, max_lag=None
                                )
                                metrics = evaluate_causal_graph(
                                    pred_mat, causal_gt,
                                    threshold=strength_threshold_eff,
                                    pred_bin_ext=pred_bin,
                                )
                                print_metrics(metrics, title="Causal Discovery Metrics (No Lag)")

                                sign_metrics = evaluate_sign_modulation(
                                    triplets, pred_bin, lag_map, causal_gt_sign, args.d_in
                                )
                                metrics.update(sign_metrics)
                                print_metrics(sign_metrics, title="Sign Modulation Metrics")

                                plot_causal_graph_comparison(
                                    gt_bin=(causal_gt > 0).astype(np.float32),
                                    pred_bin=pred_bin,
                                    d_in=args.d_in,
                                    out_dir=causal_out_dir,
                                    metrics=metrics,
                                    filename="CausalGraph_GT_vs_Pred.png",
                                )

                            # Save metrics CSV
                            metrics_csv = os.path.join(causal_out_dir, "CausalDiscovery_Metrics.csv")
                            with open(metrics_csv, 'w') as mf:
                                mf.write("metric,value\n")
                                for k, v in metrics.items():
                                    mf.write(f"{k},{v}\n")
                            print(f"[File] Metrics saved: {metrics_csv}")
                            print("=" * 60 + "\n")

                        except Exception as exc:
                            print(f"[WARNING] Metric computation failed: {exc}")
                            traceback.print_exc()
                    else:
                        if getattr(args, 'gt_path', None) is not None:
                            print("\n[Skip] GT file specified but failed to load, skipping Phase-3.")
                        else:
                            print("\n[Skip] No --gt_path provided, skipping Phase-3 metrics.")

        # ---- Collect this TS's metrics into the summary dict ----
        try:
            metrics_csv = os.path.join(
                ts_result_dir, "SeedAvg_CausalAnalysis", "CausalDiscovery_Metrics.csv"
            )
            if os.path.exists(metrics_csv):
                mdf = pd.read_csv(metrics_csv, header=0, index_col=0)
                all_metrics[ts_name] = {str(k): float(v) for k, v in mdf['value'].items()}
        except Exception as exc:
            print(f"[WARNING] Could not read metrics for {ts_data_path}: {exc}")

        # Restore d_in to the original CLI value so the next TS can auto-detect
        args.d_in = orig_d_in

    # ===========================================================================
    # Summary CSV: aggregate metrics across all time-series
    # ===========================================================================
    if all_metrics:
        rows = [{"ts": ts, **m} for ts, m in all_metrics.items()]
        dataset_name = os.path.basename(os.path.normpath(args.root_path))
        summary_df   = pd.DataFrame(rows).set_index("ts")
        summary_df.index.name = dataset_name

        num_cols = summary_df.select_dtypes(include="number").columns
        summary_df = pd.concat([
            summary_df,
            summary_df[num_cols].mean().rename("mean").to_frame().T,
            summary_df[num_cols].std().rename("std").to_frame().T,
        ])

        summary_csv = os.path.join(base_result_dir, "Summary_CausalDiscovery_Metrics.csv")
        summary_df.to_csv(summary_csv, index_label=dataset_name)
        print(f"\n[File] Summary saved: {summary_csv}")
        print(summary_df.to_string())

    print("\n[INFO] Experiment completed successfully!")
