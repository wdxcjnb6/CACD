"""
Utility Functions and Classes for Time Series Causal Discovery
Organized by functionality:
    1. Core Utilities (EarlyStopping, StandardScaler, etc.)
    2. Neural Network Modules (RevIN, CausalConv, GEGLU)
    3. Visualization (plot_heatmap, visualize_causal_attention, etc.)
    4. Model Analysis (extract_channel_mask, export_causal_results)
    5. Causal Attention (select_attn_layer, compute_causal_triplets, etc.)
"""

# ===================== Imports =====================
import math
import os
import time
import warnings
from collections import defaultdict
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import matplotlib
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from einops import rearrange


# ===================== 1. Core Utilities =====================

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Stops training when validation loss stops improving.
    """

    def __init__(self, patience: int = 7, verbose: bool = False, delta: float = 0.0):
        """
        Args:
            patience: how many epochs to wait for improvement
            verbose: whether to print status messages
            delta: minimum improvement threshold
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss: float, model: nn.Module, path: str):
        """
        Check if training should stop.

        Args:
            val_loss: current validation loss
            model: model to save
            path: checkpoint save path
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss: float, model: nn.Module, path: str):
        """Save model checkpoint using an atomic write to avoid corruption."""
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        os.makedirs(path, exist_ok=True)
        tmp_path   = os.path.join(path, 'checkpoint.pth.tmp')
        final_path = os.path.join(path, 'checkpoint.pth')
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, final_path)  # Atomic replace: avoids a corrupt file if process dies mid-write
        self.val_loss_min = val_loss


class StandardScaler:
    """Standard normalization scaler."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean
        self.std  = std

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean, unit variance."""
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse the normalization."""
        return (data * self.std) + self.mean


class dotdict(dict):
    """Dictionary with dot-notation attribute access."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    args,
    printout: bool = True,
):
    """
    Adjust the learning rate according to the configured schedule type.

    Args:
        optimizer: optimizer instance
        scheduler: learning rate scheduler (used for the 'TST' strategy)
        epoch: current epoch (1-indexed)
        args: namespace containing lradj and learning_rate
        printout: whether to print LR updates
    """
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2:  args.learning_rate * 0.5 ** 1,
            4:  args.learning_rate * 0.5 ** 2,
            6:  args.learning_rate * 0.5 ** 3,
            8:  args.learning_rate * 0.5 ** 4,
            10: args.learning_rate * 0.5 ** 5,
        }
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'fixed':
        lr_adjust = {}
        if printout and epoch == 1:
            print(f"Using fixed learning rate: {optimizer.param_groups[0]['lr']}")
    else:
        if printout and epoch == 1:
            print(f"Warning: lradj type '{args.lradj}' not recognized. Using fixed LR.")
        lr_adjust = {}

    if epoch in lr_adjust:
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout and args.lradj != 'TST':
            print(f'Updating learning rate to {lr}')


def test_params_flop(model: nn.Module, x_shape: Tuple[int, ...]):
    """
    Report model parameter count and (optionally) FLOPs.

    Args:
        model: PyTorch model
        x_shape: input tensor shape, e.g. (seq_len, d_in)
    """
    model_params = sum(p.numel() for p in model.parameters())
    print(f'Trainable parameter count: {model_params / 1e6:.2f}M')

    try:
        from ptflops import get_model_complexity_info
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(
                model.cuda(), x_shape,
                as_strings=True,
                print_per_layer_stat=True,
            )
            print(f'Computational complexity: {macs}')
            print(f'Number of parameters: {params}')
    except ImportError:
        print("Warning: ptflops not installed. Skipping FLOP calculation.")


# ===================== 2. Neural Network Modules =====================

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN).
    Normalizes the input and can denormalize the output, preserving per-instance statistics.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ):
        """
        Args:
            num_features: number of channels C
            eps: numerical stability epsilon
            affine: whether to use learnable affine parameters
            subtract_last: if True, subtract the last timestep instead of the mean
        """
        super(RevIN, self).__init__()
        self.num_features  = num_features
        self.eps           = eps
        self.affine        = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def forward(self, x: Tensor, mode: str) -> Tensor:
        """
        Args:
            x: input tensor, shape (B, C, T)
            mode: 'norm' to normalize, 'denorm' to reverse
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"mode '{mode}' not supported")
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias   = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: Tensor):
        """Compute and cache mean/std (or last value) from input."""
        dim2reduce = -1  # For (B, C, T), reduce over the time dimension

        if self.subtract_last:
            self.last = x[:, :, -1].unsqueeze(-1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: Tensor) -> Tensor:
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight.unsqueeze(-1)
            x = x + self.affine_bias.unsqueeze(-1)
        return x

    def _denormalize(self, x: Tensor) -> Tensor:
        if self.affine:
            x = x - self.affine_bias.unsqueeze(-1)
            x = x / (self.affine_weight.unsqueeze(-1) + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class CausalDepthwiseConv1d(nn.Module):
    """
    Causal depthwise 1-D convolution.
    Each channel is convolved independently (groups=C), and the input is
    left-padded so that position t only sees timesteps <= t (no future leakage).
    """

    def __init__(self, C: int, kernel_size: int = 3, dilation: int = 1):
        """
        Args:
            C: number of channels
            kernel_size: convolution kernel size
            dilation: dilation factor
        """
        super(CausalDepthwiseConv1d, self).__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=C,
            out_channels=C,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=C,
            bias=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) — same shape, causally convolved
        """
        x = F.pad(x, (self.pad, 0))  # Left-pad to enforce causality
        return self.conv(x)


class ChannelWisePointwise1d(nn.Module):
    """Channel-wise 1×1 convolution: an independent linear transform per channel."""

    def __init__(self, C: int, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=C,
            out_channels=C,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=C,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class GEGLU(nn.Module):
    """
    Gated GLU activation with GELU gate.
    Splits the last dimension in half; one half is gated by GELU of the other.
    Output dimension is half of the input dimension.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (*, d_ff) — d_ff must be even
        Returns:
            (*, d_ff // 2)
        """
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Normalization(nn.Module):
    """
    Flexible normalization wrapper supporting LayerNorm, BatchNorm, or identity.
    """

    def __init__(self, method: str, d_model: Optional[int] = None):
        """
        Args:
            method: 'layer', 'batch', or 'none'
            d_model: hidden dimension (required for 'layer' and 'batch')
        """
        super().__init__()
        assert method in ["layer", "batch", "none"], f"Unknown normalization method: {method}"

        if method == "layer":
            assert d_model is not None, "d_model required for LayerNorm"
            self.norm = nn.LayerNorm(d_model)
        elif method == "batch":
            assert d_model is not None, "d_model required for BatchNorm1d"
            self.norm = nn.BatchNorm1d(d_model)
        else:
            self.norm = lambda x: x

        self.method = method

    def forward(self, x: Tensor) -> Tensor:
        if self.method == "batch":
            # BatchNorm1d expects (B, C, T); transpose if x is (B, T, C)
            return self.norm(x.transpose(-1, 1)).transpose(-1, 1)
        return self.norm(x)


# ===================== Tensor Reshaping Utilities =====================

def Flatten(inp: Tensor) -> Tensor:
    """
    Flatten a spatiotemporal tensor.
    (B, T, C) -> (B, T*C, 1)
    """
    return rearrange(inp, "batch len dy -> batch (dy len) 1")


def Localize(inp: Tensor, variables: int) -> Tensor:
    """
    Split spatiotemporal tensor into individual variables and fold into batch.
    (B, C*T, D) -> (C*B, T, D)
    """
    return rearrange(
        inp,
        "batch (variables len) dim -> (variables batch) len dim",
        variables=variables,
    )


# ===================== 3. Visualization Functions =====================

def plot_heatmap(
    data: np.ndarray,
    folder_path: str,
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = 'Blues',
    xticklabels: Optional[List] = None,
    yticklabels: Optional[List] = None,
    origin: str = 'lower',
    invert_y: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
):
    """
    Render a 2-D array as a heatmap and save it as a PDF.

    Args:
        data: 2-D numpy array to visualize
        folder_path: output directory
        filename: output filename (should end in .pdf)
        title: plot title
        xlabel, ylabel: axis labels
        vmin, vmax: color-scale limits
        cmap: colormap name
        xticklabels, yticklabels: tick labels
        origin: 'lower' or 'upper'
        invert_y: whether to invert the y-axis
        figsize: figure size (width, height); auto-computed if None
    """
    if figsize is None:
        width   = max(8, data.shape[1] * 0.2) if data.ndim > 1 else 8
        figsize = (width, 6)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, origin=origin, aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if xticklabels is not None:
        ax.set_xticks(np.arange(len(xticklabels)))
        ax.set_xticklabels(xticklabels, rotation=90, fontsize=6)

    if yticklabels is not None:
        ax.set_yticks(np.arange(len(yticklabels)))
        ax.set_yticklabels(yticklabels)

    if invert_y:
        ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, filename)

    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig, bbox_inches='tight')

    plt.close(fig)
    print(f"Saved: {pdf_path}")


def plot_all_channels_R2(
    trues_all: np.ndarray,
    preds_all: np.ndarray,
    folder_path: str,
    filename: str = 'test_all_channels_R2.pdf',
    csv_path: Optional[str] = None,
    meta: Optional[dict] = None,
) -> List[float]:
    """
    Plot prediction vs ground truth for all channels with R² scores.

    Args:
        trues_all: ground truth,  shape (N, pred_len, C)
        preds_all: predictions,   shape (N, pred_len, C)
        folder_path: output directory
        filename: output PDF filename
        csv_path: optional CSV path to append R² scores
        meta: optional metadata dict to include in the CSV rows

    Returns:
        List of R² scores, one per channel.
    """
    if isinstance(preds_all, torch.Tensor):
        preds_all = preds_all.detach().cpu().numpy()
    else:
        preds_all = np.asarray(preds_all)

    if isinstance(trues_all, torch.Tensor):
        trues_all = trues_all.detach().cpu().numpy()
    else:
        trues_all = np.asarray(trues_all)

    N, pred_len, C = preds_all.shape
    y_true_flat = trues_all.reshape(-1, C)
    y_pred_flat = preds_all.reshape(-1, C)

    T = min(500, y_true_flat.shape[0])  # Plot at most 500 timesteps

    fig, axes = plt.subplots(C, 1, sharex=True, figsize=(16, 4 * C))
    if C == 1:
        axes = [axes]

    x            = np.arange(T)
    true_color   = '#1f77b4'
    pred_color   = '#ff7f0e'
    shadow_color = '#c6dbef'

    r2_list = []
    for ch in range(C):
        ax = axes[ch]
        ax.set_facecolor('#f9f9f9' if ch % 2 == 0 else '#ffffff')
        ax.grid(color='#e0e0e0', linestyle='--', linewidth=0.8, alpha=0.7)

        err = np.abs(y_true_flat[:T, ch] - y_pred_flat[:T, ch])
        ax.fill_between(x, y_true_flat[:T, ch] - err, y_true_flat[:T, ch] + err,
                        color=shadow_color, alpha=0.35)

        ax.plot(x, y_true_flat[:T, ch], color=true_color,
                linewidth=2.5, alpha=0.5, label='true')
        ax.plot(x, y_pred_flat[:T, ch], color=pred_color,
                linewidth=2.5, alpha=0.5,
                marker='o', markersize=3, markevery=max(T // 40, 1), label='predicted')

        ax.set_xlim((-10, None))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        r_ch = r2_score(y_true_flat[:, ch], y_pred_flat[:, ch])
        r2_list.append(r_ch)
        print(f'Channel {ch} R²: {r_ch:.4f}')

        ax.tick_params(axis='y', which='major', labelsize=16)
        ax.set_title(f'Channel {ch} (test R²={r_ch:.2f})', fontsize=22, color='#333333', pad=6)

    axes[0].legend(bbox_to_anchor=(0.97, 1), frameon=False, fontsize=18,
                   handlelength=2.8, borderpad=0.2)

    os.makedirs(folder_path, exist_ok=True)
    pdf_path = os.path.join(folder_path, filename)
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    if csv_path is not None:
        rows      = []
        base_meta = meta or {}
        for ch, r2v in enumerate(r2_list):
            row = dict(base_meta)
            row.update({'channel': ch, 'r2': float(r2v)})
            rows.append(row)

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=write_header, index=False)

    return r2_list


# ===================== 5. Model Analysis Functions =====================

def extract_channel_mask_from_model(model: nn.Module, d_in: int) -> Optional[np.ndarray]:
    """
    Extract the channel gate from the model and return a binarized mask.

    ch_gate is a learnable parameter inside CausalAttention. When
    share_ch_mask=True, the self- and cross-attention layers share the same
    Parameter object; id()-based deduplication prevents double-counting.

    Args:
        model: model instance (may be wrapped in DataParallel)
        d_in:  number of channels C

    Returns:
        float mask of shape [C, C] with values 0.0 or 1.0;
        returns None if no ch_gate is found in the model.
    """
    model_core = model.module if isinstance(model, nn.DataParallel) else model

    # Traverse all submodules; deduplicate shared Parameters by id()
    seen_ids  = set()
    gate_list = []
    for m in model_core.modules():
        if hasattr(m, 'ch_gate') and m.ch_gate is not None:
            gid = id(m.ch_gate)
            if gid not in seen_ids:
                seen_ids.add(gid)
                gate_list.append(m.ch_gate.detach().cpu().numpy())  # (C, C)

    if not gate_list:
        print('Warning: No ch_gate found in model')
        return None

    # Average across independent gates (when share_ch_mask=False, each layer has its own)
    gate_np   = np.mean(np.stack(gate_list, axis=0), axis=0)  # (C, C)
    gate_soft = 1.0 / (1.0 + np.exp(-gate_np))               # sigmoid
    mask_01   = (gate_soft > 0.5).astype(float)              # binarize

    if mask_01.shape != (d_in, d_in):
        print(f'Warning: mask shape {mask_01.shape} != ({d_in}, {d_in})')

    return mask_01


# ===================== 6. Causal Attention — Shared Computation =====================
# Pure data-processing functions with no side effects.
# Called by both the visualization and the causal-export entry points.

def select_attn_layer(
    attn_sums: List[Tensor],
    n_samples: int,
    layer_idx: int,
    avg_layers: bool,
) -> Tensor:
    """
    Select the target layer from the multi-layer cross-attention accumulators
    and divide by the sample count to obtain the per-sample mean.

    Args:
        attn_sums:  list of Tensor [P, H, S], per-layer cumulative cross-attention sums
        n_samples:  total number of accumulated samples
        layer_idx:  which layer to use (-1 = last layer)
        avg_layers: if True, average over all layers (layer_idx is ignored)

    Returns:
        attn_mean: Tensor [P, S], divided by n_samples
    """
    # Sum over the head dimension: [L, P, S]
    attn_allh = torch.stack([s.sum(dim=1) for s in attn_sums], dim=0)

    attn_mean = attn_allh.mean(dim=0) if avg_layers else attn_allh[layer_idx]  # [P, S]
    return attn_mean / (n_samples + 1e-8)


def build_lag_attn_norm(
    attn_mean: Tensor,
    C: int,
    pred_len_tokens: int,
    seq_len_tokens: int,
    do_norm: bool = True,
) -> np.ndarray:
    """
    Reshape the [P, S] attention matrix into [C, C, T] (tgt, src, lag)
    and optionally row-normalize.

    Reshape logic:
        P = C * pred_len_tokens,  S = C * seq_len_tokens
        view → [tgt_C, pred_len, src_C, T] → sum over pred_len → [tgt_C, src_C, T]

    Normalization (do_norm=True):
        flatten → [C, C*T], row-normalize per target, reshape back to [C, C, T]

    Returns:
        attn_lag_norm: np.ndarray [C, C, T]
            attn_lag_norm[tgt, src, t] = normalized attention from tgt to src at lag = T-t
    """
    attn_4d  = attn_mean.view(C, pred_len_tokens, C, seq_len_tokens)  # [tgt, Pl, src, T]
    attn_lag = attn_4d.sum(dim=1).detach().cpu().numpy()              # [tgt, src, T]

    if do_norm:
        flat         = attn_lag.reshape(C, C * seq_len_tokens)
        flat         = flat / (flat.sum(axis=1, keepdims=True) + 1e-8)
        attn_lag_norm = flat.reshape(C, C, seq_len_tokens)
    else:
        attn_lag_norm = attn_lag

    return attn_lag_norm


def compute_causal_map(
    attn_mean: Tensor,
    C: int,
    pred_len_tokens: int,
    seq_len_tokens: int,
    do_norm: bool = True,
) -> np.ndarray:
    """
    Collapse the [P, S] attention matrix to a [C, C] causal graph
    by summing over the lag dimension.

    Returns:
        causal_norm: np.ndarray [C, C]
            causal_norm[tgt, src] = total causal influence of src on tgt
            (row-normalized when do_norm=True)
    """
    attn_4d    = attn_mean.view(C, pred_len_tokens, C, seq_len_tokens)
    causal_raw = attn_4d.sum(dim=(1, 3)).detach().cpu().numpy()  # [C, C]

    if do_norm:
        causal_norm = causal_raw / (causal_raw.sum(axis=1, keepdims=True) + 1e-8)
    else:
        causal_norm = causal_raw

    return causal_norm


def apply_gate_and_normalize(
    causal_norm: np.ndarray,
    gate_avg: Optional[np.ndarray],
    tau_gate: float,
    drop_self: bool,
    C: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply channel-gate binarization, optionally remove self-connections,
    and row-normalize the result.

    Args:
        causal_norm:  [C, C] row-normalized causal map
        gate_avg:     [C, C] seed-averaged binary gate (proportion of seeds where
                      gate=1), produced by accumulating extract_channel_mask_from_model()
                      across seeds and averaging.
                      gate_avg[i, j] = fraction of seeds for which the edge (i->j) was active.
                      If None, no gate filtering is applied.
        tau_gate:     Gate activity threshold; edges with gate_avg[i, j] <= tau_gate
                      are suppressed. Recommended: 1/C or 0.5.
        drop_self:    If True, zero out the diagonal (self-causality).
        C:            Number of channels.

    Returns:
        causal_rownorm_gated: [C, C] gate-filtered, row-normalized causal strength
                              (used as the score matrix for metrics)
        gate_bin:             [C, C] binary gate (float32, 0 or 1)
    """
    if gate_avg is not None:
        G = np.asarray(gate_avg, dtype=np.float32)
        if G.shape != (C, C):
            raise ValueError(f"gate_avg shape must be ({C}, {C}), got {G.shape}")
        gate_bin     = (G > tau_gate).astype(np.float32)
        causal_gated = causal_norm * gate_bin
    else:
        gate_bin     = np.ones((C, C), dtype=np.float32)
        causal_gated = causal_norm.copy()

    if drop_self:
        np.fill_diagonal(causal_gated, 0.0)

    causal_rownorm_gated = causal_gated / (causal_gated.sum(axis=1, keepdims=True) + 1e-8)
    return causal_rownorm_gated, gate_bin


# ===================== 7. Causal Attention — Visualization =====================
# Each sub-function handles one plot type and can be called independently.
# plot_causal_attention() is the main entry point that calls all sub-functions.

def plot_lag_heatmaps(
    attn_lag_norm: np.ndarray,
    folder_path: str,
    prefix: str,
) -> None:
    """
    Produce two lag-dimension visualizations:
      1. Flattened heatmap (all src×lag concatenated) → single-page PDF
      2. One slice per lag value                       → multi-page PDF

    Args:
        attn_lag_norm: [C, C, T], row-normalized attention indexed as [tgt, src, lag]
        folder_path:   output directory
        prefix:        filename prefix
    """
    C, _, T = attn_lag_norm.shape

    # Flatten to [C, C*T]
    attn_flat    = attn_lag_norm.reshape(C, C * T)
    xtick_labels = [f"{src}_{T - t}" for src in range(C) for t in range(T)]
    ytick_labels = np.arange(C)
    vmin, vmax   = attn_flat.min(), attn_flat.max()

    # Plot 1: flattened heatmap
    plot_heatmap(
        data=attn_flat,
        folder_path=folder_path,
        filename=f'{prefix}_lag_flat.pdf',
        title='Cross-attention (all src×lag, row-normalized per tgt)',
        xlabel='Source channel × lag',
        ylabel='Target channel',
        vmin=vmin, vmax=vmax, cmap='Blues',
        xticklabels=xtick_labels,
        yticklabels=ytick_labels,
        origin='lower', invert_y=True,
    )

    # Plot 2: one page per lag slice
    pdf_path = os.path.join(folder_path, f'{prefix}_lag_slices.pdf')
    with PdfPages(pdf_path) as pdf:
        for t in range(T):
            lag = T - t
            mat = attn_lag_norm[:, :, t]  # [C, C]

            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, origin='lower', aspect='auto',
                           vmin=vmin, vmax=vmax, cmap='Blues')
            ax.set_title(f'Cross-attention slice (lag={lag})')
            ax.set_xlabel('Source channel')
            ax.set_ylabel('Target channel')
            ax.set_xticks(np.arange(C))
            ax.set_yticks(np.arange(C))
            ax.invert_yaxis()
            plt.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved lag heatmaps -> {folder_path}")


def plot_per_layer_causal(
    attn_sums: List[Tensor],
    n_samples: int,
    C: int,
    pred_len_tokens: int,
    seq_len_tokens: int,
    folder_path: str,
    prefix: str,
    do_norm: bool = True,
) -> None:
    """
    Generate one causal map per decoder layer (summed over lags → [C, C])
    and save all pages to a single PDF.

    Args:
        attn_sums:        list of [P, H, S], per-layer cross-attention accumulators
        n_samples:        total accumulated sample count
        C:                number of channels
        pred_len_tokens:  number of prediction tokens
        seq_len_tokens:   number of sequence tokens
        folder_path:      output directory
        prefix:           filename prefix
        do_norm:          whether to row-normalize
    """
    L            = len(attn_sums)
    causal_layers = []

    for l in range(L):
        attn_l   = attn_sums[l].sum(dim=1) / (n_samples + 1e-8)  # [P, S]
        causal_l = compute_causal_map(attn_l, C, pred_len_tokens, seq_len_tokens, do_norm)
        causal_layers.append(causal_l)

    causal_stack = np.stack(causal_layers, axis=0)  # [L, C, C]
    vmin = np.percentile(causal_stack, 5)
    vmax = np.percentile(causal_stack, 95)
    if vmin == vmax:
        vmin, vmax = causal_stack.min(), causal_stack.max() + 1e-8

    pdf_path = os.path.join(folder_path, f'{prefix}_causal_per_layer.pdf')
    with PdfPages(pdf_path) as pdf:
        for l, causal_l in enumerate(causal_layers):
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(causal_l, origin='lower', aspect='auto',
                           vmin=vmin, vmax=vmax, cmap='YlOrRd')
            ax.set_title(f'Causal map layer {l} (sum over lags, row-normalized)')
            ax.set_xlabel('Source channel')
            ax.set_ylabel('Target channel')
            ax.set_xticks(np.arange(C))
            ax.set_yticks(np.arange(C))
            ax.invert_yaxis()
            plt.colorbar(im, ax=ax)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    print(f"Saved per-layer causal maps -> {pdf_path}")


def plot_gated_causal(
    causal_rownorm_gated: np.ndarray,
    C: int,
    folder_path: str,
    prefix: str,
) -> None:
    """
    Plot the gate-filtered, row-normalized causal map as a single-page PDF.

    Args:
        causal_rownorm_gated: [C, C] gate-filtered row-normalized causal strengths
        C:                    number of channels
        folder_path:          output directory
        prefix:               filename prefix
    """
    vmin = np.percentile(causal_rownorm_gated, 5)
    vmax = np.percentile(causal_rownorm_gated, 95)
    if vmin == vmax:
        vmin, vmax = causal_rownorm_gated.min(), causal_rownorm_gated.max() + 1e-8

    plot_heatmap(
        data=causal_rownorm_gated,
        folder_path=folder_path,
        filename=f'{prefix}_causal_gated.pdf',
        title='Causal map (gate-filtered, row-normalized)',
        xlabel='Source channel',
        ylabel='Target channel',
        vmin=vmin, vmax=vmax, cmap='YlOrRd',
        xticklabels=np.arange(C),
        yticklabels=np.arange(C),
        origin='lower', invert_y=True, figsize=(6, 5),
    )


def plot_input_grad_effect(
    input_grad: np.ndarray,
    attn_lag_norm: np.ndarray,
    C: int,
    folder_path: str,
    prefix: str,
    inputx: Optional[np.ndarray] = None,
    preds: Optional[np.ndarray] = None,
) -> None:
    """
    Visualize input-gradient effects (excitatory/inhibitory directions).
    Optionally produces per-target source-contribution breakdown plots.

    Plot 1: flattened heatmap of input_grad (all src×lag, bwr colormap,
            red=excitatory, blue=inhibitory)
    Plot 2 (optional): one PDF per target channel:
            row 0: target channel history + prediction
            rows 1..C: each source channel's time-series, colored by
                       input_grad[tgt, src, t]

    Args:
        input_grad:    [C, C, T], ∂y_tgt/∂x_src[t] — positive=excitatory, negative=inhibitory
        attn_lag_norm: [C, C, T], normalized attention (used for shared tick labels)
        C:             number of channels
        folder_path:   output directory
        prefix:        filename prefix
        inputx:        [N, seq_len, C] historical input (optional; enables breakdown plots)
        preds:         [N, pred_len, C] predictions (optional; enables breakdown plots)
    """
    T = input_grad.shape[2]

    # Zero out the diagonal (self-causality is not meaningful here)
    grad = input_grad.copy().astype(np.float32)
    for c in range(C):
        grad[c, c, :] = 0.0

    # Normalize to [-1, 1] for display
    m         = np.max(np.abs(grad))
    grad_norm = grad / m if m > 1e-9 else grad

    grad_flat    = grad_norm.reshape(C, C * T)
    xtick_labels = [f"{src}_{T - t}" for src in range(C) for t in range(T)]

    # Plot 1: flattened heatmap
    plot_heatmap(
        data=grad_flat,
        folder_path=folder_path,
        filename=f'{prefix}_input_grad_flat.pdf',
        title='Input gradient ∂y_tgt/∂x_src[lag]\n(red=excitatory, blue=inhibitory)',
        xlabel='Source channel × lag',
        ylabel='Target channel',
        vmin=-1.0, vmax=1.0, cmap='bwr',
        xticklabels=xtick_labels,
        yticklabels=np.arange(C),
        origin='lower', invert_y=True,
    )

    # Plot 2 (optional): per-target breakdown
    if inputx is None or preds is None:
        return

    hist = inputx[0]  # [seq_len, C]
    fut  = preds[0].reshape(-1, C) if preds[0].ndim == 1 else preds[0]  # [pred_len, C]

    if hist.ndim == 2 and hist.shape[0] == C:
        hist = hist.T  # Ensure shape [seq_len, C]

    seq_len  = hist.shape[0]
    pred_len = fut.shape[0]
    t_hist   = np.arange(seq_len)
    t_fut    = np.arange(seq_len, seq_len + pred_len)

    norm_scatter = plt.Normalize(vmin=-m, vmax=m)
    cmap_scatter = plt.get_cmap('bwr')

    for tgt in range(C):
        pdf_path = os.path.join(folder_path, f'{prefix}_target{tgt}_grad_breakdown.pdf')
        fig, axes = plt.subplots(C + 1, 1, figsize=(12, 1.8 * (C + 1)), sharex=True)

        # Row 0: target channel history and prediction
        ax = axes[0]
        ax.plot(t_hist, hist[:, tgt], 'k', label='History')
        ax.plot([t_hist[-1], t_fut[0]], [hist[-1, tgt], fut[0, tgt]], 'k:', alpha=0.5)
        ax.scatter(t_fut, fut[:, tgt], c='k', marker='*', s=200, label='Pred')
        ax.set_title(f'Target {tgt} — history & prediction', fontweight='bold', loc='left')
        ax.legend(fontsize=8)

        # Rows 1..C: per-source contribution direction
        for src in range(C):
            ax       = axes[src + 1]
            contribs = grad[tgt, src]  # [T], signed gradient
            sc = ax.scatter(t_hist, hist[:, src], c=contribs,
                            cmap=cmap_scatter, norm=norm_scatter,
                            s=30, edgecolors='k', lw=0.2)
            ax.plot(t_hist, hist[:, src], c='gray', alpha=0.2)
            ax.set_ylabel(f'Src {src}', rotation=0, labelpad=20, fontsize=8)
            if src == C - 1:
                ax.set_xlabel('Timestep (lag)')
                ax.set_xticks(t_hist)
                ax.set_xticklabels([str(seq_len - i) for i in range(seq_len)])

        cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.6])
        fig.colorbar(sc, cax=cbar_ax, label='∂y_tgt/∂x_src (red=excitatory, blue=inhibitory)')

        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved input gradient breakdowns -> {folder_path}")


def plot_causal_attention(
    attn_sums: List[Tensor],
    n_samples: int,
    d_in: int,
    folder_path: str,
    layer_idx: int,
    prefix: str = 'causal',
    avg_layers: bool = False,
    do_norm: bool = True,
    gate_ch: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    input_grad: Optional[np.ndarray] = None,
    inputx: Optional[np.ndarray] = None,
    preds: Optional[np.ndarray] = None,
) -> None:
    """
    Main entry point for causal-attention visualization.
    Calls four sub-functions in sequence:
      1. plot_lag_heatmaps      — flattened lag heatmap + per-lag slice PDF
      2. plot_per_layer_causal  — one causal-map PDF per decoder layer
      3. plot_gated_causal      — gate-filtered causal map (when gate_ch is not None)
      4. plot_input_grad_effect — input-gradient effect plot (when input_grad is not None)

    Args:
        attn_sums:   list of [P, H, S], per-layer cross-attention accumulators
        n_samples:   total accumulated sample count
        d_in:        number of channels C
        folder_path: output directory
        layer_idx:   which layer to visualize (-1 = last layer)
        prefix:      filename prefix
        avg_layers:  if True, average over all layers
        do_norm:     whether to row-normalize
        gate_ch:     [C, C] channel gate (optional)
        tau:         gate binarization threshold (uses 1/C if None)
        input_grad:  [C, C, T] ∂y_tgt/∂x_src[t] (optional)
        inputx:      [N, seq_len, C] historical input (optional; enables breakdown plots)
        preds:       [N, pred_len, C] predictions (optional; enables breakdown plots)
    """
    C = d_in
    os.makedirs(folder_path, exist_ok=True)

    P_tokens = attn_sums[0].shape[0]
    S_tokens = attn_sums[0].shape[2]
    assert P_tokens % C == 0 and S_tokens % C == 0, \
        f"Token/channel mismatch: P={P_tokens}, S={S_tokens}, C={C}"
    pred_len_tokens = P_tokens // C
    seq_len_tokens  = S_tokens // C

    attn_mean     = select_attn_layer(attn_sums, n_samples, layer_idx, avg_layers)
    attn_lag_norm = build_lag_attn_norm(attn_mean, C, pred_len_tokens, seq_len_tokens, do_norm)

    # Step 1: lag heatmaps
    plot_lag_heatmaps(attn_lag_norm, folder_path, prefix)

    # Step 2: per-layer causal maps
    plot_per_layer_causal(attn_sums, n_samples, C, pred_len_tokens, seq_len_tokens,
                          folder_path, prefix, do_norm)

    # Step 3: gate-filtered causal map
    if gate_ch is not None:
        causal_norm = compute_causal_map(attn_mean, C, pred_len_tokens, seq_len_tokens, do_norm)
        tau_eff     = tau if tau is not None else (1.0 / C)
        causal_rownorm_gated, _ = apply_gate_and_normalize(causal_norm, gate_ch, tau_eff, True, C)
        plot_gated_causal(causal_rownorm_gated, C, folder_path, prefix)

    # Step 4: input-gradient effect plots
    if input_grad is not None:
        plot_input_grad_effect(input_grad, attn_lag_norm, C, folder_path, prefix, inputx, preds)


# ===================== 8. Causal Export =====================
# Causal-triplet extraction and CSV export.
# export_seedavg_delay_causal_results() is the main entry point.

def select_edges(
    gate_bin: np.ndarray,
    C: int,
    drop_self: bool,
) -> List[Tuple[int, int]]:
    """
    Use the channel gate for coarse edge filtering; return a list of candidate edges.

    The gate provides a binary exist/absent judgment; strength-based filtering
    (gradient threshold) is applied later in compute_causal_triplets.

    Args:
        gate_bin:  [C, C] binary gate (0/1), produced by apply_gate_and_normalize
        C:         number of channels
        drop_self: if True, exclude self-connections

    Returns:
        edges: list of (tgt, src) tuples
    """
    edges = []
    for tgt in range(C):
        for src in range(C):
            if drop_self and tgt == src:
                continue
            if gate_bin[tgt, src] > 0:
                edges.append((tgt, src))
    return edges


def compute_causal_triplets(
    attn_lag_norm: np.ndarray,
    input_grad: np.ndarray,
    edges: List[Tuple[int, int]],
    C: int,
    T: int,
    grad_thresh: float = 0.0,
    strength_ratio_thresh: float = 0.0,
) -> List[Tuple]:
    """
    For each candidate edge, compute lag, modulation direction, composite strength,
    and sign stability; assemble into a sorted triplets list.

    Pipeline (per edge):
        1. Joint effect:    c_{i,j,τ} = W_{i,j,τ} × |∂y_i/∂x_j[τ]|
        2. Lag estimation:  τ̂ = T - argmax_τ c_{i,j,τ}
        3. Gradient filter: |∂y_i/∂x_j[τ̂]| <= grad_thresh → discard edge
        4. Direction:       sign of ∂y_i/∂x_j[τ̂] — positive=excitatory, negative=inhibitory
        5. Strength:        c_{i,j,τ̂}, always positive; used as the metric score
        6. Sign stability:  |mean(grad[tgt,src,:])| / (std(grad[tgt,src,:]) + eps)
                            Measures directional consistency across the lag sequence.
                            Near 1.0 = stable direction; near 0 = sign frequently flips.
                            When signs cancel across batches, the batch-mean approaches 0
                            and this value approaches 0 as well.

    Args:
        attn_lag_norm:         [C, C, T], row-normalized cross-attention W_{i,j,τ}
        input_grad:            [C, C, T], batch-averaged ∂y_tgt/∂x_src[τ]
                               positive=excitatory, negative=inhibitory
        edges:                 list of (tgt, src) after gate filtering
        C:                     number of channels
        T:                     sequence length (number of time steps)
        grad_thresh:           discard edges with |∂y/∂x[τ̂]| <= grad_thresh (default: no filter)
        strength_ratio_thresh: discard edges with strength < max_strength * ratio (default: no filter)

    Returns:
        triplets: list of (src, tgt, lag, direction, causal_strength, sign_stability)
                  sorted by causal_strength descending
            lag:             τ̂, causal transmission delay (minimum = 1)
            direction:       ∂y_tgt/∂x_src[τ̂], signed
            causal_strength: W_{i,j,τ̂} × |∂y_i/∂x_j[τ̂]|, always positive
            sign_stability:  directional reliability; interpret 'direction' only when > 0.5
    """
    # Zero out the diagonal (self-connections are not meaningful)
    grad = input_grad.copy().astype(np.float32)
    for c in range(C):
        grad[c, c, :] = 0.0

    triplets = []
    for (tgt, src) in edges:
        attn_vec = attn_lag_norm[tgt, src]  # [T]
        grad_vec = np.abs(grad[tgt, src])   # [T]

        # Steps 1-2: joint-effect argmax → lag
        combined = attn_vec * grad_vec
        lag_idx  = int(np.argmax(combined))
        lag      = int(T - lag_idx)         # minimum lag = 1

        # Step 3: gradient fine-filter
        grad_at_lag = float(grad[tgt, src, lag_idx])
        if abs(grad_at_lag) <= grad_thresh:
            continue

        # Steps 4-5: direction and strength
        strength = float(combined[lag_idx])

        # Step 6: sign stability over the full lag sequence
        g_seq          = grad[tgt, src]     # [T], signed
        _mean_abs      = abs(g_seq.mean())
        sign_stability = float(_mean_abs / (g_seq.std() + 1e-8))

        triplets.append((src, tgt, lag, grad_at_lag, strength, sign_stability))

    triplets.sort(key=lambda x: x[4], reverse=True)

    # Relative-strength filter: discard edges below max_strength * threshold
    if strength_ratio_thresh > 0.0 and triplets:
        max_strength = triplets[0][4]  # Already sorted descending
        thresh_abs   = max_strength * strength_ratio_thresh
        triplets     = [t for t in triplets if t[4] >= thresh_abs]

    return triplets


def save_causal_triplets(
    triplets: List[Tuple],
    out_dir: str,
    out_name: str,
    max_strength: float = 0.0,
    strength_ratio_thresh: float = 0.0,
    cumulative_ratio: float = 0.95,
) -> None:
    """
    Save triplets as a CSV file and print a terminal preview.

    CSV column descriptions:
        rank              — causal strength rank (1 = strongest)
        src / tgt / edge  — source, target channels and human-readable edge label
        lag               — causal transmission delay (in samples)
        direction         — ∂y_tgt/∂x_src[τ̂], signed
        direction_abs     — |direction|
        effect            — 'excitatory' or 'inhibitory'
        causal_strength   — W × |∂y/∂x|, always positive
        rel_to_max        — strength / max_strength
        strength_ratio_thresh — the relative-strength threshold used
        tgt_col_total     — sum of all incoming edge strengths for this target
        cum_ratio_in_tgt  — cumulative fraction of tgt's total strength up to this edge
        cumulative_ratio  — the cumulative_ratio threshold used
        selected          — 1 if this edge was selected by binarize_by_cumulative_ratio
        sign_stability    — directional reliability score
    """
    os.makedirs(out_dir, exist_ok=True)

    # Pre-compute per-target cumulative strength fractions
    _tgt_groups = defaultdict(list)
    for row in triplets:
        _tgt_groups[int(row[1])].append(row)

    _cum_info = {}  # (src, tgt) -> (col_total, cum_ratio)
    for tgt_id, rows in _tgt_groups.items():
        rows_s    = sorted(rows, key=lambda r: float(r[4]), reverse=True)
        col_total = sum(float(r[4]) for r in rows_s)
        cumsum    = 0.0
        for row in rows_s:
            cumsum += float(row[4])
            _cum_info[(int(row[0]), tgt_id)] = (
                col_total,
                cumsum / col_total if col_total > 0 else 0.0,
            )

    csv_path = os.path.join(out_dir, out_name)
    _max_s   = max_strength if max_strength > 0 else (triplets[0][4] if triplets else 1.0)

    with open(csv_path, 'w') as f:
        f.write(
            "rank,src,tgt,edge,lag,direction,direction_abs,effect,"
            "causal_strength,rel_to_max,strength_ratio_thresh,"
            "tgt_col_total,cum_ratio_in_tgt,cumulative_ratio,"
            "selected,sign_stability\n"
        )
        for rank, (src, tgt, lag, direction, strength, sign_stab) in enumerate(triplets, start=1):
            effect    = "excitatory" if direction >= 0 else "inhibitory"
            rel       = strength / _max_s if _max_s > 0 else 0.0
            col_total, cum = _cum_info.get((src, tgt), (0.0, 0.0))

            # Determine whether this edge was selected by the cumulative_ratio criterion.
            # An edge is selected if it is one of the top-k edges whose cumulative
            # strength first reaches cumulative_ratio within its target column.
            _rows_tgt = sorted(_tgt_groups.get(tgt, []), key=lambda r: float(r[4]), reverse=True)
            _col_t    = sum(float(r[4]) for r in _rows_tgt)
            _cs, _sel = 0.0, False
            for _r in _rows_tgt:
                _cs += float(_r[4])
                if int(_r[0]) == src and abs(float(_r[4]) - strength) < 1e-12:
                    _sel = True
                    break
                if _cs / _col_t >= cumulative_ratio:
                    break
            selected = 1 if _sel else 0

            f.write(
                f"{rank},{src},{tgt},{src}->{tgt},{lag},"
                f"{direction:.8e},{abs(direction):.8e},{effect},"
                f"{strength:.8e},{rel:.6f},{strength_ratio_thresh:.4f},"
                f"{col_total:.8e},{cum:.6f},{cumulative_ratio:.4f},"
                f"{selected},{sign_stab:.4f}\n"
            )
    print(f"Saved causal triplets (CSV) -> {csv_path}")

    # Terminal preview (top 10)
    print("==== Causal triplets preview ====")
    for rank, row in enumerate(triplets[:10], start=1):
        src, tgt, lag, direction, strength, sign_stab = row
        effect = "excitatory" if direction >= 0 else "inhibitory"
        conf   = "★" if sign_stab > 0.5 else "?"
        print(
            f"  #{rank:>2}  {src} -> {tgt} @ lag={lag} | "
            f"{effect}({direction:+.4f})  strength={strength:.4f}  "
            f"sign_stability={sign_stab:.3f}{conf}"
        )


def export_seedavg_delay_causal_results(
    global_attn_sums: List[Tensor],
    global_n_total_samples: int,
    d_in: int,
    gate_avg: Optional[np.ndarray] = None,
    input_grad_avg: Optional[np.ndarray] = None,
    layer_idx: int = -1,
    avg_layers: bool = False,
    tau_gate: float = 0.5,
    grad_thresh: float = 0.0,
    strength_ratio_thresh: float = 0.0,
    cumulative_ratio: float = 0.95,
    drop_self: bool = True,
    out_dir: Optional[str] = None,
    out_name: str = "seedavg_delay_causal_triplets.csv",
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Main entry point for causal triplet extraction and export (called after seed averaging).

    Pipeline:
        1. select_attn_layer        — select target layer → attn_mean [P, S]
        2. build_lag_attn_norm      — reshape + normalize → [C, C, T]
        3. compute_causal_map       — sum over lags → [C, C] causal map
        4. apply_gate_and_normalize — gate filtering + row-normalization
        5. select_edges             — coarse candidate edges from gate
        6. compute_causal_triplets  — lag (attention argmax) + direction (input_grad)
        7. save_causal_triplets     — write CSV

    Args:
        global_attn_sums:        list of [P, H, S], per-layer cross-attention sums
                                 accumulated across seeds
        global_n_total_samples:  total number of test samples across seeds
        d_in:                    number of channels C
        gate_avg:                [C, C] seed-averaged channel gate (optional)
        input_grad_avg:          [C, C, T] seed-averaged input gradient (required)
        layer_idx:               target layer (-1 = last layer)
        avg_layers:              if True, average over all layers
        tau_gate:                gate binarization threshold (pass 1/C from caller)
        grad_thresh:             absolute gradient threshold; edges below this are discarded
        strength_ratio_thresh:   relative strength threshold; discard edges below
                                 max_strength * ratio (0.0 = no filter)
        drop_self:               whether to remove self-causal edges (default True)
        out_dir:                 output directory (None = do not save)
        out_name:                CSV filename

    Returns:
        triplets:             list of (src, tgt, lag, direction, causal_strength, sign_stability)
        causal_rownorm_gated: [C, C] causal strength matrix
        attn_lag_norm:        [C, C, T] normalized attention (reusable for visualization)
    """
    assert global_attn_sums is not None and len(global_attn_sums) > 0
    if input_grad_avg is None:
        raise ValueError("input_grad_avg is required for lag-direction computation")

    C        = d_in
    P_tokens = global_attn_sums[0].shape[0]
    S_tokens = global_attn_sums[0].shape[2]

    if P_tokens % C != 0 or S_tokens % C != 0:
        raise ValueError(f"Token/channel mismatch: P={P_tokens}, S={S_tokens}, C={C}")

    pred_len_tokens = P_tokens // C
    seq_len_tokens  = S_tokens // C
    T               = seq_len_tokens

    # Steps 1-2: select layer + build lag-dimension normalized attention
    attn_mean     = select_attn_layer(global_attn_sums, global_n_total_samples, layer_idx, avg_layers)
    attn_lag_norm = build_lag_attn_norm(attn_mean, C, pred_len_tokens, seq_len_tokens, do_norm=True)

    # Steps 3-4: causal map + gate
    causal_norm          = compute_causal_map(attn_mean, C, pred_len_tokens, seq_len_tokens, do_norm=True)
    causal_rownorm_gated, gate_bin = apply_gate_and_normalize(
        causal_norm, gate_avg, tau_gate, drop_self, C
    )

    # Step 5: coarse edge candidates from gate
    edges = select_edges(gate_bin, C, drop_self)

    # Step 6: fine-filter + compute strengths
    input_grad = np.asarray(input_grad_avg, dtype=np.float32)
    if input_grad.shape != (C, C, T):
        raise ValueError(f"input_grad_avg must be ({C}, {C}, {T}), got {input_grad.shape}")

    triplets = compute_causal_triplets(
        attn_lag_norm, input_grad, edges, C, T, grad_thresh, strength_ratio_thresh
    )

    # Step 7: save
    if out_dir is not None:
        _max_s = triplets[0][4] if triplets else 0.0
        save_causal_triplets(
            triplets, out_dir, out_name,
            max_strength=_max_s,
            strength_ratio_thresh=strength_ratio_thresh,
            cumulative_ratio=cumulative_ratio,
        )

    return triplets, causal_rownorm_gated, attn_lag_norm


# ===================== 9. Predicted Causal Graph Visualization =====================

def plot_pred_causal_matrix(
    causal_rownorm_gated: np.ndarray,
    pred_bin: np.ndarray,
    lag_map: dict,
    d_in: int,
    out_dir: str,
    filename: str = "SeedAvg_PredCausalGraph.png",
) -> str:
    """
    Visualize the final predicted causal graph (no ground truth required).

    Left panel:  continuous strength matrix (heatmap, internal convention [tgt, src],
                 transposed to [src, tgt] for display)
    Right panel: binary prediction [src, tgt] with lag annotations in each cell

    Parameters
    ----------
    causal_rownorm_gated : np.ndarray [C, C]  continuous causal strength
                           (internal [tgt, src]; auto-transposed for display)
    pred_bin             : np.ndarray [C, C]  binary prediction [src, tgt]
    lag_map              : dict (src, tgt) -> lag
    d_in                 : int
    out_dir              : str
    filename             : str
    """
    C           = d_in
    n_edges     = int(pred_bin.sum())
    tick_labels = [f'X{i}' for i in range(C)]
    fs          = 7 if C > 12 else 9

    # Use GridSpec to allocate a narrow dedicated column for the colorbar,
    # keeping the two main axes at equal width.
    # Column layout: [ax0] [colorbar (narrow)] [gap (narrow)] [ax1]
    cell_size = max(4.0, C * 0.65)
    fig = plt.figure(figsize=(cell_size * 2 + 2.2, cell_size + 1.5), facecolor='white')
    gs  = GridSpec(1, 4, figure=fig, width_ratios=[1, 0.06, 0.08, 1], wspace=0.0)
    ax0     = fig.add_subplot(gs[0, 0])
    cbar_ax = fig.add_subplot(gs[0, 1])
    ax1     = fig.add_subplot(gs[0, 3])
    axes    = [ax0, ax1]

    # Left panel: continuous strength heatmap
    cont = causal_rownorm_gated.copy().astype(np.float32)
    np.fill_diagonal(cont, 0.0)
    vmax = np.percentile(cont, 97) if cont.max() > 0 else 1.0
    vmax = max(vmax, 1e-8)
    im = ax0.imshow(cont, origin='upper', aspect='equal',
                    cmap='YlOrRd', vmin=0, vmax=vmax, interpolation='nearest')
    ax0.set_title(
        'Causal strength (continuous)\nseed-avg, gate-filtered, row-normalized',
        fontsize=10, fontweight='bold', pad=8,
    )
    fig.colorbar(im, cax=cbar_ax)

    # Right panel: binary prediction.
    # pred_bin is [src, tgt]; transpose to [tgt, src] so that Y=tgt, X=src, matching the left panel.
    def _hex(h):
        return np.array([int(h[1:3], 16) / 255., int(h[3:5], 16) / 255., int(h[5:7], 16) / 255.])

    pred_bin_disp = pred_bin.T  # [tgt, src]
    rgb = np.ones((C, C, 3))
    for i in range(C):
        for j in range(C):
            if i == j:
                rgb[i, j] = _hex('#CCCCCC')
            elif pred_bin_disp[i, j]:
                rgb[i, j] = _hex('#1A9850')
            else:
                rgb[i, j] = _hex('#F7F7F7')

    axes[1].imshow(rgb, origin='upper', aspect='equal', interpolation='nearest')
    axes[1].set_title(
        f'Predicted causal graph\n{n_edges} edges  (cumulative_ratio applied)',
        fontsize=10, fontweight='bold', pad=8,
    )

    cell_fs = max(5, 8 - C // 4)
    for (src, tgt), lag in lag_map.items():
        if 0 <= src < C and 0 <= tgt < C:
            # After transposition: col=src, row=tgt → text(x=src, y=tgt)
            axes[1].text(src, tgt, f'τ={lag}',
                         ha='center', va='center',
                         fontsize=cell_fs, color='white', fontweight='bold')

    for ax in axes:
        ax.set_xticks(np.arange(C))
        ax.set_yticks(np.arange(C))
        ax.set_xticklabels(tick_labels, fontsize=fs)
        ax.set_yticklabels(tick_labels, fontsize=fs)
        ax.set_xlabel('Source (src)', fontsize=9)
        ax.set_ylabel('Target (tgt)', fontsize=9)
        ax.set_xticks(np.arange(C) - 0.5, minor=True)
        ax.set_yticks(np.arange(C) - 0.5, minor=True)
        ax.grid(which='minor', color='#BBBBBB', linewidth=0.4)
        ax.tick_params(which='minor', bottom=False, left=False)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[File] Predicted causal graph saved -> {out_path}")
    return out_path


# ===================== 10. GT vs Pred Comparison (requires --gt_path) =====================

def plot_causal_graph_comparison(
    gt_bin: np.ndarray,
    pred_bin: np.ndarray,
    d_in: int,
    out_dir: str,
    metrics: Optional[dict] = None,
    filename: str = "CausalGraph_GT_vs_Pred.png",
    gt_matrix_lag: Optional[np.ndarray] = None,
    lag_map: Optional[dict] = None,
) -> str:
    """
    Three-panel side-by-side visualization:
      Left:   Ground Truth (blue cells + τ=GT_lag annotations)
      Middle: Predicted (TP/FP/FN coloring)
      Right:  Predicted lag matrix (τ=pred_lag annotations on predicted positive cells)

    Uses a color-blind-friendly palette (Wong 2011).

    Parameters
    ----------
    gt_bin        : np.ndarray [C, C]  ground-truth binary adjacency [src, tgt]
    pred_bin      : np.ndarray [C, C]  predicted binary adjacency    [src, tgt]
    d_in          : int
    out_dir       : str
    metrics       : dict or None
    filename      : str
    gt_matrix_lag : np.ndarray [C, C, max_lag+1] or None — GT lag annotations
    lag_map       : dict (src, tgt) -> pred_lag or None  — predicted lag annotations
    """
    C = d_in
    p = pred_bin.astype(int).copy()
    g = gt_bin.astype(int).copy()
    np.fill_diagonal(p, 0)
    np.fill_diagonal(g, 0)

    tp = int(np.sum((p == 1) & (g == 1)))
    fp = int(np.sum((p == 1) & (g == 0)))
    fn = int(np.sum((p == 0) & (g == 1)))

    def _hex(h):
        return np.array([int(h[1:3], 16) / 255., int(h[3:5], 16) / 255., int(h[5:7], 16) / 255.])

    # Wong color-blind-friendly palette
    C_BLUE  = _hex('#0072B2')
    C_GREEN = _hex('#009E73')
    C_RED   = _hex('#D55E00')
    C_ORANG = _hex('#CC79A7')
    C_BG    = _hex('#F7F7F7')
    C_DIAG  = _hex('#BBBBBB')

    p_disp = p.T
    g_disp = g.T

    # Left panel: ground truth
    gt_rgb = np.ones((C, C, 3))
    for i in range(C):
        for j in range(C):
            gt_rgb[i, j] = C_DIAG if i == j else (C_BLUE if g_disp[i, j] else C_BG)

    # Middle panel: TP / FP / FN coloring
    pred_rgb = np.ones((C, C, 3))
    for i in range(C):
        for j in range(C):
            if i == j:
                pred_rgb[i, j] = C_DIAG
            elif p_disp[i, j] and g_disp[i, j]:
                pred_rgb[i, j] = C_GREEN   # TP
            elif p_disp[i, j] and not g_disp[i, j]:
                pred_rgb[i, j] = C_RED     # FP
            elif not p_disp[i, j] and g_disp[i, j]:
                pred_rgb[i, j] = C_ORANG  # FN
            else:
                pred_rgb[i, j] = C_BG     # TN

    # Right panel: TP edges colored by whether lag prediction is correct.
    #   Green = TP and lag correct
    #   Red   = TP but lag wrong
    #   All other cells remain background (non-TP edges are not evaluated for lag)
    lag_rgb = np.ones((C, C, 3))
    for i in range(C):
        for j in range(C):
            lag_rgb[i, j] = C_DIAG if i == j else C_BG

    if gt_matrix_lag is not None and lag_map is not None:
        for src in range(C):
            for tgt in range(C):
                if src == tgt:
                    continue
                # Only process TP edges (transposed display: i=tgt, j=src)
                if p_disp[tgt, src] and g_disp[tgt, src]:
                    pred_lag = lag_map.get((src, tgt), -1)
                    gt_lags  = np.where(gt_matrix_lag[src, tgt] > 0)[0]
                    if len(gt_lags) > 0 and pred_lag == int(gt_lags[0]):
                        lag_rgb[tgt, src] = C_GREEN  # Lag correct
                    else:
                        lag_rgb[tgt, src] = C_RED    # Lag wrong

    tick_labels = [f'X{i}' for i in range(C)]
    fs        = 7 if C > 12 else 9
    cell_fs   = max(5, 8 - C // 4)
    cell_size = max(4.0, C * 0.65)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(cell_size * 3 + 1.5, cell_size + 1.5),
        facecolor='white',
    )

    for ax, rgb, title in zip(
        axes,
        [gt_rgb, pred_rgb, lag_rgb],
        [f'Ground Truth  ({int(g.sum())} edges)',
         f'Predicted  ({tp + fp} detected)',
         'Predicted lag'],
    ):
        ax.imshow(rgb, origin='upper', aspect='equal', interpolation='nearest')
        ax.set_title(title, fontsize=10, fontweight='bold', pad=8)
        ax.set_xticks(np.arange(C))
        ax.set_yticks(np.arange(C))
        ax.set_xticklabels(tick_labels, fontsize=fs)
        ax.set_yticklabels(tick_labels, fontsize=fs)
        ax.set_xlabel('Source (src)', fontsize=9)
        ax.set_ylabel('Target (tgt)', fontsize=9)
        ax.set_xticks(np.arange(C) - 0.5, minor=True)
        ax.set_yticks(np.arange(C) - 0.5, minor=True)
        ax.grid(which='minor', color='#BBBBBB', linewidth=0.4)
        ax.tick_params(which='minor', bottom=False, left=False)

    # Left panel: annotate GT lags
    if gt_matrix_lag is not None:
        for src in range(C):
            for tgt in range(C):
                if src == tgt:
                    continue
                gt_lags = np.where(gt_matrix_lag[src, tgt] > 0)[0]
                if len(gt_lags) > 0:
                    axes[0].text(src, tgt, f'τ={int(gt_lags[0])}',
                                 ha='center', va='center',
                                 fontsize=cell_fs, color='white', fontweight='bold')

    # Right panel: annotate predicted lags
    if lag_map is not None:
        for (src, tgt), pred_lag in lag_map.items():
            if 0 <= src < C and 0 <= tgt < C:
                axes[2].text(src, tgt, f'τ={pred_lag}',
                             ha='center', va='center',
                             fontsize=cell_fs, color='white', fontweight='bold')

    # Middle panel footer: classification metrics
    _nan = float('nan')

    def _get(key, fallback):
        if metrics is None:
            return _nan
        return metrics.get(key, metrics.get(fallback, _nan))

    parts = [f'TP={tp}', f'FP={fp}', f'FN={fn}']
    for label, key, fb in [
        ('F1',    'f1',        'edge_f1'),
        ('Prec',  'precision', 'edge_precision'),
        ('Rec',   'recall',    'edge_recall'),
        ('AUROC', 'auroc',     'edge_auroc'),
        ('AUPRC', 'auprc',     'edge_auprc'),
    ]:
        v = _get(key, fb)
        parts.append(f'{label}={v:.3f}' if not math.isnan(v) else f'{label}=N/A')
    axes[1].set_xlabel('Source (src)\n' + '  |  '.join(parts), fontsize=7)

    # Right panel footer: lag metrics
    if metrics is not None:
        n_tp_lag = metrics.get('lag_n_tp', _nan)
        n_corr   = metrics.get('lag_n_correct', _nan)
        lag_acc  = metrics.get('lag_accuracy', _nan)
        lag_str  = (
            f'TP={n_tp_lag}  lag_correct={n_corr}  lag_acc={lag_acc:.3f}'
            if not math.isnan(lag_acc) else ''
        )
        axes[2].set_xlabel(f'Source (src)\n{lag_str}', fontsize=7)

    legend_handles = [
        mpatches.Patch(color=C_BLUE,  label='GT edge'),
        mpatches.Patch(color=C_GREEN, label=f'TP={tp} / lag correct'),
        mpatches.Patch(color=C_RED,   label=f'FP={fp} / lag wrong'),
        mpatches.Patch(color=C_ORANG, label=f'FN={fn}'),
        mpatches.Patch(color=C_BG,    label='TN'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=5,
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[File] Causal graph comparison saved -> {out_path}")
    return out_path
