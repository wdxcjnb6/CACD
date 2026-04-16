# CACD

**CACD** is a time-series causal discovery framework that learns directed causal graphs, together with transmission lag and modulation polarity, directly from multivariate time-series data without requiring any predefined graph structure.

The model uses a dual-branch decoder architecture: the self-attention branch models within-channel temporal dynamics, whereas the cross-attention branch captures cross-channel causal relationships. Causal strength, transmission lag, and modulation polarity are jointly inferred from attention weights and input-gradient attribution.
---

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Dataset Format](#dataset-format)
- [Configuration Reference](#configuration-reference)
- [Output Files](#output-files)
- [Model Architecture](#model-architecture)
- [Three-Phase Pipeline](#three-phase-pipeline)
- [Evaluation Against Ground Truth](#evaluation-against-ground-truth)

---

## Key Features

- **Lag-aware causal discovery** — estimates both existence and transmission delay of each causal edge
- **Modulation direction** — distinguishes excitatory (+1) from inhibitory (−1) influences via input-gradient sign
- **Multi-seed averaging** — runs N independent seeds and averages attention and gradient maps for stable results
- **Channel-gate regularization** — learnable binary gate encourages sparse, interpretable connectivity
- **AGC auxiliary loss** — aligns cross-attention weights with gradient-based attribution during training
- **Phase-2 ablation validation** — optional student-model ablation study to prune false-positive edges
- **Ground-truth evaluation** — AUROC / AUPRC / F1 / Precision / Recall / SHD / lag-accuracy metrics when a GT graph is provided
- **Multi-file support** — process a list of time-series CSVs in a single run and produce a summary CSV

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CACD.git
cd CACD

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:** `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `einops`

---

## Quick Start

### 1. Run the demo

```bash
bash demo.sh
```

This trains a model on the bundled demo dataset (`./dataset/demo/demo_dataset.csv`) with 10 random seeds and saves all results under `./test_results/<YYYYMMDD>/demo_dataset/`.

### 2. Override arguments inline

```bash
# Fewer seeds for a quick sanity check
bash demo.sh --seed_iter 2 --train_epochs 50

# Enable Phase-2 ablation study (requires a 4-part ratio)
bash demo.sh --ratios 0.6,0.1,0.2,0.1

# Evaluate against a ground-truth graph
bash demo.sh --gt_path gt_causal.csv --gt_with_lag True
```

### 3. Run directly

```bash
python -u ./run_main.py \
    --root_path  ./dataset/demo/ \
    --data_path  demo_dataset.csv \
    --model_id   my_experiment \
    --d_in       4 \
    --seq_len    10 \
    --pred_len   1 \
    --seed_iter  5
```

---

## Project Structure

```
CACD/
├── run_main.py             # Main entry point (argument parsing, training loop)
├── demo.sh                 # Quick-start shell script
│
├── exp/
│   ├── exp_basic.py        # Base experiment class
│   └── exp_discover.py     # Training / validation / testing loop
│
├── models/
│   └── CACD.py      # Model architecture (dual-branch decoder)
│
├── utils/
│   ├── tools.py            # Core utilities, visualization, causal export
│   ├── metrics.py          # Forecasting + causal discovery metrics
│   └── explain_agc.py      # AGC loss (Attention-Gradient Consistency)
│
└── dataset/
    └── demo/
        └── demo_dataset.csv
```

---

## Dataset Format

Each dataset is a **CSV file with a header row**. Each column is one time-series variable; an optional first column with non-numeric timestamps is automatically excluded from the channel count.

```
X0,X1,X2,X3
0.312,-0.814, 1.203, 0.045
0.298,-0.771, 1.189, 0.061
...
```

**Multiple files** can be processed in one run:

```bash
python run_main.py --data_path TS1.csv TS2.csv TS3.csv ...
```

### Ground-truth causal graph (optional)

Provide a CSV with a header row `src,tgt,lag[,sign]`:

```
src,tgt,lag,sign
0,1,0,+1       # variable 0 → variable 1, no lag,   excitatory
0,2,1,+1       # variable 0 → variable 2, lag = 1,  excitatory
1,3,2,-1       # variable 1 → variable 3, lag = 2,  inhibitory
```

- `src`, `tgt`: 0-indexed channel numbers
- `lag`: transmission delay in samples (`0` = no lag)
- `sign` *(optional)*: `+1` = excitatory, `−1` = inhibitory

---

## Configuration Reference

### Data

| Argument | Default | Description |
|---|---|---|
| `--root_path` | `./dataset/demo/` | Directory containing the data files |
| `--data_path` | `demo_dataset.csv` | One or more CSV filenames |
| `--ratios` | `0.6,0.1,0.3` | Train / val / test split (4 parts enables Phase-2) |
| `--scale` | `True` | Standardize inputs (zero mean, unit variance) |
| `--gt_path` | `None` | Ground-truth graph CSV(s) for metric evaluation |
| `--gt_with_lag` | `True` | `True` = lag-aware evaluation; `False` = collapsed |

### Model architecture

| Argument | Default | Description |
|---|---|---|
| `--d_in` | auto | Number of input channels (auto-detected from CSV header) |
| `--seq_len` | `10` | Look-back window length (samples) |
| `--pred_len` | `1` | Forecast horizon length (samples) |
| `--d_model` | `128` | Hidden dimension |
| `--d_ff` | `256` | Feed-forward network dimension |
| `--n_heads` | `16` | Number of attention heads |
| `--d_layers` | `3` | Number of decoder layers |
| `--kernel_size` | `3` | Causal depthwise-conv kernel size |

### Training

| Argument | Default | Description |
|---|---|---|
| `--train_epochs` | `200` | Maximum training epochs |
| `--batch_size` | `32` | Training batch size |
| `--learning_rate` | `0.001` | Initial learning rate |
| `--lradj` | `TST` | LR schedule: `TST` (OneCycleLR) · `type1` · `type2` · `fixed` |
| `--patience` | `10` | Early-stopping patience |
| `--seed_iter` | `1` | Number of independent random seeds |

### Causal discovery

| Argument | Default | Description |
|---|---|---|
| `--lambda_ch` | `0.05` | Channel-gate sparsity regularization weight |
| `--lambda_agc` | `0.05` | AGC auxiliary loss weight |
| `--cumulative_ratio` | `0.95` | Keep edges covering this fraction of total causal strength |
| `--grad_thresh` | `0.0` | Discard edges with `|∂y/∂x[τ̂]|` below this value |
| `--strength_ratio_thresh` | `0.05` | Discard edges weaker than `max_strength × ratio` |

---

## Output Files

All results are written to `./test_results/<YYYYMMDD>/<model_id>/`:

```
<model_id>/
├── <ts_name>/
│   ├── seed0/                        # Per-seed outputs (if --save_seed_plots)
│   │   └── ...
│   └── SeedAvg_CausalAnalysis/
│       ├── causal_matrix.csv         # Binary adjacency matrix [src → tgt]
│       ├── lag_matrix.csv            # Transmission lag per edge (samples)
│       ├── modulation_matrix.csv     # Direction: +1=excitatory, -1=inhibitory
│       ├── strength_matrix.csv       # Raw causal strength scores
│       ├── strength_matrix_norm.csv  # Globally normalized strengths [0, 1]
│       ├── SeedAvg_delay_causal_triplets.csv  # Full ranked edge list
│       ├── SeedAvg_PredCausalGraph.png        # Predicted graph (no GT needed)
│       ├── CausalGraph_GT_vs_Pred.png         # GT vs predicted (if --gt_path)
│       └── CausalDiscovery_Metrics.csv        # AUROC/F1/lag-acc (if --gt_path)
└── Summary_CausalDiscovery_Metrics.csv        # Mean ± std across all TS files
```

### Triplets CSV columns

The `SeedAvg_delay_causal_triplets.csv` file contains one row per discovered edge:

| Column | Description |
|---|---|
| `src`, `tgt` | Source and target channel indices (0-indexed) |
| `lag` | Estimated transmission delay (samples; minimum = 1) |
| `direction` | `∂y_tgt/∂x_src[τ̂]` — signed gradient at the estimated lag |
| `effect` | `excitatory` or `inhibitory` |
| `causal_strength` | `W_{i,j,τ̂} × |∂y/∂x[τ̂]|` — composite score (always positive) |
| `sign_stability` | Directional reliability across the lag sequence (> 0.5 = trustworthy) |
| `cum_ratio_in_tgt` | Cumulative fraction of target's total incoming strength |
| `selected` | `1` if included by the `cumulative_ratio` threshold |

---

## Model Architecture

```
Input x [B, T, C]
       │
       ▼
  Embedding layer
  ├── K: causal-conv(x) + Time2Vec + channel embedding   [B, C·T, d_model]
  └── V: x             + Time2Vec + channel embedding   [B, C·T, d_model]
       │
  Future queries (two independent branches)
  ├── Q_self  = future_vector_self  × scale_self  [B, C·P, d_model]
  └── Q_cross = future_vector_cross × scale_cross [B, C·P, d_model]
       │
       ▼
  Decoder (N layers)
  ├── Self-attention  branch  ← same-channel mask   (within-channel dynamics)
  └── Cross-attention branch  ← cross-channel mask  (cross-channel causality)
       │    └── Channel gate (STE binary gate, optional)
       │    └── AGC cache → AGC loss during training
       │
  Branch fusion: α·Q_self + (1−α)·Q_cross   (α = sigmoid(branch_gate))
       │
       ▼
  Linear projection [d_model → 1]  →  Output ŷ [B, pred_len, C]
```

**Causal attribution** is extracted post-training:

1. **Attention map** `W[tgt, src, τ]` — normalized cross-attention, summed over pred-tokens and heads
2. **Input gradient** `∂y_tgt/∂x_src[τ]` — batch-averaged sensitivity of each prediction to each past input
3. **Joint effect** `c[tgt, src, τ] = W × |∂y/∂x|` — argmax gives lag τ̂; sign gives direction

---

## Three-Phase Pipeline

| Phase | Trigger | What happens |
|---|---|---|
| **Phase 1** — Multi-seed training | `--is_training 1` | Trains N seeds, averages attention + gradients, extracts causal triplets |
| **Phase 2** — Ablation validation | `--ratios` has 4 parts | Student model ablation study on the verification split to prune FP edges |
| **Phase 3** — GT metric evaluation | `--gt_path` provided | Computes AUROC, AUPRC, F1, Precision, Recall, SHD, and lag accuracy |

---

## Evaluation Against Ground Truth

```bash
# Lag-aware evaluation (default)
python run_main.py \
    --data_path  your_data.csv \
    --gt_path    your_gt.csv   \
    --gt_with_lag True

# Lag-free (collapsed) evaluation
python run_main.py \
    --data_path  your_data.csv \
    --gt_path    your_gt.csv   \
    --gt_with_lag False
```

Reported metrics:

| Metric | Description |
|---|---|
| `auroc` | Area under the ROC curve (edge existence) |
| `auprc` | Area under the precision-recall curve |
| `f1` / `precision` / `recall` | At the `cumulative_ratio` operating point |
| `shd` | Structural Hamming Distance |
| `lag_accuracy` | Fraction of TP edges whose lag is predicted correctly |
| `sign_accuracy` | Fraction of TP edges whose modulation direction is predicted correctly |
