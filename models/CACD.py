"""
Causal Discovery Model Architecture (CACD)

Multi-channel time-series forecasting with a dual-branch decoder:
  - Self-attention  branch: captures same-channel temporal dependencies.
  - Cross-attention branch: captures cross-channel causal relationships.

Key design choices:
  - Causal depthwise convolution in K embedding (no future leakage, no channel mixing).
  - Separate learnable future vectors for self / cross attention queries.
  - Per-channel learnable scaling factors for query intensities.
  - Optional channel-gate regularization (STE-based binary gating).
  - Optional AGC (Attention-Gradient Consistency) explainability cache.
"""

# CACD

This repository hosts the official implementation of the manuscript:

**Cross-Attention Causal Discovery for Asynchronous EEG Functional Network Modeling in Affective Disorders**

The complete source code is temporarily unavailable during the manuscript review process. Upon publication of the article, we will release the full implementation, including the CACD model, data-processing pipeline, training and evaluation scripts, and instructions for reproducing the reported results.

For questions regarding the manuscript or code release, please contact the corresponding authors.
