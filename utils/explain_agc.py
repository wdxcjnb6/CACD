"""
utils/explain_agc.py
====================
Attention-Gradient Consistency (AGC) Module

During training: uses value-gradient attribution (v ⊙ ∇_v y) as a soft
supervision signal to align cross-attention weights with gradient-based
causal effects.

Design conventions:
  - Applied only to the cross-attention branch (cross-channel causality);
    the self-attention branch (within-channel temporal dynamics) is unconstrained.
  - Normalization is performed only over the valid cross-attention region
    (the (C-1)*seq_len positions that do NOT correspond to channel ch_i),
    preventing the positions zeroed out by the cross-mask from polluting
    the normalization direction.
  - E_i is treated as a stop-gradient target: it serves as a supervision
    signal only, without propagating gradients back through itself.

During inference:
  - Causal strength is determined by the cumulative mean of cross-attention
    weights W [C, C, T].
  - Lag estimation uses the argmax of the joint effect
    |W_{i,j,τ} × ∂y_i/∂x_j[τ]|, computed in the original signal space
    for clearer physical interpretation.
"""

import torch
import torch.nn.functional as F


def compute_agc_grad_effect(v, y_scalar, create_graph=True):
    """
    Compute the value-gradient attribution: E_i = Σ_{h,d} v ⊙ ∇_v y_i

    This is the Gradient×Input attribution applied in the value space,
    quantifying the actual contribution of each Channel-Lag Token (CLT)
    to the prediction of target channel i.
    Positive values indicate excitatory influence; negative indicate inhibitory.

    Args:
        v            (Tensor): Cached value vectors from the last decoder layer,
                               shape (B, S, H, D_h). S = C * seq_len, where each
                               position corresponds to one CLT.
        y_scalar     (Tensor): Scalar prediction for target channel i, shape (B,).
        create_graph (bool):   Whether to retain the second-order graph
                               (required when this is called inside the AGC loss).

    Returns:
        E_i (Tensor): Gradient effect per CLT, shape (B, S).
                      Positive = excitatory, negative = inhibitory.
    """
    y = y_scalar.sum()
    grads = torch.autograd.grad(
        outputs=y,
        inputs=v,
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=False,
    )[0]  # (B, S, H, D_h)

    # Gradient×Input; sum over head and d_h dimensions
    return (v * grads).sum(dim=(2, 3))  # (B, S)


def agc_consistency_loss(attn_cross, grad_effect, ch_idx, pred_len, seq_len, C, eps=1e-8):
    """
    AGC consistency loss: align cross-attention weights with value-gradient effects.

    Core design:
      1. theta_i: mean of cross-attention weights over pred_len and head dimensions
                  → shape (B, S). Due to the cross-mask, theta_i is non-zero only
                  at positions belonging to channels other than ch_i.
      2. E_i:     grad_effect, shape (B, S). Non-zero over all S positions.
      3. Normalization is restricted to the valid cross-attention region
         ((C-1)*seq_len positions, excluding channel ch_i). This ensures both
         vectors share the same support domain, making the MSE meaningful.

    Handling the sign mismatch:
      theta_i is always non-negative (softmax output), while grad_effect can be
      positive or negative. Directly normalizing E_i would create an MSE floor
      that can never reach zero, producing incorrect gradient directions.
      Solution: take |E_i| before normalization — align "which positions matter",
      leaving the sign (excitatory vs inhibitory) to the input-gradient analysis.

    Args:
        attn_cross  (Tensor): Cross-attention weights, shape (B, P, H, S).
                              P = C * pred_len, S = C * seq_len.
        grad_effect (Tensor): E_i = compute_agc_grad_effect(v, y_i), shape (B, S).
        ch_idx      (int):    Target channel index i (0-indexed).
        pred_len    (int):    Prediction horizon length.
        seq_len     (int):    Input sequence length.
        C           (int):    Number of channels.
        eps         (float):  Numerical stability term for normalization.

    Returns:
        loss (Tensor): Scalar MSE loss.
                       MSE(normalize(theta_i), normalize(|E_i|).detach())
                       Equivalent to maximizing cos(theta_i, |E_i|), since
                       MSE = 2 * (1 - cos(theta_i, |E_i|)).
    """
    # Step 1: Extract theta_i from cross-attention
    st = ch_idx * pred_len
    ed = (ch_idx + 1) * pred_len
    attn_seg = attn_cross[:, st:ed, :, :]        # (B, pred_len, H, S)
    theta    = attn_seg.mean(dim=2).mean(dim=1)  # (B, S)

    # Step 2: Restrict to positions belonging to channels other than ch_idx.
    # The cross-mask forces theta to zero at ch_idx's own channel positions.
    # Normalizing over all S would dilute the norm with those zeros,
    # misaligning the direction of the two vectors.
    cross_cols = []
    for s in range(C):
        if s != ch_idx:
            cross_cols += list(range(s * seq_len, (s + 1) * seq_len))

    theta_c = theta[:, cross_cols]        # (B, (C-1)*seq_len)
    grad_c  = grad_effect[:, cross_cols]  # (B, (C-1)*seq_len)

    # Step 3: Normalize and compute MSE (E_i treated as stop-gradient)
    t_norm = theta_c / (theta_c.norm(dim=1, keepdim=True) + eps)
    g_norm = grad_c.abs() / (grad_c.abs().norm(dim=1, keepdim=True) + eps)

    return F.mse_loss(t_norm, g_norm.detach())
