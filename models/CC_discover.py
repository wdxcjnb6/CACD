"""
Causal Discovery Model Architecture (CC_discover)

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import repeat

import utils.tools
from utils.tools import CausalDepthwiseConv1d, GEGLU
from .time2vec import Time2Vec


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """
    Top-level wrapper for CC_discover.

    Handles the (B, T, C) <-> (B, C, T) tensor layout convention expected by
    the rest of the codebase and delegates all computation to Model_backbone.
    """

    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.model = Model_backbone(
            d_in=args.d_in,
            t_in=args.t_in,
            time_emb_dim=args.time_emb_dim,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            n_layers=args.d_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            dropout=args.dropout,
            attn_dropout=args.attn_dropout,
            revin_flag=args.revin_flag,
            norm_flag=args.norm_flag,
            res_attention=args.res_attention,
            kernel_size=args.kernel_size,
            lambda_ch=args.lambda_ch,
            share_ch_mask=args.share_ch_mask,
            **kwargs,
        )

    def forward(self, x, x_stamp, y, y_stamp, return_attn=False):
        """
        Args:
            x        (Tensor): Input sequence,       shape (B, T, C).
            x_stamp  (Tensor): Time stamps for x.
            y        (Tensor): Target sequence,       shape (B, pred_len, C).
            y_stamp  (Tensor): Time stamps for y.
            return_attn (bool): If True, also return per-layer attention maps.

        Returns:
            out      (Tensor): Predictions,           shape (B, pred_len, C).
            attn_list (list | None): Per-layer attention maps when return_attn=True.
        """
        # External convention: (B, T, C); internal: (B, C, T)
        x = x.permute(0, 2, 1)
        y = y.permute(0, 2, 1)

        if return_attn:
            out, attn_list = self.model(x, x_stamp, y, y_stamp, return_attn=True)
        else:
            out = self.model(x, x_stamp, y, y_stamp, return_attn=False)
            attn_list = None

        # Restore external convention: (B, C, pred_len) -> (B, pred_len, C)
        out = out.permute(0, 2, 1)

        if return_attn:
            return out, attn_list
        return out

    def regularization(self):
        """Compute channel-gate sparsity regularization loss."""
        if hasattr(self.model, "regularization"):
            return self.model.regularization()
        return 0.0


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Model_backbone(nn.Module):
    """
    Core model backbone.

    Responsibilities:
      - Optional RevIN or standard (mean/std) normalization.
      - Embedding + multi-layer decoder (Embedding_layer).
      - Linear projection from d_model to a scalar per channel.
      - Denormalization.
    """

    def __init__(
        self,
        share_ch_mask: bool,
        kernel_size: int,
        d_in: int,
        t_in: int,
        seq_len: int,
        time_emb_dim: int,
        pred_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        res_attention: bool = True,
        revin_flag: bool = False,
        norm_flag: bool = False,
        lambda_ch: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_vars   = d_in
        self.seq_len  = seq_len
        self.pred_len = pred_len

        # Embedding + decoder stack
        self.backbone = Embedding_layer(
            d_in=d_in,
            t_in=t_in,
            time_emb_dim=time_emb_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            res_attention=res_attention,
            kernel_size=kernel_size,
            lambda_ch=lambda_ch,
            share_ch_mask=share_ch_mask,
            **kwargs,
        )

        # Project per-channel decoder output to a scalar prediction
        self.value_proj = nn.Linear(d_model, 1)

        # Normalization settings
        self.revin_flag  = revin_flag
        self.norm_flag   = norm_flag
        self.revin_layer = utils.tools.RevIN(num_features=d_in)

    def forward(self, x, x_stamp, y, y_stamp, return_attn=False):
        """
        Args:
            x        (Tensor): shape (B, C, T).
            x_stamp  (Tensor): Time stamps for x.
            y        (Tensor): shape (B, C, pred_len).
            y_stamp  (Tensor): Time stamps for y.
            return_attn (bool): If True, also return attention maps.

        Returns:
            z (Tensor): Predictions, shape (B, C, pred_len).
            attn_list (list | None).
        """
        # --- Normalization ---
        if self.revin_flag:
            x = self.revin_layer(x, "norm")
        elif self.norm_flag:
            mean = x.mean(2, keepdim=True)
            std  = torch.sqrt(torch.var(x, dim=2, keepdim=True, unbiased=False) + 1e-5)
            x    = (x - mean) / std

        # --- Encoder-Decoder ---
        z, attn_list = self.backbone(x, x_stamp, y, y_stamp, return_attn=return_attn)

        # Project: (B, C, pred_len, d_model) -> (B, C, pred_len)
        z = self.value_proj(z).squeeze(-1)

        # --- Denormalization ---
        if self.revin_flag:
            z = self.revin_layer(z, "denorm")
        elif self.norm_flag:
            z = z * std[:, :, 0:1] + mean[:, :, 0:1]

        if return_attn:
            return z, attn_list
        return z

    def regularization(self):
        """Propagate channel-gate regularization from the decoder."""
        if hasattr(self.backbone, "decoder") and hasattr(self.backbone.decoder, "regularization"):
            return self.backbone.decoder.regularization()
        return 0.0


# ---------------------------------------------------------------------------
# Embedding layer
# ---------------------------------------------------------------------------

class Embedding_layer(nn.Module):
    """
    Builds token embeddings for past (encoder) and future (decoder query) tokens,
    then passes them to the Decoder.

    Token representation = value embedding + temporal (Time2Vec) + spatial (channel) embedding.

    Past K tokens: concatenate [raw_value, causal_conv_output] to capture local
    context without mixing channels or leaking future information.

    Past V tokens: use raw values only, so that input-gradient ∂y/∂V[t] maps
    cleanly to a causal lag of t time steps.

    Future queries are split into two independent branches:
      - Self  branch: focuses on same-channel temporal dependencies.
      - Cross branch: focuses on cross-channel causal relationships.
    Each branch has its own learned future vector and per-channel scale factor.
    """

    def __init__(
        self,
        d_in: int,
        t_in: int,
        time_emb_dim: int,
        seq_len: int,
        pred_len: int,
        share_ch_mask: bool,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        kernel_size: int = 3,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        res_attention: bool = True,
        lambda_ch: float = 0.0,
        pos_t2v_scale: float = 1.0,
        pos_t2v_normalize: bool = False,
        **kwargs,
    ):
        super().__init__()

        # K embedding: input dim = 2 ([raw_value, causal_conv_output])
        # The causal conv captures local temporal context without channel mixing
        # or future leakage, giving each K token richer local information.
        self.causal_conv = CausalDepthwiseConv1d(d_in, kernel_size)
        self.k_emb = nn.Linear(2, d_model, bias=True)

        # V embedding: raw values only — keeps ∂y/∂V[t] interpretable as causal lag
        self.v_emb = nn.Linear(1, d_model, bias=True)

        # Separate Q projections for self / cross branches
        self.q_self_emb  = nn.Linear(1, d_model, bias=True)  # self-attention query
        self.q_cross_emb = nn.Linear(1, d_model, bias=True)  # cross-attention query

        self.layer_norm = nn.LayerNorm(d_model)
        self.seq_len    = seq_len
        self.pred_len   = pred_len

        # Temporal positional encoding (Time2Vec)
        self.pos_t2v = Time2Vec(input_dim=1, embed_dim=d_model, act_function=torch.sin)

        # Spatial (channel identity) embedding
        self.space_emb = nn.Embedding(num_embeddings=d_in, embedding_dim=d_model)
        self.dropout   = nn.Dropout(dropout)

        # Independent future vectors for self / cross branches.
        # Splitting the vectors prevents the self branch from dominating
        # the cross branch's gradients during training.
        self.future_vector_self  = nn.Parameter(torch.randn(d_in, pred_len) * 0.02)
        self.future_vector_cross = nn.Parameter(torch.randn(d_in, pred_len) * 0.02)

        # Per-channel learnable scale factors for query intensities
        self.channel_scale_self  = nn.Parameter(torch.ones(d_in, 1))
        self.channel_scale_cross = nn.Parameter(torch.ones(d_in, 1))

        # Decoder
        self.decoder = Decoder(
            seq_len=seq_len,
            d_model=d_model,
            n_heads=n_heads,
            pred_len=pred_len,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            res_attention=res_attention,
            n_layers=n_layers,
            d_in=d_in,
            lambda_ch=lambda_ch,
            share_ch_mask=share_ch_mask,
        )

    def forward(self, x, x_stamp, y, y_stamp, return_attn=False) -> Tensor:
        """
        Args:
            x        (Tensor): shape (B, C, T).
            x_stamp  (Tensor): Time stamps for x (unused; kept for API consistency).
            y        (Tensor): shape (B, C, pred_len) (unused; kept for API consistency).
            y_stamp  (Tensor): Time stamps for y (unused; kept for API consistency).
            return_attn (bool): If True, return attention maps.

        Returns:
            Future_output (Tensor): shape (B, C, pred_len, d_model).
            attn_list (list | None).
        """
        B, C_in, T = x.shape
        assert T == self.seq_len, f"Expected seq_len={self.seq_len}, got T={T}"

        # ---- Past token processing ----
        # Causal conv: each position sees only current and past (kernel_size-1) frames,
        # with no channel mixing and no future leakage.
        x_conv = self.causal_conv(x)  # (B, C, T)

        # Temporal (Time2Vec) positional embedding for past tokens
        t_idx_past       = torch.arange(self.seq_len, device=x.device).float().view(1, self.seq_len, 1)
        P_past           = self.pos_t2v(t_idx_past)                                          # (1, T, d_model)
        P_emb_past       = P_past.unsqueeze(1).expand(B, C_in, self.seq_len, -1)             # (B, C, T, d_model)

        # Spatial (channel) embedding for past tokens
        ch_idx           = torch.arange(C_in, device=x.device).long().view(1, C_in, 1)
        S_emb_past       = self.space_emb(ch_idx).expand(B, C_in, self.seq_len, -1)         # (B, C, T, d_model)

        # K: concatenate [raw_value, causal_conv_output] — local context without leakage
        x_k              = torch.cat([x.unsqueeze(-1), x_conv.unsqueeze(-1)], dim=-1)        # (B, C, T, 2)
        Past_K           = self.k_emb(x_k) + P_emb_past + S_emb_past                       # (B, C, T, d_model)
        Past_K           = self.layer_norm(Past_K.reshape(B, C_in * self.seq_len, -1))      # (B, C*T, d_model)

        # V: raw values only — gradient ∂y/∂V[t] isolates causal effect at lag t
        TV_V             = self.v_emb(x.unsqueeze(-1))                                       # (B, C, T, d_model)
        Past_V           = self.layer_norm(
            (TV_V + P_emb_past + S_emb_past).reshape(B, C_in * self.seq_len, -1)
        )                                                                                     # (B, C*T, d_model)

        # Shared temporal and spatial embeddings for future tokens
        t_idx_future     = torch.arange(
            self.seq_len, self.seq_len + self.pred_len, device=x.device
        ).float().view(1, self.pred_len, 1)
        P_future         = self.pos_t2v(t_idx_future)                                        # (1, P, d_model)
        P_emb_future     = P_future.unsqueeze(1).expand(B, C_in, self.pred_len, -1)         # (B, C, P, d_model)
        S_emb_future     = self.space_emb(ch_idx).expand(B, C_in, self.pred_len, -1)        # (B, C, P, d_model)

        # ---- Self-branch future query ----
        ft_self          = (self.future_vector_self * self.channel_scale_self
                            ).unsqueeze(0).expand(B, C_in, self.pred_len)                    # (B, C, P)
        TV_q_self        = self.q_self_emb(ft_self.unsqueeze(-1))                            # (B, C, P, d_model)
        Future_query_self = self.layer_norm(
            (TV_q_self + P_emb_future + S_emb_future).reshape(B, C_in * self.pred_len, -1)
        )                                                                                     # (B, C*P, d_model)

        # ---- Cross-branch future query ----
        ft_cross         = (self.future_vector_cross * self.channel_scale_cross
                            ).unsqueeze(0).expand(B, C_in, self.pred_len)                    # (B, C, P)
        TV_q_cross       = self.q_cross_emb(ft_cross.unsqueeze(-1))                         # (B, C, P, d_model)
        Future_query_cross = self.layer_norm(
            (TV_q_cross + P_emb_future + S_emb_future).reshape(B, C_in * self.pred_len, -1)
        )                                                                                     # (B, C*P, d_model)

        # ---- Decoder ----
        Future_output, attn_list = self.decoder(
            seq_K=Past_K,
            q_self=Future_query_self,
            q_cross=Future_query_cross,
            seq_V=Past_V,
            return_attn=return_attn,
        )

        # Reshape: (B, C*P, d_model) -> (B, C, P, d_model)
        Future_output = Future_output.view(B, C_in, self.pred_len, -1)

        if return_attn:
            return Future_output, attn_list
        return Future_output, None


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Stack of DecoderLayer modules.

    Maintains separate self- and cross-attention query tensors across layers,
    and combines them at the end via a learned sigmoid gate (branch_gate).
    The gating prevents the self branch from overwhelming cross-branch gradients.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        pred_len: int,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        res_attention: bool = False,
        n_layers: int = 3,
        d_in: int = 1,
        lambda_ch: float = 0.0,
        share_ch_mask: bool = True,
    ):
        super().__init__()
        self.n_layers     = n_layers
        self.res_attention = res_attention

        # Optional shared channel gate across all layers (initialized near 0 → sigmoid ≈ 0.5)
        shared_ch_gate = None
        if share_ch_mask and (lambda_ch > 0):
            shared_ch_gate = nn.Parameter(torch.zeros(d_in, d_in))

        self.layers = nn.ModuleList([
            DecoderLayer(
                seq_len=seq_len,
                d_model=d_model,
                pred_len=pred_len,
                n_heads=n_heads,
                d_ff=d_ff,
                attn_dropout=attn_dropout,
                dropout=dropout,
                res_attention=res_attention,
                shared_ch_gate=shared_ch_gate,
                d_in=d_in,
                lambda_ch=lambda_ch,
            )
            for _ in range(n_layers)
        ])

        # Learned scalar gate for combining self and cross branch outputs
        self.branch_gate = nn.Parameter(torch.zeros(1))

    def forward(self, seq_K, q_self, q_cross, seq_V, return_attn=False):
        """
        Args:
            seq_K    (Tensor): Keys from encoder,          shape (B, C*T, d_model).
            q_self   (Tensor): Self-branch queries,        shape (B, C*P, d_model).
            q_cross  (Tensor): Cross-branch queries,       shape (B, C*P, d_model).
            seq_V    (Tensor): Values from encoder,        shape (B, C*T, d_model).
            return_attn (bool): Collect per-layer attention maps.

        Returns:
            final_q  (Tensor): Combined output,            shape (B, C*P, d_model).
            attn_list (list[tuple] | None): (self_attn, cross_attn) per layer.
        """
        attn_list  = []
        prev_self  = None
        prev_cross = None

        for layer in self.layers:
            if self.res_attention:
                q_self, q_cross, prev_self, prev_cross = layer(
                    seq_K, q_self, q_cross, seq_V,
                    prev=(prev_self, prev_cross),
                    return_attn=return_attn,
                )
            else:
                q_self, q_cross = layer(
                    seq_K, q_self, q_cross, seq_V,
                    prev=None,
                    return_attn=return_attn,
                )

            if return_attn:
                attn_list.append(layer.attn_maps)

        # Weighted combination of the two branches
        alpha   = torch.sigmoid(self.branch_gate)
        final_q = alpha * q_self + (1 - alpha) * q_cross

        if return_attn:
            return final_q, attn_list
        return final_q, None

    def regularization(self):
        """Aggregate channel-gate regularization from all layers."""
        reg = 0.0
        for layer in self.layers:
            reg = reg + layer.regularization()
        return reg


# ---------------------------------------------------------------------------
# Decoder layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """
    Single decoder layer with a dual-branch architecture.

    Self-attention  branch: Q queries only attend to K/V tokens from the
        *same* channel  → captures temporal self-dependencies.

    Cross-attention branch: Q queries only attend to K/V tokens from
        *different* channels → captures causal cross-channel relationships.

    The channel gate (ch_gate) is only applied to the cross-attention branch;
    it has no effect on same-channel softmax distributions and is therefore
    omitted from the self-attention branch.
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        pred_len: int,
        n_heads: int,
        d_ff: int = 256,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        res_attention: bool = False,
        shared_ch_gate: nn.Parameter = None,
        d_in: int = 1,
        lambda_ch: float = 0.0,
    ):
        super().__init__()
        self.seq_len      = seq_len
        self.pred_len     = pred_len
        self.res_attention = res_attention
        self.lambda_ch    = lambda_ch

        # When share_ch_mask=False each layer gets its own independent gate;
        # when share_ch_mask=True the same Parameter is injected from Decoder.
        if (shared_ch_gate is None) and (lambda_ch > 0):
            shared_ch_gate = nn.Parameter(torch.zeros(d_in, d_in))

        # ---- Self-attention branch ----
        # ch_gate is not used here: within a single channel the softmax
        # distribution is unaffected by channel-level gating.
        self.self_attn = CausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            res_attention=res_attention,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qkv_bias=bias,
            d_in=d_in,
            seq_len=seq_len,
            use_ch_gate=False,
            shared_ch_gate=None,
        )
        self.norm_self    = nn.LayerNorm(d_model)
        self.dropout_self = nn.Dropout(dropout)

        # ---- Cross-attention branch ----
        self.cross_attn = CausalAttention(
            d_model=d_model,
            n_heads=n_heads,
            res_attention=res_attention,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            qkv_bias=bias,
            d_in=d_in,
            seq_len=seq_len,
            use_ch_gate=(lambda_ch > 0),
            shared_ch_gate=shared_ch_gate,
        )
        self.norm_cross    = nn.LayerNorm(d_model)
        self.dropout_cross = nn.Dropout(dropout)

        # ---- Feed-forward networks ----
        # GEGLU activation halves the intermediate dimension, so the second
        # linear projects from d_ff // 2 back to d_model.
        self.ffn_self = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, d_model, bias=bias),
        )
        self.norm_ffn_self    = nn.LayerNorm(d_model)
        self.dropout_ffn_self = nn.Dropout(dropout)

        self.ffn_cross = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, d_model, bias=bias),
        )
        self.norm_ffn_cross    = nn.LayerNorm(d_model)
        self.dropout_ffn_cross = nn.Dropout(dropout)

        self.attn_maps = None

    def forward(
        self,
        seq_K: Tensor,
        q_self: Tensor,
        q_cross: Tensor,
        seq_V: Tensor,
        prev=None,
        return_attn: bool = False,
    ):
        """
        Args:
            seq_K    (Tensor): Keys,                shape (B, S_tokens, d_model).
            q_self   (Tensor): Self-branch queries, shape (B, P_tokens, d_model).
            q_cross  (Tensor): Cross-branch queries,shape (B, P_tokens, d_model).
            seq_V    (Tensor): Values,              shape (B, S_tokens, d_model).
            prev     (tuple | None): (prev_self_logits, prev_cross_logits) for
                residual attention; None disables the feature.
            return_attn (bool): Cache attention maps in self.attn_maps.

        Returns:
            When res_attention=False: (q_self, q_cross)
            When res_attention=True:  (q_self, q_cross, self_logits, cross_logits)
        """
        B, P_tokens, _ = q_self.shape
        S_tokens = seq_K.size(1)
        device   = q_self.device

        # ---- Token-level same/cross-channel masks ----
        # Each future token belongs to a channel: token_idx // pred_len
        # Each past token belongs to a channel:   token_idx // seq_len
        pred_ch = torch.arange(P_tokens, device=device) // self.pred_len
        seq_ch  = torch.arange(S_tokens, device=device) // self.seq_len

        # self_mask[p, s] = True iff query p and key s share the same channel
        self_mask  = (
            pred_ch.view(1, P_tokens, 1) == seq_ch.view(1, 1, S_tokens)
        ).unsqueeze(2)  # (1, P, 1, S)
        cross_mask = ~self_mask

        prev_self  = None if prev is None else prev[0]
        prev_cross = None if prev is None else prev[1]

        # ---- Self-attention branch ----
        x_self_in  = self.norm_self(q_self)
        self_out, self_attn_w, self_logits = self.self_attn(
            x_self_in, seq_K, seq_V, attn_mask=self_mask, prev=prev_self
        )
        q_self = q_self + self.dropout_self(self_out)
        q_self = q_self + self.dropout_ffn_self(self.ffn_self(self.norm_ffn_self(q_self)))

        # ---- Cross-attention branch ----
        x_cross_in = self.norm_cross(q_cross)
        cross_out, cross_attn_w, cross_logits = self.cross_attn(
            x_cross_in, seq_K, seq_V, attn_mask=cross_mask, prev=prev_cross
        )
        q_cross = q_cross + self.dropout_cross(cross_out)
        q_cross = q_cross + self.dropout_ffn_cross(self.ffn_cross(self.norm_ffn_cross(q_cross)))

        if return_attn:
            self.attn_maps = (self_attn_w, cross_attn_w)

        if self.res_attention:
            return q_self, q_cross, self_logits, cross_logits
        return q_self, q_cross

    def regularization(self) -> Tensor:
        """
        Channel-gate entropy regularization.

        Encourages gate values toward 0 or 1 (binary connectivity).
        Applied only to the cross-attention branch; the self branch has no gate.
        """
        if self.lambda_ch == 0.0 or self.cross_attn.ch_gate is None:
            return 0.0
        return self.cross_attn.regularization(self.lambda_ch)


# ---------------------------------------------------------------------------
# Causal attention
# ---------------------------------------------------------------------------

class CausalAttention(nn.Module):
    """
    Multi-head attention with optional channel gating and residual attention.

    Features:
      - Scaled dot-product attention with an explicit boolean attention mask.
      - Channel-level gating via a learnable [C, C] matrix with
        Straight-Through Estimator (STE) for binary gradients.
      - Residual attention: raw logits are accumulated across layers.
      - AGC explainability cache: stores (attn, v) for computing the
        Attention-Gradient Consistency auxiliary loss in the training loop.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        res_attention: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        qkv_bias: bool = True,
        d_in: int = None,
        seq_len: int = None,
        use_ch_gate: bool = False,
        shared_ch_gate: torch.Tensor = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads      = n_heads
        self.d_h          = d_model // n_heads
        self.scale        = self.d_h ** -0.5
        self.res_attention = res_attention

        self.q_proj = nn.Linear(d_model, n_heads * self.d_h, bias=qkv_bias)
        self.k_proj = nn.Linear(d_model, n_heads * self.d_h, bias=qkv_bias)
        self.v_proj = nn.Linear(d_model, n_heads * self.d_h, bias=qkv_bias)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.to_out       = nn.Sequential(
            nn.Linear(n_heads * self.d_h, d_model),
            nn.Dropout(proj_dropout),
        )

        self.d_in    = d_in
        self.seq_len = seq_len
        self.use_ch_gate = use_ch_gate

        # Channel gate: shared across layers when provided, independent otherwise
        if shared_ch_gate is not None:
            self.ch_gate = shared_ch_gate
        elif use_ch_gate and d_in:
            self.ch_gate = nn.Parameter(torch.zeros(d_in, d_in))
        else:
            self.ch_gate = None

        # AGC cache — populated only when cache_expl=True (set by the training loop)
        self.cache_expl = False
        self.last_attn  = None
        self.last_v     = None

    def forward(
        self,
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        attn_mask: Tensor,
        prev: Tensor = None,
    ):
        """
        Args:
            Q         (Tensor): Queries, shape (B, P, d_model).
            K         (Tensor): Keys,    shape (B, S, d_model).
            V         (Tensor): Values,  shape (B, S, d_model).
            attn_mask (Tensor): Boolean mask (B or 1, P, 1, S); True = attend.
            prev      (Tensor | None): Previous logits (B, P, H, S) for
                residual attention. None disables the feature.

        Returns:
            out        (Tensor): shape (B, P, d_model).
            attn       (Tensor): Normalized attention weights, shape (B, P, H, S).
            raw_scores (Tensor | None): Pre-softmax logits (only when res_attention=True).
        """
        B, P, _ = Q.shape
        S = K.size(1)
        H = self.n_heads

        # Q/K/V projections
        q = self.q_proj(Q).view(B, P, H, self.d_h)
        k = self.k_proj(K).view(B, S, H, self.d_h)
        v = self.v_proj(V).view(B, S, H, self.d_h)

        # Scaled dot-product scores: (B, P, H, S)
        raw_scores = torch.einsum("bphd,bshd->bphs", q, k) * self.scale

        # Residual attention: accumulate logits from previous layer
        if self.res_attention and prev is not None:
            raw_scores = raw_scores + prev

        # Channel gate (STE): learn which source channels to attend
        if self.use_ch_gate and self.ch_gate is not None:
            pred_len_val = P // self.d_in
            pred_ch      = torch.arange(P, device=Q.device) // pred_len_val
            seq_ch       = torch.arange(S, device=Q.device) // self.seq_len

            s        = torch.sigmoid(self.ch_gate)
            hard     = (s >= 0.5).float()
            gate_ch  = hard + (s - s.detach())            # STE: gradient flows through s
            gate_full = gate_ch[pred_ch][:, seq_ch].view(1, P, 1, S)

            # Only modify finite scores to avoid NaN propagation
            raw_scores = torch.where(
                torch.isfinite(raw_scores),
                raw_scores * gate_full,
                raw_scores,
            )

        # Numerically stable masked softmax
        max_scores    = raw_scores.max(dim=-1, keepdim=True)[0]
        stable_scores = raw_scores - max_scores
        masked_scores = stable_scores.masked_fill(~attn_mask, -1e9)
        attn          = F.softmax(masked_scores, dim=-1)
        attn          = attn * attn_mask.to(attn.dtype)

        # Renormalize to handle fully-masked rows gracefully
        den  = attn.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        attn = attn / den
        attn = self.attn_dropout(attn)

        # AGC cache (populated only when enabled by the training loop)
        if self.cache_expl:
            self.last_attn = attn
            self.last_v    = v
        else:
            self.last_attn = None
            self.last_v    = None

        # Weighted sum over values
        out = torch.einsum("bphs,bshd->bphd", attn, v)
        out = out.reshape(B, P, H * self.d_h)
        out = self.to_out(out)

        if self.res_attention:
            return out, attn, raw_scores
        return out, attn, None

    def regularization(self, lambda_ch: float) -> Tensor:
        """
        Channel-gate entropy regularization.

        Penalizes gate values near 0.5 (maximum entropy), pushing them toward
        0 (prune connection) or 1 (keep connection).

        Args:
            lambda_ch (float): Regularization weight.

        Returns:
            Tensor: Scalar regularization loss.
        """
        if not self.use_ch_gate or self.ch_gate is None or lambda_ch == 0.0:
            return 0.0

        s       = torch.sigmoid(self.ch_gate)
        reg_ch  = (s * (1 - s)).mean()   # maximized at s=0.5, minimized at s=0 or 1
        return lambda_ch * reg_ch
