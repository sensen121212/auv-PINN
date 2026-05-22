# =============================================================================
# model.py  鈥? Dynamics-Informed Neural Network for AUV Trajectory Prediction
# =============================================================================
"""
Production-grade Physics-Informed Neural Network (PINN) for AUV trajectory
prediction under degraded sensor conditions (DVL/IMU dropout, impulse noise).

Architecture Revision History (v2.0 鈥?Deep Theoretical Refactoring):
    Breaking changes from v1.x are documented in-line with [BREAKING] tags.

Core Architectural Innovations (v2.0):
    1. **Fossen Dynamics-Informed Loss** (replaces kinematic-only residual)
       Enforces the 6-DOF rigid-body equation:
           M路v虈 + C(v)路v + D(v)路v = 蟿
       with learnable hydrodynamic coefficient matrices M, C(v), D(v).
       The dataset provides propulsion thrust and control-surface commands.
       Controlled modes construct a simplified generalized force vector
       蟿=[X,0,0,0,M,N] from thrust_net_N, stern_rad, and rudder_rad.
       The 蟿≈0 passive-damping form is retained only as a no-control
       ablation.

    2. **Unified Information Bottleneck** (replaces ConfidenceGate)
       [BREAKING] ConfidenceGate is abolished. All uncertainty handling is
       consolidated into the loss layer via maximum-likelihood homoscedastic
       uncertainty weighting (Kendall et al., CVPR 2018):
           L = (1/2蟽虏_data)路L_data + (1/2蟽虏_phy)路L_phy + log(蟽_data路蟽_phy)
       蟽 parameters live in log-variance space for numerical stability.

    3. **Rotary Position Embedding (RoPE)** (replaces sinusoidal PE)
       [BREAKING] Absolute positional encoding is abolished. RoPE injects
       relative position information directly into Q/K dot-products,
       preserving translational invariance for sliding-window inference.
       Zero additional learnable parameters introduced.
       Reference: Su et al., "RoFormer: Enhanced Transformer with Rotary
       Position Embedding", 2021.

    4. **Strict Attention Masking** (causal 鈯?validity)
       Self-attention explicitly combines a causal mask (upper-triangular -鈭?
       with a data-validity mask derived from the sensor dropout indicator.
       This guarantees that invalid (NaN-filled / zero-padded) frames receive
       exactly zero attention weight after softmax, preventing hidden
       information pollution.

Mathematical Notation:
    B = batch size, T = sequence length, F = n_features, P = pred_len,
    D = d_model, H = nhead, d_h = D/H.

References:
    [1] Fossen, T.I., "Handbook of Marine Craft Hydrodynamics and Motion
        Control", Wiley, 2011.  (Chapters 6鈥?: rigid-body & hydrodynamic
        coefficient modelling)
    [2] Kendall, A., Gal, Y., & Cipolla, R., "Multi-Task Learning Using
        Uncertainty to Weigh Losses for Scene Geometry and Semantics",
        CVPR 2018.
    [3] Su, J. et al., "RoFormer: Enhanced Transformer with Rotary Position
        Embedding", arXiv 2104.09864, 2021.
    [4] Raissi, M. et al., "Physics-Informed Neural Networks: A Deep Learning
        Framework for Solving Forward and Inverse Problems Involving Nonlinear
        Partial Differential Equations", JCP 2019.
    [5] Vaswani, A. et al., "Attention Is All You Need", NeurIPS 2017.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class ModelConfig:
    """Immutable configuration for the Dynamics-Informed PINN.

    All hyper-parameters that affect the computational graph are collected here
    so that a single serialised ``ModelConfig`` suffices to reconstruct the
    model without ambiguity.

    Attributes:
        n_features: Number of input features per timestep (default 12:
            x, y, z, vn, ve, vu, roll, pitch, yaw, ax, ay, az).
        seq_len: Input sequence length (historical window).
        pred_len: Prediction horizon length (future steps).
        d_model: Transformer hidden dimension.  Must be divisible by
            ``nhead`` **and** by 2 (RoPE requires even head-dim).
        nhead: Number of attention heads.
        num_encoder_layers: Depth of Transformer encoder stack.
        dim_feedforward: Width of position-wise FFN.
        dropout: Dropout probability applied after attention & FFN.
        n_dof: Degrees of freedom for Fossen dynamics (3 for translational
            surge/sway/heave; set to 6 for full 6-DOF if angular rates are
            available in the prediction target).
        init_log_var_data: Initial value for log(蟽虏_data).
        init_log_var_phy: Initial value for log(蟽虏_phy).
        init_log_var_dyn: Initial value for log(蟽虏_dyn) 鈥?dynamics residual.
    """
    n_features: int = 12  # 鏋佸潗鏍?绌洪棿鍧愭爣/閫熷害绛夌壒寰佺殑鎬绘暟閲忥紝榛樿涓?2缁?
    seq_len: int = 20  # 杈撳叆搴忓垪闀垮害锛堝巻鍙茶娴嬬獥鍙ｅぇ灏廡锛?
    pred_len: int = 10  # Prediction horizon for the main paper experiment.
    d_model: int = 128  # Transformer鐨勯殣钘忓眰缁村害(D)
    nhead: int = 8  # 澶氬ご娉ㄦ剰鍔涚殑澶存暟(H)
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    n_dof: int = 3  # AUV鐨勮嚜鐢卞害鏁扮洰 (3琛ㄧず绾甸銆佹í椋樸€佸瀭鑽′綅绉伙紝6琛ㄧず鍚Э鎬佺殑鍏ㄨ嚜鐢卞害)
    init_log_var_data: float = 0.0
    init_log_var_phy: float = 0.0
    init_log_var_dyn: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model % self.nhead != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by nhead "
                f"({self.nhead})"
            )
        head_dim = self.d_model // self.nhead
        if head_dim % 2 != 0:
            raise ValueError(
                f"head_dim = d_model/nhead = {head_dim} must be even for RoPE. "
                f"Adjust d_model or nhead."
            )
        if self.n_dof not in (3, 6):
            raise ValueError(
                f"n_dof must be 3 (translational) or 6 (full), got {self.n_dof}"
            )


# =============================================================================
# 搂1  Rotary Position Embedding (RoPE)
# =============================================================================

    # ======= 銆愭ā鍧?锛氭棆杞綅缃紪鐮?RoPE銆?=======
    # 鎽掑純浜嗙粷瀵圭殑姝ｅ鸡娉綅缃紪鐮侊紝閫氳繃澶嶆暟鍩熸棆杞殑鏂瑰紡鍦ˋttention鐨凲鍜孠鐨勭偣绉腑鑷姩鎼哄甫涓よ€呯殑鐩稿璺濈
class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for causal Transformers.

    Injects *relative* position information into Q and K tensors by rotating
    pairs of dimensions using sinusoidal frequencies, without adding any
    learnable parameters.

    Mathematical Formulation (per head, per position ``t``):
        Given query/key vector q 鈭?鈩漗{d_h}, partition into consecutive
        pairs (q_{2i}, q_{2i+1}).  Define:

            胃_i = 1 / 10000^{2i / d_h}

        Apply 2-D rotation:
            鈹?q'_{2i}   鈹?  鈹?cos(t路胃_i)  鈭抯in(t路胃_i) 鈹?鈹?q_{2i}   鈹?
            鈹?           鈹?= 鈹?                           鈹?鈹?          鈹?
            鈹?q'_{2i+1} 鈹?  鈹?sin(t路胃_i)   cos(t路胃_i)  鈹?鈹?q_{2i+1} 鈹?

    The inner product 鉄╭'_t, k'_s鉄?then depends only on the *relative*
    offset (t 鈭?s), yielding translational invariance.

    Time complexity:  O(T 路 d_h)  per head (element-wise, no matmul).
    Space complexity: O(T 路 d_h/2) for cached sin/cos tables.

    Attributes:
        head_dim: Dimension per attention head.
        max_seq_len: Maximum sequence length for pre-computed tables.

    Note:
        This module adds **zero** learnable parameters to the model.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 2048) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        # Pre-compute inverse frequency vector: 胃_i = 1/10000^{2i/d_h}
        inv_freq = 1.0 / (  # 璁＄畻浣嶇疆缂栫爜鐨勬寚鏁伴鐜囪“鍑?
            10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Pre-compute sin/cos tables 鈥?lazily resized if seq_len exceeds cache.
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Build sin/cos lookup tables up to ``seq_len``.

        Args:
            seq_len: Maximum position index to cache.
        """
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        # Outer product: [T] x [d_h/2] 鈫?[T, d_h/2]
        freqs = torch.outer(t, self.inv_freq)
        # Duplicate along last dim to match head_dim: [T, d_h]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos_cached = emb.cos()
        sin_cached = emb.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)
        self.max_seq_len = seq_len

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        """Rotate consecutive pairs: (x0, x1, x2, x3, ...) 鈫?
        (鈭抶1, x0, 鈭抶3, x2, ...).

        Args:
            x: Tensor of shape [..., d_h] where d_h is even.

        Returns:
            Rotated tensor of same shape.
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply RoPE to query and key tensors.

        Args:
            q: Query tensor of shape [B, H, T, d_h].
            k: Key tensor of shape [B, H, T, d_h].

        Returns:
            Tuple of rotated (q', k') with same shapes.

        Raises:
            RuntimeError: If sequence length exceeds maximum cache size and
                cache rebuild fails (should not happen in practice).
        """
        seq_len = q.shape[2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].to(q.dtype)  # [T, d_h]
        sin = self.sin_cached[:seq_len].to(q.dtype)  # [T, d_h]

        # Broadcast: [1, 1, T, d_h] over [B, H, T, d_h]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        q_rot = q * cos + self._rotate_half(q) * sin  # 鏌ヨ鍚戦噺Q杩涜浜岀淮鏃嬭浆锛屾敞鍏ョ浉瀵逛綅缃壒寰?
        k_rot = k * cos + self._rotate_half(k) * sin

        return q_rot, k_rot


# =============================================================================
# 搂2  Multi-Head Attention with RoPE & Strict Masking
# =============================================================================

    # ======= 銆愭ā鍧?锛氬甫鏈塕oPE鍜屼弗鏍兼帺鐮佺殑澶氬ご鑷敞鎰忓姏銆?=======
class RoPEMultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with RoPE and strict composite masking.

    This module replaces ``nn.MultiheadAttention`` to integrate RoPE into
    the Q/K projection and to enforce a *composite* attention mask that
    is the element-wise conjunction of:

        final_mask = causal_mask 鈭?validity_mask

    where both are applied as additive 鈭掆垶 biases *before* softmax,
    guaranteeing that masked positions receive exactly zero weight in the
    attention output.

    Mathematical Formulation:
        A = softmax( (Q_rot 路 K_rot^T) / 鈭歞_h + M_final ) 路 V

        where M_final[i,j] = 0    if j 鈮?i AND frame j is valid,
                            = 鈭掆垶   otherwise.

    Time complexity:  O(T虏 路 d_h 路 H)  鈥?standard quadratic attention.
    Space complexity: O(T虏 路 H)  for attention weight matrix.

    Attributes:
        d_model: Total model dimension.
        nhead: Number of attention heads.
        head_dim: Dimension per head (d_model / nhead).
        rope: Rotary position embedding module (zero extra params).
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Fused QKV projection for efficiency (single matmul).
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)

        self.rope = RotaryPositionEmbedding(
            head_dim=self.head_dim, max_seq_len=max_seq_len
        )

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Self-attention forward pass with RoPE and composite mask.

        Args:
            x: Input tensor of shape [B, T, D].
            attn_mask: Pre-computed composite mask of shape [B路H, T, T] or
                [T, T].  Positions to mask must contain ``-inf``; unmasked
                positions must contain ``0.0``.  If ``None``, no masking is
                applied (not recommended for causal models).

        Returns:
            Output tensor of shape [B, T, D].
        """
        B, T, D = x.shape
        H = self.nhead
        d_h = self.head_dim

        # Fused QKV: [B, T, 3D] 鈫?3 脳 [B, T, D]
        qkv = self.qkv_proj(x)  # 涓€姝ョ煩闃典箻娉曠敓鎴怮,K,V锛屾彁楂樿绠楁晥鐜?
        qkv = qkv.reshape(B, T, 3, H, d_h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)  # each [B, H, T, d_h]

        # Apply RoPE to Q and K (zero extra params).
        q, k = self.rope(q, k)

        # Scaled dot-product attention: [B, H, T, T]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # 缂╂斁鐐圭Н娉ㄦ剰鍔涳紝璁＄畻鍚勬椂闂存涔嬪墠鐨勭浉鍏虫€?

        # Apply composite mask (causal 鈭?validity).
        allowed_mask: Optional[Tensor] = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                # [T, T] 鈫?broadcast over [B, H, T, T]
                mask = attn_mask.unsqueeze(0).unsqueeze(0)
                attn_weights = attn_weights + mask
                allowed_mask = mask > -5e8
            elif attn_mask.dim() == 3:
                # [B*H, T, T] 鈫?[B, H, T, T]
                mask = attn_mask.view(B, H, T, T)
                attn_weights = attn_weights + mask
                allowed_mask = mask > -5e8
            elif attn_mask.dim() == 4:
                allowed_mask = attn_mask > -5e8
                attn_weights = attn_weights + attn_mask  # 銆愰噸鐐广€戝皢鍥犳灉鎺╃爜鍜屾暟鎹湁鏁堟€ф帺鐮佸姞鍒版敞鎰忓姏鏉冮噸涓婏紙澶辨晥甯х殑寰楀垎涓?inf锛?

        attn_weights = F.softmax(attn_weights, dim=-1)
        if allowed_mask is not None:
            keep = allowed_mask.to(dtype=attn_weights.dtype)
            attn_weights = attn_weights * keep
            denom = attn_weights.sum(dim=-1, keepdim=True)
            attn_weights = torch.where(
                denom > 0,
                attn_weights / denom.clamp_min(1e-12),
                torch.zeros_like(attn_weights),
            )
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values.
        out = torch.matmul(attn_weights, v)  # [B, H, T, d_h]
        out = out.transpose(1, 2).reshape(B, T, D)
        out = self.out_proj(out)

        return out


# =============================================================================
# 搂3  Transformer Encoder Layer (Pre-Norm + RoPE)
# =============================================================================

    # ======= 銆愭ā鍧?锛歍ransformer缂栫爜鍣ㄥ眰 (Pre-Norm鏋舵瀯)銆?=======
class RoPETransformerEncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer with RoPE attention.

    Uses Pre-Norm (norm-first) architecture for more stable gradients in
    deep stacks, consistent with modern best practices (Xiong et al., 2020).

    Architecture:
        x 鈫?LN 鈫?RoPE-MHA 鈫?+ 鈫?LN 鈫?FFN 鈫?+
        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?   鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
              residual                 residual

    Time complexity:  O(T虏 路 D + T 路 D 路 D_ff)
    Space complexity: O(T虏 + T 路 D_ff)  (attention matrix + FFN activations)

    Attributes:
        d_model: Model dimension.
        nhead: Number of attention heads.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ) -> None:
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass.

        Args:
            x: [B, T, D].
            attn_mask: Composite mask [T, T] or [B*H, T, T].

        Returns:
            [B, T, D].
        """
        # Pre-Norm Self-Attention
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, attn_mask=attn_mask)  # 鍚湁灞忚斀閫昏緫鍜孯oPE鑷敞鎰忓姏灞傚鐞?
        x = self.dropout1(x) + residual

        # Pre-Norm FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual

        return x


# =============================================================================
# 搂4  Mask Construction Utilities
# =============================================================================

    # ======= 銆愭ā鍧?锛氭帺鐮佹瀯閫犲伐鍏枫€?=======
def build_causal_mask(seq_len: int, device: torch.device) -> Tensor:
    """Build a strict upper-triangular causal mask.

    Returns a [T, T] float tensor where position (i, j) is:
        0.0   if j 鈮?i  (attend)
        -1e9  if j > i  (block)

    Uses ``-1e9`` instead of ``-inf`` for numerical safety.  When combined
    with the validity mask (which may also be ``-1e9``), this avoids the
    ``softmax([-inf, ...]) 鈫?NaN`` pathology.

    Args:
        seq_len: Sequence length T.
        device: Target device.

    Returns:
        Causal mask tensor of shape [T, T].

    Time complexity:  O(T虏)
    Space complexity: O(T虏)
    """
    return torch.triu(  # 鐢熸垚涓婁笁瑙掔煩闃碉紝闃绘柇鏈潵淇℃伅浼犲鍒拌繃鍘伙紙鍗充弗鏍煎洜鏋滄帹鐞嗭級
        torch.full((seq_len, seq_len), -1e9, device=device),
        diagonal=1,
    )


def build_validity_mask(
    validity: Tensor,
    nhead: int,
) -> Tensor:
    """Convert per-frame validity indicators to an attention-compatible mask.

    For each pair (i, j), if frame j is invalid (validity[b, j] == 0), the
    attention weight from query i to key j must be zero.  We achieve this
    by setting the mask value to a large negative number at those positions.

    Safety Note:
        We use ``-1e9`` instead of ``-inf`` to prevent NaN when *all* keys
        in a row are invalid.  With ``-inf``, softmax([-inf, -inf, ...])
        produces 0/0 = NaN.  With ``-1e9``, softmax([-1e9, ...]) produces
        a near-uniform distribution with negligible magnitude, which is
        numerically safe and functionally equivalent.

    Args:
        validity: [B, T, 1] float tensor with values in {0, 1}.
        nhead: Number of attention heads (for broadcasting).

    Returns:
        Validity mask of shape [B, 1, 1, T] (broadcastable over [B, H, T, T]).
        Entries are 0.0 (valid) or -1e9 (invalid).

    Time complexity:  O(B 路 T)
    Space complexity: O(B 路 T)
    """
    # validity: [B, T, 1] 鈫?key_mask: [B, 1, 1, T]
    key_valid = validity.squeeze(-1)  # [B, T]
    # 0 鈫?-1e9 (large negative, safe for softmax), 1 鈫?0.0
    key_mask = (1.0 - key_valid) * (-1e9)  # [B, T]  # 濡傛灉杩欎竴甯ф槸鏃犳晥(濡備紶鎰熷櫒鏂仈)锛岃祴浜堟瀬灏忓€?1e9锛孲oftmax涔嬪悗娉ㄦ剰鍔涘己鍒跺綊闆?
    return key_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]


def build_composite_mask(
    seq_len: int,
    validity: Tensor,
    nhead: int,
    device: torch.device,
) -> Tensor:
    """Build the final composite attention mask: causal 鈭?validity.

    Combines the causal mask and data-validity mask into a single additive
    bias tensor.  The result guarantees that:
        1. Future frames are invisible (causality).
        2. Invalid/corrupt sensor frames contribute zero attention weight
           (information purity).

    Mathematical Guarantee:
        final_mask[b, h, i, j] = causal[i, j] + validity[b, j]

        After softmax:
            伪_{ij} = exp(score_{ij} + mask_{ij}) / 危_k exp(score_{ik} + mask_{ik})

        If mask_{ij} = -inf 鈫?伪_{ij} = 0  鈭?

    Args:
        seq_len: Sequence length T.
        validity: [B, T, 1] validity mask.
        nhead: Number of attention heads.
        device: Target device.

    Returns:
        Composite mask of shape [B, 1, T, T] (broadcastable over [B, H, T, T]).

    Time complexity:  O(B 路 T虏)   dominated by broadcasting.
    Space complexity: O(B 路 T虏)
    """
    causal = build_causal_mask(seq_len, device)          # [T, T]
    val_mask = build_validity_mask(validity, nhead)       # [B, 1, 1, T]

    # causal: [1, 1, T, T] + val_mask: [B, 1, 1, T] 鈫?[B, 1, T, T]
    composite = causal.unsqueeze(0).unsqueeze(0) + val_mask
    return composite


# =============================================================================
# 搂5  Transformer-based Trajectory Predictor (RoPE + Strict Masking)
# =============================================================================

    # ======= 銆愭ā鍧?锛氫富杞ㄨ抗棰勬祴鍣?(楠ㄥ共缃戠粶)銆?=======
class AUVTransformerPredictor(nn.Module):
    """Transformer Encoder trajectory predictor with RoPE and strict masking.

    [BREAKING v2.0] Major changes from v1.x:
        - Absolute sinusoidal PE removed; replaced by per-head RoPE.
        - Attention mask is a composite of causal + validity masks.
        - ``forward()`` now accepts ``validity`` and builds the composite mask
          internally, so the caller need not construct masks.
        - Output is position *offsets* (螖p) relative to last observed position.

    Architecture:
        Input [B,T,F] 鈫?Linear(F鈫扗) 鈫?[RoPE-TransformerEncoder 脳L] 鈫?
        last-step hidden 鈫?MLP Decoder 鈫?offsets [B, P, 3]

    The causal masking guarantees that position t attends only to s 鈮?t.
    The validity masking guarantees that corrupt/NaN frames are excluded
    from the softmax denominator.

    Attributes:
        d_model: Hidden dimension D.
        pred_len: Prediction horizon P.
        n_targets: Number of spatial coordinates (always 3 for x, y, z).
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pred_len: int = 10,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.pred_len = pred_len
        self.n_targets = 3

        # Input projection: F 鈫?D.
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # Encoder stack with RoPE.
        self.encoder_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Final LayerNorm (pre-norm architecture convention).
        self.final_norm = nn.LayerNorm(d_model)

        # v2.1: Validity-aware attention pooling.
        # A learnable query attends over the encoder output with validity
        # masking, producing a single [B, D] vector. This is strictly more
        # informative than `x[:, -1, :]` because (1) it weights frames by
        # learned relevance rather than recency, and (2) it respects the
        # per-frame validity mask so corrupt frames contribute zero weight.
        self.pool_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.pool_proj = nn.Linear(d_model, d_model, bias=False)
        # Learnable temporal position bias for pooling attention.
        # Without this, all frames of a smooth trajectory produce near-identical
        # hidden states, causing softmax to degenerate to uniform 1/T.
        # This bias lets the network learn "recent frames matter more" directly.
        self.pool_pos_bias = nn.Parameter(torch.zeros(max_seq_len))

        # MLP decoder: D 鈫?P*3.
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, pred_len * self.n_targets),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for stability.

        Biases are zero-initialised.  LayerNorm parameters are left at
        PyTorch defaults (weight=1, bias=0).

        v2.1 鈥?Decoder last-layer is initialised with very small weights
        (std=0.01) so that initial position offsets are near zero.  This
        prevents a common failure mode where early-training offsets are
        large, which causes M路v虈 in the Fossen residual to explode and
        destabilises the uncertainty weights.
        """
        for name, param in self.named_parameters():
            if "norm" in name:
                continue
            if "weight" in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Override: shrink the last decoder Linear so initial offsets 鈮?0.
        last_linear: Optional[nn.Linear] = None
        for module in self.decoder:
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is not None:
            nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        x_seq: Tensor,
        validity: Tensor,
    ) -> Tensor:
        """Forward pass with strict masking.

        Args:
            x_seq: Input features [B, T, F].
            validity: Per-frame validity mask [B, T, 1], values in {0, 1}.

        Returns:
            Position offsets [B, pred_len, 3].
        """
        B, T, _ = x_seq.shape

        # Project to model dimension.
        x = self.input_proj(x_seq)  # [B, T, D]

        # Build composite mask: causal 鈭?validity 鈫?[B, 1, T, T].
        composite_mask = build_composite_mask(
            seq_len=T,
            validity=validity,
            nhead=self.nhead,
            device=x.device,
        )

        # Transformer encoder stack.
        for layer in self.encoder_layers:
            x = layer(x, attn_mask=composite_mask)

        x = self.final_norm(x)

        # v2.1: Validity-aware attention pooling.
        # Replace last-frame pooling with a learned query that attends over
        # all frames, with validity-masked softmax. This respects corrupt
        # frames (zero weight) and captures long-range context rather than
        # relying on the recency of the final frame.
        #
        #   scores[b, t] = (pool_query 路 pool_proj(x[b, t])) / 鈭欴
        #   scores[b, t] += -1e9 路 (1 鈭?validity[b, t])
        #   伪 = softmax(scores)
        #   pooled = 危_t 伪[b, t] 路 x[b, t]
        q = self.pool_query / math.sqrt(self.d_model)  # [D]
        scores = (self.pool_proj(x) * q).sum(dim=-1)   # [B, T]
        # Add learnable temporal position bias to break uniform attention.
        scores = scores + self.pool_pos_bias[:T]
        key_valid = validity.squeeze(-1)               # [B, T]
        scores = scores + (1.0 - key_valid) * (-1e9)
        attn = F.softmax(scores, dim=-1)
        valid_float = key_valid.to(dtype=attn.dtype)
        attn = attn * valid_float
        denom = attn.sum(dim=-1, keepdim=True)
        attn = torch.where(
            denom > 0,
            attn / denom.clamp_min(1e-12),
            torch.zeros_like(attn),
        ).unsqueeze(-1)                                # [B, T, 1]
        pooled = (x * attn).sum(dim=1)                 # [B, D]

        # Decode to position offsets.
        flat_output = self.decoder(pooled)  # [B, P*3]
        offsets = flat_output.view(B, self.pred_len, self.n_targets)

        return offsets


# =============================================================================
# 搂6  Fossen Dynamics-Informed Loss
# =============================================================================

    # ======= 銆愭ā鍧?锛欶ossen 姘村姩鍔涘鐗╃悊娈嬪樊鎹熷け灞傘€?=======
    # 鏍稿績鐗╃悊绾︽潫灞傦細鎶夾UV杩愬姩瑙嗕负鑷敱闃诲凹杩愬姩锛屽弽鎺ㄥ叾涓殑闄勫姞璐ㄩ噺鍜岄樆灏煎弬鏁?
class FossenDynamicsLoss(nn.Module):
    """Dynamics-informed loss based on Fossen's marine vehicle equations.

    Enforces the 6-DOF (or 3-DOF translational) rigid-body + hydrodynamic
    equation of motion as a soft constraint:

        M 路 v虈 + C(v) 路 v + D(v) 路 v = 蟿

    Modelling Assumptions (declared for reproducibility):
        1. **Passive damping (蟿 鈮?0)**: The dataset does not contain thruster
           commands.  We therefore treat the system as freely decelerating
           under hydrodynamic forces and penalise the residual
           ``r = M路v虈 + C(v)路v + D(v)路v``.
        2. **Linear damping approximation**: D(v) is modelled as a constant
           diagonal matrix (linear drag), which is a standard first-order
           approximation valid at low-to-moderate Reynolds numbers.
        3. **Diagonal inertia**: M is fixed to physically plausible AUV mass
           and inertia values for identifiability; damping and control
           effectiveness are learned during training.
        4. **Quadratic Coriolis**: C(v) is implemented as a skew-symmetric
           function of v (simplified centripetal/Coriolis coupling).

    Learnable Parameters:
        - ``damping_diag``: Diagonal elements of the linear damping matrix.
        - ``quad_damping_diag``: Diagonal elements of the quadratic damping
          matrix.
        - ``raw_rudder_coeff`` / ``raw_stern_coeff``: Control-effectiveness
          coefficients for yaw and pitch moments.

    Time complexity per sample:  O(P 路 n虏) where P = pred_len, n = n_dof.
    Space complexity:            O(n虏) for coefficient matrices.

    Attributes:
        n_dof: Number of degrees of freedom (3 or 6).
        fixed_mass_diag: Fixed diagonal mass/inertia [n_dof].
        damping_diag: Learnable diagonal damping [n_dof].
        raw_rudder_coeff/raw_stern_coeff: Learnable control effectiveness.
    """

    def __init__(
        self,
        n_dof: int = 3,
        init_mass: float = 50.0,
        init_damping: float = 10.0,
        reduction: str = "mean",
        vel_scale: float = 1.0,
        dt: float = 0.05,
        residual_beta: float = 1.0,
    ) -> None:
        """Initialise Fossen dynamics loss with non-dimensionalization.

        Args:
            n_dof: Degrees of freedom (3 for translational, 6 for full).
            init_mass: Initial diagonal mass value (kg).
            init_damping: Initial diagonal damping (N路s/m).
            reduction: 'mean', 'sum', or 'none'.
            vel_scale: Characteristic velocity std (m/s) from training data.
                Used to non-dimensionalize the dynamics residual so that
                L_dyn is O(1) and comparable to L_data / L_kin.
            dt: Sampling interval (s), used with vel_scale to define the
                characteristic force scale F_char = M_init * vel_scale / dt.

        Raises:
            ValueError: If reduction mode is invalid.
        """
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction
        self.n_dof = n_dof
        self.residual_beta = residual_beta
        self.last_residual_loss: Optional[Tensor] = None
        self.last_prior_loss: Optional[Tensor] = None

        # Characteristic force for non-dimensionalization.
        # F_char = M_init 脳 V_char / dt makes the force residual O(1).
        f_char = max(init_mass * vel_scale / dt, 1e-6)
        self.register_buffer(
            "force_scale", torch.tensor(f_char, dtype=torch.float32)
        )

        # Learnable physics parameters (log-space for positivity).
        #
        # Structural constraint for elongated (torpedo-shaped) bodies:
        #   M_22 = M_33 >= M_11  (lateral added mass > axial added mass)
        # Parameterization:
        #   M_11 = exp(log_m_axial)
        #   M_22 = M_33 = M_11 * (1 + softplus(log_lateral_ratio))
        # For 6-DOF, rotational inertias are independent learnable params:
        #   I_roll  = exp(log_I_roll)    ~0.15 kg路m虏 for REMUS 100
        #   I_pitch = I_yaw = exp(log_I_lateral)  ~4.2 kg路m虏 for REMUS 100
        # Frozen mass properties
        m_axial = max(init_mass, 25.0)
        m_lateral = m_axial * 1.5
        m_diag_list = [m_axial, m_lateral, m_lateral]
        if n_dof == 6:
            m_diag_list.append(0.15)
            m_diag_list.append(5.0)
            m_diag_list.append(5.0)
        
        self.register_buffer('fixed_mass_diag', torch.tensor(m_diag_list))
        self.log_damping_diag = nn.Parameter(
            torch.full((n_dof,), math.log(init_damping))
        )
        
        # Quadratic damping parameters D_q * |v| * v
        # Initialized to small positive values (0.1) so as not to overwhelm linear D early on
        self.log_D_quad_diag = nn.Parameter(
            torch.full((n_dof,), math.log(0.1))
        )

        # Learnable rudder/stern control effectiveness coefficients (N路m/rad)
        self.raw_rudder_coeff = nn.Parameter(torch.tensor(0.0))
        self.raw_stern_coeff = nn.Parameter(torch.tensor(0.0))

    @property
    def mass_matrix(self) -> Tensor:
        return torch.diag(self.fixed_mass_diag)

    @property
    def log_mass_diag(self) -> Tensor:
        """Reconstruct log_mass_diag for diagnostic logging."""
        with torch.no_grad():
            return torch.log(torch.diag(self.mass_matrix).clamp(min=1e-6))

    @property
    def damping_matrix(self) -> Tensor:
        """Construct diagonal damping matrix D = diag(exp(clamp(log_d))).

        log_damping_min = 0.0 鈫?D_min = 1.0 N路s/m per DOF.

        Returns:
            D: [n_dof, n_dof] positive-definite diagonal tensor.
        """
        D_raw = self.log_damping_diag
        D_val = 1.0 + F.softplus(D_raw)
        return torch.diag(D_val)

    @property
    def quad_damping_matrix(self) -> Tensor:
        """Construct diagonal quadratic damping matrix D_q.

        Returns:
            D_q: [n_dof, n_dof] positive-definite diagonal tensor.
        """
        D_q_raw = self.log_D_quad_diag
        D_q_val = 1e-4 + F.softplus(D_q_raw)
        return torch.diag(D_q_val)

    def _build_coriolis_matrix(self, v: Tensor) -> Tensor:
        """Build skew-symmetric Coriolis matrix C(v) from velocity.

        For 3-DOF (surge u, sway v_s, heave w):
            C(v) = c 路 鈹? 0   -w   v_s 鈹?
                       鈹? w    0  -u   鈹?
                       鈹?-v_s  u    0  鈹?

        where c is a learnable scalar coupling strength.  For 6-DOF the
        construction generalises to the full Coriolis/centripetal matrix.

        Args:
            v: Velocity tensor [B, n_dof].

        Returns:
            C(v): [B, n_dof, n_dof] skew-symmetric tensor.

        Time complexity:  O(B 路 n虏)
        Space complexity: O(B 路 n虏)
        """
        B = v.shape[0]
        n = v.shape[1]
        device = v.device

        C = torch.zeros(B, n, n, device=device, dtype=v.dtype)
        M = self.mass_matrix[:n, :n].to(device=device, dtype=v.dtype)

        if n == 3:
            # Physically rigorous Coriolis matrix derived from Mass matrix
            # C_12 = -M_22 v + M_33 w (simplified, w is heave)
            # Fossen p.120: Rigid body coriolis + added mass coriolis
            m11, m22, m33 = M[0,0], M[1,1], M[2,2]
            u, vs, w = v[:, 0], v[:, 1], v[:, 2]

            C[:, 0, 1] = -m22 * w
            C[:, 0, 2] = m33 * vs
            C[:, 1, 0] = m11 * w
            C[:, 1, 2] = -m33 * u
            C[:, 2, 0] = -m11 * vs
            C[:, 2, 1] = m22 * u

        elif n == 6:
            # Fossen p.120: C(v) = C_RB(v) + C_A(v)
            m11, m22, m33 = M[0,0], M[1,1], M[2,2]
            m44, m55, m66 = M[3,3], M[4,4], M[5,5]
            u, vs, w = v[:, 0], v[:, 1], v[:, 2]
            p, q, r = v[:, 3], v[:, 4], v[:, 5]

            # Translational - Translational
            # C_11 = 0

            # Translational - Rotational
            C[:, 0, 4] = m33 * w
            C[:, 0, 5] = -m22 * vs
            C[:, 1, 3] = -m33 * w
            C[:, 1, 5] = m11 * u
            C[:, 2, 3] = m22 * vs
            C[:, 2, 4] = -m11 * u

            # Rotational - Translational
            C[:, 3, 1] = -C[:, 1, 3]
            C[:, 3, 2] = -C[:, 2, 3]
            C[:, 4, 0] = -C[:, 0, 4]
            C[:, 4, 2] = -C[:, 2, 4]
            C[:, 5, 0] = -C[:, 0, 5]
            C[:, 5, 1] = -C[:, 1, 5]

            # Rotational - Rotational
            C[:, 3, 4] = -m66 * r
            C[:, 3, 5] = m55 * q
            C[:, 4, 3] = m66 * r
            C[:, 4, 5] = -m44 * p
            C[:, 5, 3] = -m55 * q
            C[:, 5, 4] = m44 * p

        return C

    @staticmethod
    def _ned_to_body_velocity(vel_ned: Tensor, attitude: Tensor) -> Tensor:
        """Rotate NED velocity into the body frame using roll/pitch/yaw.

        The data generator uses v_ned = R_body_to_ned(roll, pitch, yaw) v_body,
        so the inverse transform is v_body = R^T v_ned.
        """
        roll = attitude[..., 0]
        pitch = attitude[..., 1]
        yaw = attitude[..., 2]

        cphi, sphi = torch.cos(roll), torch.sin(roll)
        cth, sth = torch.cos(pitch), torch.sin(pitch)
        cpsi, spsi = torch.cos(yaw), torch.sin(yaw)

        r00 = cpsi * cth
        r01 = cpsi * sth * sphi - spsi * cphi
        r02 = cpsi * sth * cphi + spsi * sphi
        r10 = spsi * cth
        r11 = spsi * sth * sphi + cpsi * cphi
        r12 = spsi * sth * cphi - cpsi * sphi
        r20 = -sth
        r21 = cth * sphi
        r22 = cth * cphi

        vn = vel_ned[..., 0]
        ve = vel_ned[..., 1]
        vd = vel_ned[..., 2]

        u = r00 * vn + r10 * ve + r20 * vd
        v = r01 * vn + r11 * ve + r21 * vd
        w = r02 * vn + r12 * ve + r22 * vd
        return torch.stack((u, v, w), dim=-1)

    def forward(
        self,
        pred_pos: Tensor,
        last_vel: Tensor,
        dt: float,
        thrust_data: Optional[Tensor] = None,
        target_vel: Optional[Tensor] = None,
        target_attitude: Optional[Tensor] = None,
        last_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute Fossen dynamics residual loss from predicted translation.

        The full Fossen rigid-body + hydrodynamic equation is:
            M 路 v虈 + C(v) 路 v + D(v) 路 v = 蟿

        The residual r = M路v虈 + C(v)路v + D路v 鈭?蟿 is penalised.

        The current implementation derives translational velocity from
        predicted NED positions so the residual regularises the trajectory
        predictor. If target_attitude is supplied, that velocity is rotated
        into the body frame before evaluating Fossen dynamics. In 6-DOF mode,
        angular rates come from target_vel because this model does not
        forecast future attitude.

        This loss still contains finite differences of predicted position.
        To reduce short-window gradient spikes, the non-dimensionalized
        residual is clipped and penalized with SmoothL1/Huber loss instead of
        a pure squared residual.

        Args:
            pred_pos: Predicted positions [B, P, 3], used to derive
                translational velocity and acceleration.
            last_vel: Velocity at sequence boundary [B, n_dof].
            dt: Sampling interval (seconds).
            thrust_data: Optional thrust/control containing
                (thrust_net_N, rudder_rad, stern_rad). Shape may be [B, 3]
                for boundary zero-order hold or [B, P, 3] for step-wise
                forcing. If None, tau = 0.
            target_vel: Ground-truth velocity over prediction horizon
                [B, P, n_dof].  In 6-DOF mode, angular rates are taken from
                this tensor because the predictor only outputs positions.
            target_attitude: Ground-truth attitude [B, P, 3] used to rotate
                predicted NED translational velocity into the body frame.
            last_pos: Last observed position [B, 3], used to estimate the
                first predicted translational velocity from pred_pos.

        Returns:
            Dynamics residual loss (scalar if reduction != 'none').
        """
        B, P, D_pos = pred_pos.shape
        device = pred_pos.device
        dtype = pred_pos.dtype

        if P < 2:
            return torch.zeros(1, device=device, requires_grad=True)

        # Current training supplies target_body_vel, but the dynamics loss must
        # still depend on pred_pos to regularise the trajectory predictor.
        # Derive translational velocity from predicted NED positions, rotate it
        # into the body frame with GT attitude, then fill angular rates from
        # target_vel in 6-DOF mode because the model does not predict attitude.
        if self.n_dof >= 6 and target_vel is not None and target_vel.shape[2] >= 6:
            n = 6
        else:
            n = min(3, self.n_dof, D_pos)

        pos = pred_pos[:, :, :3]
        vel_trans = torch.zeros(B, P, 3, device=device, dtype=dtype)
        if last_pos is not None:
            vel_trans[:, 0, :] = (pos[:, 0, :] - last_pos[:, :3]) / (dt + 1e-8)
        else:
            vel_trans[:, 0, :] = last_vel[:, :3]
        vel_trans[:, 1:, :] = (pos[:, 1:, :] - pos[:, :-1, :]) / (dt + 1e-8)

        if target_attitude is not None:
            vel_trans = self._ned_to_body_velocity(
                vel_trans,
                target_attitude[:, :, :3].to(device=device, dtype=dtype),
            )

        vel = torch.zeros(B, P, n, device=device, dtype=dtype)
        vel[:, :, :3] = vel_trans[:, :, :min(3, n)]
        if n > 3 and target_vel is not None:
            vel[:, :, 3:n] = target_vel[:, :, 3:n].to(device=device, dtype=dtype)

        # --- Acceleration from finite differences of predicted velocity ---
        # vel_trans is derived from pred_pos, so this is still a second-order
        # position finite difference. The SmoothL1 residual below is the
        # gradient safety net for short horizons and small dt.
        accel = torch.zeros(B, P, n, device=device, dtype=dtype)
        if P >= 2:
            # Forward difference for first timestep.
            accel[:, 0, :] = (vel[:, 1, :] - vel[:, 0, :]) / (dt + 1e-8)
            if P >= 3:
                # Central difference for interior points (2nd-order accurate).
                accel[:, 1:-1, :] = (
                    vel[:, 2:, :] - vel[:, :-2, :]
                ) / (2.0 * dt + 1e-8)
            # Backward difference for last timestep.
            accel[:, -1, :] = (vel[:, -1, :] - vel[:, -2, :]) / (dt + 1e-8)

        # --- Force-residual form: r = M路v虈 + C(v)路v + D路v 鈭?蟿 ---
        # The force form's trivial solution is M鈫? (blocked by lower clamp
        # M 鈮?1 kg). Unlike the acceleration form (M鈦宦?, there is no
        # M鈫掆垶 pathology. The residual is non-dimensionalized by force_scale
        # so that L_dyn is O(L_data).
        M = self.mass_matrix[:n, :n].to(device=device, dtype=dtype)          # [n, n]
        D_mat = self.damping_matrix[:n, :n].to(device=device, dtype=dtype)   # [n, n]
        D_quad_mat = self.quad_damping_matrix[:n, :n].to(device=device, dtype=dtype) # [n, n]

        M_accel = torch.einsum("bpn,nn->bpn", accel, M)
        D_vel = torch.einsum("bpn,nn->bpn", vel, D_mat)
        
        # Quadratic damping D_q * |v| * v
        quad_damping = torch.einsum("bpn,nn->bpn", vel * torch.abs(vel), D_quad_mat)
        D_vel = D_vel + quad_damping

        vel_flat = vel.reshape(B * P, n)
        C_v = self._build_coriolis_matrix(vel_flat)
        C_vel = torch.bmm(
            C_v, vel_flat.unsqueeze(-1)
        ).squeeze(-1).reshape(B, P, n)

        # --- Construct generalised force vector 蟿 ---
        # thrust_data: [B, 3] or [B, P, 3] =
        #   (thrust_net_N, rudder_rad, stern_rad)
        # For 6-DOF: 蟿 = [X_thrust, 0, 0, 0, M_stern, N_rudder]
        #   X_thrust = thrust_net (surge force)
        #   M_stern  ~ stern_rad (pitch moment, simplified)
        #   N_rudder ~ rudder_rad (yaw moment, simplified)
        rudder_coeff = 15.0 + F.softplus(self.raw_rudder_coeff)
        stern_coeff = 15.0 + F.softplus(self.raw_stern_coeff)

        tau = torch.zeros(B, P, n, device=device, dtype=dtype)
        if thrust_data is not None:
            thrust = thrust_data.to(device=device, dtype=dtype)
            if thrust.dim() == 2:
                thrust = thrust.unsqueeze(1).expand(-1, P, -1)
            elif thrust.dim() == 3:
                if thrust.shape[1] != P:
                    raise ValueError(
                        f"thrust_data horizon mismatch: expected P={P}, "
                        f"got {thrust.shape[1]}"
                    )
            else:
                raise ValueError(
                    "thrust_data must have shape [B, 3] or [B, P, 3]"
                )

            tau[:, :, 0] = thrust[:, :, 0]  # surge thrust [N]
            if n >= 5 and thrust.shape[2] >= 3:
                tau[:, :, 4] = thrust[:, :, 2] * stern_coeff  # stern -> pitch moment [N*m]
            if n >= 6 and thrust.shape[2] >= 2:
                tau[:, :, 5] = thrust[:, :, 1] * rudder_coeff  # rudder -> yaw moment [N*m]

        residual = M_accel + C_vel + D_vel - tau  # [B, P, n]

        # Non-dimensionalize by characteristic force F_char = M_init * V_char / dt
        residual = residual / self.force_scale

        residual = torch.clamp(residual, -1e3, 1e3)

        # Physical Prior Regularization for Damping Matrix (prevent direction reversal)
        # Encourage D_sway / D_surge ~ 5.0 (typical for slender bodies)
        d_surge = D_mat[0, 0] + 1e-6
        d_sway = D_mat[1, 1]
        prior_loss = 0.5 * ((d_sway / d_surge) - 5.0) ** 2
        
        # Safety net for D_heave to prevent parameter collapse since w 鈮?0
        if n >= 3:
            d_heave = D_mat[2, 2]
            # Requires D_heave >= 0.5 * d_sway
            prior_loss = prior_loss + 0.5 * F.relu(0.5 * d_sway - d_heave) ** 2

        if self.reduction == "mean":
            dyn_loss = F.smooth_l1_loss(
                residual,
                torch.zeros_like(residual),
                beta=self.residual_beta,
                reduction="mean",
            )
            self.last_residual_loss = dyn_loss.detach()
            self.last_prior_loss = prior_loss.detach()
            return dyn_loss + prior_loss
        elif self.reduction == "sum":
            dyn_loss = F.smooth_l1_loss(
                residual,
                torch.zeros_like(residual),
                beta=self.residual_beta,
                reduction="sum",
            )
            self.last_residual_loss = dyn_loss.detach()
            self.last_prior_loss = prior_loss.detach()
            return dyn_loss + prior_loss
        else:
            dyn_loss = F.smooth_l1_loss(
                residual,
                torch.zeros_like(residual),
                beta=self.residual_beta,
                reduction="none",
            )
            self.last_residual_loss = dyn_loss.detach().mean()
            self.last_prior_loss = prior_loss.detach()
            return dyn_loss + prior_loss


# =============================================================================
# 搂7  Trapezoidal Kinematic Loss (Retained, Complementary to Dynamics)
# =============================================================================

    # ======= 銆愭ā鍧?锛氭褰㈣繍鍔ㄥ绾︽潫灞傘€?=======
class KinematicPhysicsLoss(nn.Module):
    """Position鈥搗elocity consistency constraint using ground-truth velocity.

    Enforces that predicted position changes are consistent with the
    *observed* (ground-truth) velocity at the sequence boundary via the
    trapezoidal integration rule:

        p虃_{k+1} 鈭?p虃_k 鈮?螖t 路 v_gt

    For the first prediction step, the GT velocity at the boundary is used.
    This provides a genuine learning signal because the velocity comes from
    an *independent* data source (not derived from the predictions themselves).

    Mathematical Formulation:
        R_0 = p虃_0 鈭?p_last 鈭?螖t 路 v_last
        R_k = p虃_{k+1} 鈭?p虃_k 鈭?螖t 路 v_last   (constant velocity assumption)

    The constant-velocity assumption over the short prediction horizon
    (P 脳 螖t = 5 脳 0.05 = 0.25s) is physically reasonable for AUVs at
    moderate speed where acceleration is small relative to velocity.

    Time complexity:  O(B 路 P 路 3)
    Space complexity: O(B 路 P 路 3)

    Attributes:
        reduction: Loss reduction mode ('mean', 'sum', 'none').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction: {reduction}")
        self.reduction = reduction

    def forward(
        self,
        pred_pos: Tensor,
        last_vel: Tensor,
        dt: float,
        target_vel: Optional[Tensor] = None,
        last_pos: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute position-velocity consistency residual.

        v2.1 鈥?When `target_vel` (GT velocity over the prediction horizon)
        is supplied, the residual at step k uses the *per-step* GT velocity
        (trapezoidal integration is tight):

            R_k = (p虃_{k+1} 鈭?p虃_k) 鈭?螖t 路 陆路(v_gt[k] + v_gt[k+1])

        This correctly handles curved trajectories (S-turns, spirals) where
        the constant-velocity assumption fails. When `target_vel` is None,
        falls back to the legacy `螖t 路 v_last` form.

        Args:
            pred_pos: Predicted positions [B, P, 3].
            last_vel: GT velocity at sequence boundary [B, 3].
            dt: Time step interval.
            target_vel: GT velocity across the prediction horizon
                [B, P, 3]. Preferred; enables trajectory-aware residual.
            last_pos: Last observed position [B, 3].  When supplied, the
                first predicted step is constrained against the sequence
                boundary as well as internal prediction increments.

        Returns:
            Kinematic residual loss.
        """
        if pred_pos.shape[1] < 1:
            return torch.zeros(1, device=pred_pos.device, requires_grad=True)

        P = pred_pos.shape[1]

        residual_terms: List[Tensor] = []

        if last_pos is not None:
            first_step = pred_pos[:, 0, :] - last_pos[:, :3]
            if target_vel is not None:
                first_expected = dt * 0.5 * (last_vel[:, :3] + target_vel[:, 0, :3])
            else:
                first_expected = dt * last_vel[:, :3]
            residual_terms.append((first_step - first_expected).unsqueeze(1))

        if P < 2:
            if not residual_terms:
                return torch.zeros(1, device=pred_pos.device, requires_grad=True)
            residual = torch.cat(residual_terms, dim=1)
            if self.reduction == "mean":
                return (residual ** 2).mean()
            elif self.reduction == "sum":
                return (residual ** 2).sum()
            else:
                return residual ** 2

        # Actual position increments from predictions: [B, P-1, 3]
        pos_diff = pred_pos[:, 1:, :] - pred_pos[:, :-1, :]

        if target_vel is not None:
            # Trapezoidal rule per step: dt 路 陆路(v_k + v_{k+1})
            v_avg = 0.5 * (target_vel[:, :-1, :] + target_vel[:, 1:, :])  # [B, P-1, 3]
            expected_step = dt * v_avg  # [B, P-1, 3]
        else:
            # Legacy fallback: constant-velocity assumption.
            expected_step = (last_vel * dt).unsqueeze(1)  # [B, 1, 3]

        residual_terms.append(pos_diff - expected_step)  # [B, P-1, 3]
        residual = torch.cat(residual_terms, dim=1)

        if self.reduction == "mean":
            return (residual ** 2).mean() if residual.numel() > 0 else \
                torch.zeros(1, device=pred_pos.device, requires_grad=True)
        elif self.reduction == "sum":
            return (residual ** 2).sum()
        else:
            return residual ** 2


# =============================================================================
# 搂8  Adaptive Multi-Task Loss (Unified Information Bottleneck)
# =============================================================================

    # ======= 銆愭ā鍧?锛氳嚜閫傚簲澶氫换鍔￠瞾妫掓崯澶卞眰銆?=======
    # 涓嶉€氳繃浜哄伐鎷嶈剳琚嬪畾鏉冮噸锛屽埄鐢ㄦ瀬澶т技鐒朵腑鐨勫悓鏂瑰樊涓嶇‘瀹氭€?Homoscedasticity)鏉ヨ嚜閫傚簲璋冩潈
class AdaptiveRobustLoss(nn.Module):
    """Unified multi-task loss with homoscedastic uncertainty weighting.

    [BREAKING v2.0] All confidence-gate logic has been removed.  The loss
    function is now a *pure* maximum-likelihood formulation with no
    forward-pass modulation, eliminating the gradient conflict between
    ConfidenceGate and the homoscedastic 蟽 parameters.

    Mathematical Formulation (strict MLE derivation):
        Given data loss L_data and physics losses L_phy, L_dyn:

        L = 1/(2路蟽虏_data) 路 L_data
          + 1/(2路蟽虏_phy)  路 L_phy
          + 1/(2路蟽虏_dyn)  路 L_dyn
          + log(蟽_data) + log(蟽_phy) + log(蟽_dyn)

    Parameterisation:
        We optimise s_k = log(蟽虏_k), so 蟽_k = exp(s_k / 2) > 0 always.
        The precision is 1/(2蟽虏_k) = 0.5 路 exp(-s_k).
        The log-barrier is 0.5 路 s_k (equivalent to log(蟽_k)).

    This parameterisation guarantees:
        - 蟽虏_k > 0 without constrained optimisation.
        - Numerically stable gradients (no division by small 蟽虏).
        - Automatic task weighting: high-noise tasks get down-weighted.

    Attributes:
        log_var_data: Learnable s_data = log(蟽虏_data).
        log_var_phy:  Learnable s_phy  = log(蟽虏_phy).
        log_var_dyn:  Learnable s_dyn  = log(蟽虏_dyn).
    """

    def __init__(
        self,
        init_log_var_data: float = 0.0,
        init_log_var_phy: float = 0.0,
        init_log_var_dyn: float = 0.0,
    ) -> None:
        super().__init__()
        self.log_var_data = nn.Parameter(torch.tensor(init_log_var_data))
        self.log_var_phy = nn.Parameter(torch.tensor(init_log_var_phy))
        self.log_var_dyn = nn.Parameter(torch.tensor(init_log_var_dyn))

    # --- Diagnostic properties (for logging, not in computational graph) ---

    @property
    def sigma_data(self) -> float:
        """Current 蟽_data = exp(s_data / 2)."""
        return math.exp(0.5 * self.log_var_data.item())

    @property
    def sigma_phy(self) -> float:
        """Current 蟽_phy = exp(s_phy / 2)."""
        return math.exp(0.5 * self.log_var_phy.item())

    @property
    def sigma_dyn(self) -> float:
        """Current 蟽_dyn = exp(s_dyn / 2)."""
        return math.exp(0.5 * self.log_var_dyn.item())

    @property
    def weight_data(self) -> float:
        """Effective data weight: 1/(2蟽虏_data) = 0.5路exp(鈭抯_data)."""
        return 0.5 * math.exp(-self.log_var_data.item())

    @property
    def weight_phy(self) -> float:
        """Effective kinematic weight: 0.5路exp(鈭抯_phy)."""
        return 0.5 * math.exp(-self.log_var_phy.item())

    @property
    def weight_dyn(self) -> float:
        """Effective dynamics weight: 0.5路exp(鈭抯_dyn)."""
        return 0.5 * math.exp(-self.log_var_dyn.item())

    def forward(
        self,
        pred_pos: Tensor,
        target_pos: Tensor,
        loss_physics: Tensor,
        loss_dynamics: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute unified multi-task loss.

        Args:
            pred_pos: Predicted positions [B, P, 3].
            target_pos: Ground truth positions [B, P, 3].
            loss_physics: Pre-computed trapezoidal kinematic loss (scalar).
            loss_dynamics: Pre-computed Fossen dynamics loss (scalar).

        Returns:
            Tuple of:
                - loss_total: Scalar tensor for backpropagation.
                - log_dict: Dictionary of all loss components & diagnostics.

        Time complexity:  O(B 路 P 路 3) for MSE; O(1) for weighting.
        Space complexity: O(1) for learnable 蟽 parameters.
        """
        # --- Data loss (MSE with NaN-safe masking) ---
        valid_mask = ~torch.isnan(target_pos)
        if valid_mask.any():
            loss_data_raw = F.mse_loss(
                pred_pos[valid_mask],
                target_pos[valid_mask],
                reduction="mean",
            )
        else:
            loss_data_raw = torch.zeros(
                1, device=pred_pos.device, dtype=pred_pos.dtype
            )

        # --- Precision (inverse variance) computation ---
        # Clamp log_var from below to prevent 蟽 collapse.
        # log_var_min = -6 鈫?蟽_min = exp(-3) 鈮?0.05, preventing infinite precision.
        LOG_VAR_MIN = -6.0
        log_var_data_c = torch.clamp(self.log_var_data, min=LOG_VAR_MIN)
        log_var_phy_c = torch.clamp(self.log_var_phy, min=LOG_VAR_MIN)
        log_var_dyn_c = torch.clamp(self.log_var_dyn, min=LOG_VAR_MIN)

        precision_data = torch.exp(-log_var_data_c)
        precision_phy = torch.exp(-log_var_phy_c)
        precision_dyn = torch.exp(-log_var_dyn_c)

        # --- Total loss with log-barrier regularisation ---
        # L = 危_k [ 0.5 路 exp(-s_k) 路 L_k + 0.5 路 s_k ]
        loss_total = (
            0.5 * precision_data * loss_data_raw
            + 0.5 * precision_phy * loss_physics
            + 0.5 * precision_dyn * loss_dynamics
            + 0.5 * (log_var_data_c + log_var_phy_c + log_var_dyn_c)
        )

        # --- Diagnostics (detached, no graph) ---
        log_dict: Dict[str, float] = {
            "loss_total": loss_total.detach().item(),
            "loss_data_raw": loss_data_raw.detach().item(),
            "loss_physics_raw": loss_physics.detach().item(),
            "loss_dynamics_raw": loss_dynamics.detach().item(),
            "sigma_data": self.sigma_data,
            "sigma_phy": self.sigma_phy,
            "sigma_dyn": self.sigma_dyn,
            "weight_data": self.weight_data,
            "weight_phy": self.weight_phy,
            "weight_dyn": self.weight_dyn,
            "log_var_data": self.log_var_data.item(),
            "log_var_phy": self.log_var_phy.item(),
            "log_var_dyn": self.log_var_dyn.item(),
        }

        return loss_total, log_dict


# =============================================================================
# 搂9  Main Model: RobustPINN v2.0
# =============================================================================

    # ======= 銆愭ā鍧?锛氱粍瑁呰捣鏉ョ殑涓绘ā鍨?RobustPINN銆?=======
class RobustPINN(nn.Module):
    """Dynamics-Informed Neural Network for AUV trajectory prediction.

    [BREAKING v2.0] Changes from v1.x:
        - ConfidenceGate removed.  ``forward()`` returns only ``pred_pos``.
        - Validity mask is now consumed by the Transformer attention layer
          directly (strict masking), not by a separate gate module.
        - The model is purely a *predictor*; all loss/uncertainty logic
          is handled by external loss modules.

    Architecture Overview:
        鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
        鈹? Input: x_seq [B, T, F] + validity_mask [B, T, 1]              鈹?
        鈹?                                                                鈹?
        鈹? 鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?  鈹?
        鈹? 鈹?  AUVTransformerPredictor                                鈹?  鈹?
        鈹? 鈹?  鈼?RoPE (no absolute PE)                                鈹?  鈹?
        鈹? 鈹?  鈼?Composite mask = causal 鈭?validity                   鈹?  鈹?
        鈹? 鈹?  鈼?Output: offsets [B, P, 3]                            鈹?  鈹?
        鈹? 鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?  鈹?
        鈹?                        鈹?                                      鈹?
        鈹?   pred_pos = last_pos + offsets                                鈹?
        鈹?                        鈹?                                      鈹?
        鈹? Output: pred_pos [B, P, 3]                                    鈹?
        鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

    Loss computation is external (see ``create_loss_modules``):
        L = 1/(2蟽虏_data)路MSE + 1/(2蟽虏_phy)路L_trap + 1/(2蟽虏_dyn)路L_fossen
          + log(蟽_data路蟽_phy路蟽_dyn)

    Attributes:
        config: Frozen model configuration.
        predictor: Transformer-based trajectory predictor.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.predictor = AUVTransformerPredictor(
            n_features=config.n_features,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_encoder_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            pred_len=config.pred_len,
            max_seq_len=config.seq_len + 64,
        )

    def forward(
        self,
        x_seq: Tensor,
        validity: Tensor,
        last_pos: Tensor,
    ) -> Tensor:
        """Forward pass: predict future trajectory positions.

        [BREAKING v2.0] Returns only ``pred_pos`` (no confidence).

        Args:
            x_seq: Standardised input sequence [B, T, F].
            validity: Per-frame validity mask [B, T, 1], values in {0, 1}.
            last_pos: Last observed position [B, 3] (for offset anchoring).

        Returns:
            pred_pos: Predicted absolute positions [B, pred_len, 3].

        Raises:
            RuntimeError: If input dimensions are inconsistent with config.
        """
        if x_seq.shape[-1] != self.config.n_features:
            raise RuntimeError(
                f"Expected {self.config.n_features} input features, "
                f"got {x_seq.shape[-1]}"
            )

        offsets = self.predictor(x_seq, validity)
        pred_pos = last_pos.unsqueeze(1) + offsets  # 鎶婄粷瀵瑰潗鏍囬娴嬭浆鍖栦负鐩稿鍋忕Щ鍙犲姞锛屽ぇ澶х紦瑙ransformer鐨勫钩绉讳笉鍧囬棶棰?

        return pred_pos

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 搂10  Factory Functions
# =============================================================================

def create_model(config: ModelConfig) -> RobustPINN:
    """Factory: instantiate RobustPINN from configuration.

    Args:
        config: Immutable model configuration.

    Returns:
        Initialised RobustPINN model (not yet moved to device).
    """
    return RobustPINN(config)


def create_loss_modules(
    config: ModelConfig,
    vel_scale: float = 1.0,
    dt: float = 0.05,
) -> Tuple[KinematicPhysicsLoss, FossenDynamicsLoss, AdaptiveRobustLoss]:
    """Factory: instantiate all loss modules from configuration.

    Returns three modules that should all be registered with the optimiser
    (the adaptive loss and dynamics loss have learnable parameters).

    Args:
        config: Model configuration.
        vel_scale: Velocity std from training data for dynamics normalization.
        dt: Sampling interval for dynamics normalization.

    Returns:
        Tuple of:
            - kinematic_loss: Trapezoidal integration residual.
            - dynamics_loss: Fossen dynamics residual with learnable damping
              and control-effectiveness parameters.
            - adaptive_loss: Homoscedastic uncertainty weighting.
    """
    kinematic_loss = KinematicPhysicsLoss(reduction="mean")
    dynamics_loss = FossenDynamicsLoss(
        n_dof=config.n_dof,
        reduction="mean",
        vel_scale=vel_scale,
        dt=dt,
    )
    adaptive_loss = AdaptiveRobustLoss(
        init_log_var_data=config.init_log_var_data,
        init_log_var_phy=config.init_log_var_phy,
        init_log_var_dyn=config.init_log_var_dyn,
    )
    return kinematic_loss, dynamics_loss, adaptive_loss


# =============================================================================
# 搂11  Backward Compatibility Layer
# =============================================================================

class LegacyRobustPINN(nn.Module):
    """Legacy-compatible wrapper for existing training code (v1.x API).

    [DEPRECATED] This wrapper preserves the v1.x call signature where
    ``forward()`` returns ``(pred_pos, confidence)``.  The confidence value
    is always ``torch.ones(B, 1)`` since ConfidenceGate has been removed.

    For new code, use ``RobustPINN`` + separate loss modules directly.

    Migration Guide:
        Old:  pred_pos, confidence = model(x_seq, validity, last_pos)
        New:  pred_pos = model(x_seq, validity, last_pos)
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_len: int = 10,
        dropout: float = 0.2,
        w_physics_max: float = 3.0,
        w_physics_min: float = 0.1,
    ) -> None:
        super().__init__()
        import warnings
        warnings.warn(
            "LegacyRobustPINN is deprecated.  Use RobustPINN + "
            "create_loss_modules() for the v2.0 API.",
            DeprecationWarning,
            stacklevel=2,
        )

        config = ModelConfig(
            n_features=n_features,
            pred_len=pred_len,
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=num_layers,
            dropout=dropout,
        )

        self.model = RobustPINN(config)
        self.kinematic_loss_fn = KinematicPhysicsLoss()
        self.dynamics_loss_fn = FossenDynamicsLoss(n_dof=config.n_dof)
        self.adaptive_loss_fn = AdaptiveRobustLoss()
        self.pred_len = pred_len

    def forward(
        self,
        x_seq: Tensor,
        validity: Tensor,
        last_pos: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Legacy forward: returns (pred_pos, dummy_confidence).

        The confidence tensor is always ones for backward compatibility.
        """
        pred_pos = self.model(x_seq, validity, last_pos)
        B = x_seq.shape[0]
        dummy_confidence = torch.ones(B, 1, device=x_seq.device)
        return pred_pos, dummy_confidence

    def compute_loss(
        self,
        pred_pos: Tensor,
        target_pos: Tensor,
        last_vel: Tensor,
        dt: float,
        confidence: Optional[Tensor] = None,
    ) -> Tuple[Tensor, float, float, float, float]:
        """Legacy loss computation.

        Args:
            pred_pos: [B, P, 3].
            target_pos: [B, P, 3].
            last_vel: [B, 3].
            dt: Time step.
            confidence: Ignored (kept for API compat).

        Returns:
            (loss_total, loss_data, loss_physics, w_data, w_phy).
        """
        loss_kin = self.kinematic_loss_fn(pred_pos, last_vel, dt)
        loss_dyn = self.dynamics_loss_fn(pred_pos, last_vel, dt)
        loss_total, log_dict = self.adaptive_loss_fn(
            pred_pos, target_pos, loss_kin, loss_dyn
        )
        return (
            loss_total,
            log_dict["loss_data_raw"],
            log_dict["loss_physics_raw"],
            log_dict["weight_data"],
            log_dict["weight_phy"],
        )


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    "ModelConfig",
    "RotaryPositionEmbedding",
    "RoPEMultiHeadAttention",
    "RoPETransformerEncoderLayer",
    "build_causal_mask",
    "build_validity_mask",
    "build_composite_mask",
    "AUVTransformerPredictor",
    "FossenDynamicsLoss",
    "KinematicPhysicsLoss",
    "AdaptiveRobustLoss",
    "RobustPINN",
    "create_model",
    "create_loss_modules",
    "LegacyRobustPINN",
]
