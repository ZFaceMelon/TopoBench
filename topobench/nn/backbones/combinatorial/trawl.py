"""TRAWL: topological random-walk sequence backbone.

TRAWL samples random walks over the cells of a lifted complex and encodes the
walk sequences with configurable sequence layers (Mamba, SISA, hybrid,
transformer, GRU, MLP or any Hydra-instantiated module).
"""

import copy
import functools
import hashlib
import json
import math
from bisect import bisect_right
from collections import OrderedDict
from itertools import accumulate

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import scatter

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised when numba is absent
    njit = None

from topobench.nn.encoders.base import AbstractFeatureEncoder

__all__ = ["TRAWL"]


# ---------------------------------------------------------------------------
# Sequence layers
# ---------------------------------------------------------------------------


class SequenceBlock(nn.Module):
    """Pre-normalized residual adapter for sequence-shaped modules.

    Parameters
    ----------
    module : torch.nn.Module
        Module mapping ``[walk, time, hidden]`` tensors to the same shape. A
        tuple output is reduced to its first element.
    d_model : int
        Hidden dimension used by the pre-normalization layer.
    """

    def __init__(self, module, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.module = module

    def residual(self, x):
        """Compute the module update on the normalized input.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Residual update with the same shape as ``x``.
        """
        result = self.module(self.norm(x))
        if isinstance(result, tuple):
            result = result[0]
        if result.shape != x.shape:
            raise ValueError(
                "Custom sequence layers must preserve [walk, time, hidden] shape"
            )
        return result

    def forward(self, x, residual_only=False):
        """Apply the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[walk, time, hidden]``.
        residual_only : bool, optional
            If True, return only the update instead of ``x + update``
            (default: False).

        Returns
        -------
        torch.Tensor
            Updated sequences, or the update alone, shaped like ``x``.
        """
        update = self.residual(x)
        return update if residual_only else x + update


class WalkGraphLayer(nn.Module):
    """Apply a PyG-compatible graph module to disjoint walk path graphs.

    Parameters
    ----------
    module : torch.nn.Module
        Graph module called as ``module(x, edge_index)``.
    bidirectional : bool, optional
        Whether path edges are added in both directions (default: True).
    """

    def __init__(self, module, bidirectional=True):
        super().__init__()
        self.module, self.bidirectional = module, bidirectional

    def forward(self, x):
        """Run the graph module on each walk viewed as a path graph.

        Parameters
        ----------
        x : torch.Tensor
            Walk states of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Updated walk states with the same shape as ``x``.
        """
        walks, length, hidden = x.shape
        sources = (
            torch.arange(walks * length, device=x.device)
            .reshape(walks, length)[:, :-1]
            .flatten()
        )
        edges = torch.stack((sources, sources + 1))
        if self.bidirectional:
            edges = torch.cat((edges, edges.flip(0)), dim=1)
        return self.module(x.reshape(-1, hidden), edges).reshape(
            walks, length, hidden
        )


def feed_forward(d_model, expansion=2, dropout=0.0):
    """Two-layer GELU feed-forward network.

    Parameters
    ----------
    d_model : int
        Input and output width.
    expansion : int, optional
        Hidden width multiplier (default: 2).
    dropout : float, optional
        Dropout between the two layers (default: 0.0).

    Returns
    -------
    torch.nn.Sequential
        The network.
    """
    return nn.Sequential(
        nn.Linear(d_model, expansion * d_model),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(expansion * d_model, d_model),
    )


def make_layer(config, d_model):
    """Construct a layer from a short name or a Hydra target.

    Parameters
    ----------
    config : str or dict
        Layer kind (``"sisa"``, ``"mamba"``, ``"gru"``, ``"mlp"``,
        ``"transformer"`` or ``"graph"``), or a mapping with a ``kind`` key
        plus layer options, or a Hydra config with a ``_target_`` key.
    d_model : int
        Hidden dimension of the layer.

    Returns
    -------
    torch.nn.Module
        Sequence layer preserving ``[walk, time, hidden]`` shapes.
    """
    config = {"kind": config} if isinstance(config, str) else dict(config)
    if "_target_" in config:
        from hydra.utils import instantiate

        return SequenceBlock(instantiate(config), d_model)
    kind = config.pop("kind")
    if kind == "sisa":
        return SISABlock(d_model=d_model, **config)
    if kind == "mamba":
        backend = config.pop("backend", "torch")
        if backend == "torch":
            module = PureTorchMambaBlock(d_model=d_model, **config)
        elif backend == "mamba_ssm":
            from mamba_ssm import Mamba

            # Base Hydra configs retain the pure-PyTorch scan setting when
            # switching backends. Official Mamba manages its own CUDA scan.
            config.pop("scan", None)
            module = Mamba(d_model=d_model, **config)
        else:
            raise ValueError(f"Unknown Mamba backend {backend}")
    elif kind == "gru":
        module = nn.GRU(d_model, d_model, batch_first=True, **config)
    elif kind == "mlp":
        module = feed_forward(d_model, **config)
    elif kind == "transformer":
        return nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.pop("n_heads", 8),
            batch_first=True,
            norm_first=True,
            **config,
        )
    elif kind == "graph":
        from hydra.utils import instantiate

        module = WalkGraphLayer(instantiate(config.pop("module")), **config)
    else:
        raise ValueError(f"Unknown TRAWL sequence layer {kind}")
    return SequenceBlock(module, d_model)


class PureTorchMambaBlock(nn.Module):
    """Selective SSM block for when mamba-ssm is unavailable.

    Parameters
    ----------
    d_model : int
        Input and output hidden dimension.
    d_state : int, optional
        SSM state dimension (default: 16).
    d_conv : int, optional
        Kernel size of the causal depthwise convolution (default: 4).
    expand : int, optional
        Expansion factor of the inner dimension (default: 2).
    scan : str, optional
        Scan implementation, ``"parallel"`` or ``"sequential"``
        (default: "parallel").
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        scan: str = "parallel",
    ):
        super().__init__()
        if scan not in {"parallel", "sequential"}:
            raise ValueError("scan must be parallel or sequential")
        self.scan = scan
        self.d_state = d_state
        self.d_inner = d_model * expand

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            self.d_inner, 1
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the selective SSM block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[batch, length, d_model]``.

        Returns
        -------
        torch.Tensor
            Output sequences of shape ``[batch, length, d_model]``.
        """
        _, seqlen, _ = x.shape
        x_branch, z = self.in_proj(x).chunk(2, dim=-1)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :seqlen]
        x_conv = F.silu(x_conv.transpose(1, 2))

        x_dbl = self.x_proj(x_conv)
        B = x_dbl[..., : self.d_state]
        C = x_dbl[..., self.d_state : 2 * self.d_state]
        dt = F.softplus(self.dt_proj(x_dbl[..., -1:]))
        A = -torch.exp(self.A_log.float())

        y = self._selective_scan(x_conv, dt, A, B, C)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, u, dt, A, B, C):
        """Run the selective scan, sequentially or as a parallel prefix scan.

        Parameters
        ----------
        u : torch.Tensor
            Inputs, ``[batch, length, d_inner]``.
        dt : torch.Tensor
            Positive step sizes, ``[batch, length, d_inner]``.
        A : torch.Tensor
            Transition rates, ``[d_inner, d_state]``.
        B : torch.Tensor
            Input projections, ``[batch, length, d_state]``.
        C : torch.Tensor
            Output projections, ``[batch, length, d_state]``.

        Returns
        -------
        torch.Tensor
            Outputs, ``[batch, length, d_inner]``.
        """
        bsz, seqlen, d_inner = u.shape
        deltaA = torch.exp(dt.unsqueeze(-1) * A)  # (B, L, D, N)
        deltaB_u = dt.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        use_seq = self.scan == "sequential"
        if use_seq:
            # Preallocate output to avoid Python list + torch.stack overhead.
            n = A.shape[1]
            h = u.new_zeros(bsz, d_inner, n)
            ys = u.new_zeros(bsz, seqlen, d_inner)
            for t in range(seqlen):
                h = deltaA[:, t] * h + deltaB_u[:, t]
                ys[:, t] = (h * C[:, t].unsqueeze(1)).sum(-1)
            return ys

        # Parallel associative scan (log-depth)
        n_pad = 1 << ((seqlen - 1).bit_length())
        if n_pad != seqlen:
            deltaA = F.pad(deltaA, (0, 0, 0, 0, 0, n_pad - seqlen), value=1.0)
            deltaB_u = F.pad(
                deltaB_u, (0, 0, 0, 0, 0, n_pad - seqlen), value=0.0
            )

        k = 1
        while k < n_pad:
            A_prev = F.pad(deltaA[:, :-k], (0, 0, 0, 0, k, 0), value=1.0)
            X_prev = F.pad(deltaB_u[:, :-k], (0, 0, 0, 0, k, 0), value=0.0)
            deltaB_u = deltaA * X_prev + deltaB_u
            deltaA = deltaA * A_prev
            k *= 2

        h = deltaB_u[:, :seqlen]
        return (h * C.unsqueeze(2)).sum(-1)


def sequence_cumsum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Cumulative sum that stays deterministic on CUDA when requested.

    CUDA ``cumsum`` has no deterministic kernel. Under deterministic mode the
    short walk dimension is summed with a float32 lower-triangular matmul.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    dim : int
        Dimension along which to accumulate.

    Returns
    -------
    torch.Tensor
        Cumulative sum of ``x`` along ``dim`` with the dtype of ``x``.
    """
    if not (x.is_cuda and torch.are_deterministic_algorithms_enabled()):
        return torch.cumsum(x, dim=dim)
    length = x.shape[dim]
    with torch.autocast(x.device.type, enabled=False):
        lower = torch.ones(
            length, length, device=x.device, dtype=torch.float32
        ).tril()
        moved = x.movedim(dim, -1).float() @ lower.T
    return moved.movedim(-1, dim).to(x.dtype)


def build_rope_cache(seq_len: int, dim: int, device, dtype):
    """Build rotary position embedding cosine and sine tables.

    Parameters
    ----------
    seq_len : int
        Number of positions.
    dim : int
        Rotary dimension; ``dim // 2`` frequencies are used.
    device : torch.device
        Device of the returned tables.
    dtype : torch.dtype
        Dtype of the returned tables.

    Returns
    -------
    tuple of torch.Tensor
        Cosine and sine tables, each of shape ``[1, 1, seq_len, dim // 2]``.
    """
    half = dim // 2
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(half, device=device, dtype=torch.float32) / half)
    )
    freqs = torch.outer(
        torch.arange(seq_len, device=device, dtype=torch.float32), inv_freq
    )
    return freqs.cos().to(dtype)[None, None], freqs.sin().to(dtype)[None, None]


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Rotate interleaved feature pairs by the given phases.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``[..., dim]``.
    cos : torch.Tensor
        Phase cosines broadcastable to ``[..., dim // 2]``.
    sin : torch.Tensor
        Phase sines broadcastable to ``[..., dim // 2]``.

    Returns
    -------
    torch.Tensor
        Rotated tensor with the same shape as ``x``.
    """
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack(
        (x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1
    ).flatten(-2)


class SISALayer(nn.Module):
    """Multi-head state-space augmented causal attention.

    Parameters
    ----------
    d_model : int
        Input and output hidden dimension; must be divisible by ``n_heads``.
    n_heads : int, optional
        Number of attention heads (default: 8).
    d_ssm : int, optional
        Per-head state-space key dimension; must be even (default: 16).
    attention_dropout : float, optional
        Dropout probability on attention weights during training
        (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ssm: int = 16,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        self.attention_dropout = attention_dropout
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_ssm = d_ssm
        if self.d_head % 2 or d_ssm % 2:
            raise ValueError("SISA RoPE dimensions must be even")
        h, dh, ds = n_heads, self.d_head, d_ssm
        self.qkv = nn.Linear(d_model, 3 * h * dh)
        self.b_proj = nn.Linear(d_model, h * ds)
        self.c_proj = nn.Linear(d_model, h * ds)
        self.alpha_proj = nn.Linear(d_model, h)
        self.theta_proj = nn.Linear(d_model, h * (ds // 2))
        self.lambd = nn.Parameter(torch.full((h,), 0.01))
        self.out_proj = nn.Linear(h * dh, d_model)

        # alpha=exp(-softplus(-5)) ~= .9933: about a 103-step half-life.
        # Unlike the old Zinc clamp formulation, this starts with nonzero gradients.
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -5.0)
        nn.init.normal_(self.theta_proj.weight, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)

    # Beyond this cumulative log-decay span the centered factors exp(+-span/2)
    # and masked products exp(span) leave the safe fp32/bf16 range.
    MAX_FACTORED_SPAN = 60.0

    def _explicit_attention(self, q, k, v, c_ssm, b_ssm, g, scale, dh):
        """Causal attention with the decay formed only for ``j <= i``.

        Stays finite under arbitrarily strong decay. It omits dropout so it
        draws no random numbers, and only replaces heads that would overflow.

        Parameters
        ----------
        q, k, v : torch.Tensor
            Queries, keys and values, ``[batch, heads, length, d_head]``.
        c_ssm, b_ssm : torch.Tensor
            State-space factors, ``[batch, heads, length, d_ssm]``.
        g : torch.Tensor
            Cumulative log-decay, ``[batch, heads, length, 1]``.
        scale : torch.Tensor
            Per-head state-space scale, ``[1, heads, 1, 1]``.
        dh : int
            Head dimension.

        Returns
        -------
        torch.Tensor
            Attention outputs, ``[batch, heads, length, d_head]``.
        """
        length = q.shape[-2]
        causal = torch.ones(
            length, length, dtype=torch.bool, device=q.device
        ).tril()
        difference = (g - g.transpose(-1, -2)).masked_fill(~causal, 0.0)
        decay = torch.exp(difference) * causal
        state = (scale * c_ssm) @ (scale * b_ssm).transpose(-1, -2)
        logits = q @ k.transpose(-1, -2) + state * decay.to(state)
        logits = (logits / math.sqrt(dh)).masked_fill(~causal, -torch.inf)
        return logits.float().softmax(dim=-1).to(v.dtype) @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply state-space augmented causal attention.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[batch, length, d_model]``.

        Returns
        -------
        torch.Tensor
            Output sequences of shape ``[batch, length, d_model]``.
        """
        bsz, length, _ = x.shape
        h, dh, ds = self.n_heads, self.d_head, self.d_ssm
        qkv = self.qkv(x).view(bsz, length, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        cos, sin = build_rope_cache(length, dh, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        b_ssm = self.b_proj(x).view(bsz, length, h, ds).permute(0, 2, 1, 3)
        c_ssm = self.c_proj(x).view(bsz, length, h, ds).permute(0, 2, 1, 3)
        theta = (
            self.theta_proj(x)
            .view(bsz, length, h, ds // 2)
            .permute(0, 2, 1, 3)
        )
        phase = sequence_cumsum(theta, dim=2)
        cos, sin = phase.cos(), phase.sin()
        b_ssm, c_ssm = apply_rope(b_ssm, cos, sin), apply_rope(c_ssm, cos, sin)

        decay = F.softplus(self.alpha_proj(x))
        alpha = torch.exp(-decay).clamp(1e-4, 0.9999)
        alpha = alpha.permute(0, 2, 1).unsqueeze(-1)
        g = sequence_cumsum(torch.log(alpha), dim=2)
        scale = (dh**0.25) * torch.sqrt(F.softplus(self.lambd) + 1e-6)
        scale = scale.view(1, h, 1, 1)
        # Per-head flag, computed without a host sync so compiled graphs stay
        # whole. Wide heads use the explicit path; the factored path sees a
        # neutral decay there so neither branch (nor its gradient) overflows.
        span = g.amax(dim=2, keepdim=True) - g.amin(dim=2, keepdim=True)
        wide = span.detach() > self.MAX_FACTORED_SPAN
        explicit = self._explicit_attention(
            q, k, v, c_ssm, b_ssm, g, scale, dh
        )
        g = torch.where(wide, torch.zeros_like(g), g)
        # Center the two exponential factors to avoid overflow without changing
        # their pairwise product exp(g_i - g_j).
        center = (
            g.detach().amin(dim=2, keepdim=True)
            + g.detach().amax(dim=2, keepdim=True)
        ) / 2
        c_bar = torch.exp(g - center) * c_ssm
        b_bar = torch.exp(-g + center) * b_ssm

        q_aug = torch.cat((q, scale * c_bar), dim=-1)
        k_aug = torch.cat((k, scale * b_bar), dim=-1)
        y = F.scaled_dot_product_attention(
            q_aug,
            k_aug,
            v,
            is_causal=True,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(dh),
        )
        y = torch.where(wide, explicit.to(y.dtype), y)
        return self.out_proj(
            y.transpose(1, 2).contiguous().view(bsz, length, h * dh)
        )


class SISABlock(nn.Module):
    """Pre-normalized SISA attention followed by a feed-forward network.

    Parameters
    ----------
    d_model : int
        Input and output hidden dimension.
    n_heads : int, optional
        Number of attention heads (default: 8).
    d_ssm : int, optional
        Per-head state-space key dimension (default: 16).
    attention_dropout : float, optional
        Dropout probability on attention weights (default: 0.1).
    dropout : float, optional
        Dropout probability inside the feed-forward network (default: 0.3).
    expansion : int, optional
        Hidden expansion factor of the feed-forward network (default: 2).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ssm: int = 16,
        attention_dropout: float = 0.1,
        dropout: float = 0.3,
        expansion: int = 2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.sisa = SISALayer(
            d_model,
            n_heads=n_heads,
            d_ssm=d_ssm,
            attention_dropout=attention_dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = feed_forward(d_model, expansion, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SISA block.

        Parameters
        ----------
        x : torch.Tensor
            Input sequences of shape ``[batch, length, d_model]``.

        Returns
        -------
        torch.Tensor
            Output sequences of shape ``[batch, length, d_model]``.
        """
        x = x + self.sisa(self.norm1(x))
        return x + self.ffn(self.norm2(x))


# ---------------------------------------------------------------------------
# Walk sampling
# ---------------------------------------------------------------------------
# Weighted non-backtracking random walks with coverage-biased starts. The
# optional numba kernel reproduces ``simulate_nbrw_sparse`` bit for bit: it
# draws one ``Generator.random()`` per step from the caller's generator and
# repeats NumPy's floating-point operation order (sequential sums below
# eight values, pairwise summation above, sequential cumulative sums).


def transition(matrix):
    """Row-normalize positive weights, retaining isolated states.

    Non-finite and nonpositive weights are dropped; empty rows receive a
    self-loop.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        Row-stochastic transition matrix.
    """
    matrix = matrix.astype(np.float64).tocsr(copy=True)
    matrix.data = np.where(
        np.isfinite(matrix.data) & (matrix.data > 0), matrix.data, 0
    )
    matrix.eliminate_zeros()
    empty = np.asarray(matrix.sum(axis=1)).ravel() == 0
    matrix = matrix + sp.diags(empty.astype(float))
    return sp.diags(1 / np.asarray(matrix.sum(axis=1)).ravel()) @ matrix


def chung_laplacian(matrix):
    """Symmetric directed Laplacian with a power-iterated stationary measure.

    Laziness makes iteration converge on bipartite Hasse graphs. Disconnected
    components retain their uniform-initialization mass.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix.

    Returns
    -------
    scipy.sparse.spmatrix
        Symmetric Chung Laplacian of shape ``(n, n)``.
    """
    p = transition(matrix)
    n = p.shape[0]
    pi = np.full(n, 1 / n)
    for _ in range(1000):
        updated = (pi + p.T @ pi) / 2
        if np.max(np.abs(updated - pi)) < 1e-12:
            pi = updated
            break
        pi = updated
    pi = np.maximum(pi, 1e-15)
    normalized = sp.diags(np.sqrt(pi)) @ p @ sp.diags(1 / np.sqrt(pi))
    return sp.eye(n) - (normalized + normalized.T) / 2


def guided_transition(matrix, gamma=0.2, diffusion_t=0.1, dense_limit=2048):
    """Heat-kernel-guided weights on existing transitions only.

    Exact dense decomposition is deliberately bounded: never silently change
    an experiment to an approximation when its graph exceeds the limit.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Square weighted adjacency matrix.
    gamma : float, optional
        Nonnegative exponent on the heat-kernel magnitude; 0 disables
        guidance (default: 0.2).
    diffusion_t : float, optional
        Nonnegative heat-kernel diffusion time (default: 0.1).
    dense_limit : int, optional
        Maximum number of states for the dense eigendecomposition
        (default: 2048).

    Returns
    -------
    scipy.sparse.csr_matrix
        Row-stochastic guided transition matrix.

    Raises
    ------
    ValueError
        If ``gamma`` or ``diffusion_t`` is negative, or guidance is enabled
        and the matrix exceeds ``dense_limit``.
    """
    if gamma < 0 or diffusion_t < 0:
        raise ValueError(
            "Guidance gamma and diffusion time must be nonnegative"
        )
    p = transition(matrix).tocsr()
    if gamma == 0:
        return p
    if matrix.shape[0] > dense_limit:
        raise ValueError(
            "Heat guidance exceeds dense_limit; disable guidance or explicitly raise the limit"
        )
    eigenvalues, vectors = scipy.linalg.eigh(chung_laplacian(matrix).toarray())
    heat = (vectors * np.exp(-diffusion_t * eigenvalues)) @ vectors.T
    coo = p.tocoo()
    coo.data *= np.abs(heat[coo.row, coo.col]) ** gamma
    return transition(coo.tocsr())


def walk_seed(*parts):
    """Collision-free sampling seed from integer seed components.

    Additive offsets alias distinct (identity, step, view) tuples; hashing
    through ``SeedSequence`` keeps every combination distinct.

    Parameters
    ----------
    *parts : tuple
        Integer-convertible seed components, reduced modulo ``2**63``.

    Returns
    -------
    int
        Seed derived from all components.
    """
    entropy = [int(part) % 2**63 for part in parts]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


class WalkSampler:
    """Bounded cache of CPU transition distributions; paths still resample.

    Cache keys include actual weights, so masked-topology pretraining cannot
    accidentally reuse an unmasked transition matrix. Derived cache state is
    intentionally excluded from neural checkpoints.

    Parameters
    ----------
    cache_size : int, optional
        Maximum number of cached transition distributions; 0 disables
        caching (default: 1024).
    """

    def __init__(self, cache_size=1024):
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def __call__(self, matrix, guidance=None, **kwargs):
        """Sample walks on ``matrix`` using a cached transition distribution.

        Parameters
        ----------
        matrix : scipy.sparse.spmatrix
            Square weighted relation matrix over graph-local states.
        guidance : dict, optional
            Keyword arguments for ``guided_transition``; plain ``transition``
            is used when empty (default: None).
        **kwargs : dict
            Options forwarded to ``sample_neighbors`` (``k``, ``length``,
            ``seed``, ``start_policy``, ``epsilon``, ``reverse``).

        Returns
        -------
        numpy.ndarray
            Walks of graph-local state IDs with shape ``(num_walks, length)``.
        """
        matrix = matrix.tocsr()
        digest = hashlib.sha256()
        for value in (matrix.data, matrix.indices, matrix.indptr):
            digest.update(value.tobytes())
        key = (
            matrix.shape,
            digest.digest(),
            json.dumps(dict(guidance or {}), sort_keys=True),
        )
        if key in self.cache:
            active, neighbors, probs, rows = self.cache.pop(key)
            self.cache[key] = (active, neighbors, probs, rows)
        else:
            active = np.flatnonzero(
                np.asarray((matrix + matrix.T).sum(axis=1)).ravel() > 0
            )
            if not len(active):
                return np.empty((0, kwargs.get("length", 32)), dtype=np.int64)
            subgraph = matrix[active][:, active]
            p = (
                guided_transition(subgraph, **guidance)
                if guidance
                else transition(subgraph)
            )
            neighbors, probs = csr_rows(p)
            rows = prepare_walk_rows(neighbors, probs)
            if self.cache_size:
                self.cache[key] = (active, neighbors, probs, rows)
                if len(self.cache) > self.cache_size:
                    self.cache.popitem(last=False)
        return active[sample_neighbors(neighbors, probs, rows=rows, **kwargs)]


def csr_rows(matrix):
    """Split a CSR matrix into per-row column and value lists.

    Parameters
    ----------
    matrix : scipy.sparse.spmatrix
        Sparse matrix.

    Returns
    -------
    tuple of list
        ``(neighbors, probs)``: column indices and values of each row.
    """
    matrix = matrix.tocsr()
    bounds = zip(matrix.indptr[:-1], matrix.indptr[1:], strict=True)
    neighbors, probs = [], []
    for start, stop in bounds:
        neighbors.append(matrix.indices[start:stop].tolist())
        probs.append(matrix.data[start:stop].tolist())
    return neighbors, probs


def sample_neighbors(
    neighbors,
    probs,
    k=32,
    length=32,
    seed=0,
    start_policy="coverage",
    epsilon=0.05,
    reverse=False,
    rows=None,
):
    """Sample ``k`` non-backtracking walks over adjacency lists.

    Coverage starts favour rarely visited states until coverage stops
    growing by ``epsilon``, then fall back to uniform starts.

    Parameters
    ----------
    neighbors : list of list of int
        Neighbor state indices for each state.
    probs : list of list of float
        Transition weights aligned with ``neighbors``.
    k : int, optional
        Number of walks to sample (default: 32).
    length : int, optional
        Number of states per walk (default: 32).
    seed : int, optional
        Seed for the random generator (default: 0).
    start_policy : str, optional
        Start-state policy, ``"uniform"`` or ``"coverage"``
        (default: "coverage").
    epsilon : float, optional
        Coverage-change threshold below which coverage starts fall back to
        uniform starts (default: 0.05).
    reverse : bool, optional
        If True, append the reversal of every walk (default: False).
    rows : WalkRows, optional
        Precomputed output of ``prepare_walk_rows`` (default: None).

    Returns
    -------
    numpy.ndarray
        Walks of state indices with shape ``(num_walks, length)``.
    """
    if k < 1 or length < 1 or start_policy not in {"uniform", "coverage"}:
        raise ValueError("Invalid walk budget, length or start policy")
    if not neighbors:
        return np.empty((0, length), dtype=np.int64)
    rng = np.random.default_rng(seed)
    if rows is None:
        rows = prepare_walk_rows(neighbors, probs)
    visits = np.zeros(len(neighbors), dtype=np.float64)
    previous = 0.0
    paths = []
    for _ in range(k):
        if start_policy == "coverage" and visits.sum() > 0:
            coverage = float((visits > 0).mean())
            if abs(coverage - previous) < epsilon:
                start = int(rng.integers(0, len(neighbors)))
            else:
                weights = 1 / (1 + visits)
                start = int(
                    rng.choice(len(neighbors), p=weights / weights.sum())
                )
            previous = coverage
        else:
            start = int(rng.integers(0, len(neighbors)))
        path = simulate_nbrw_sparse(
            neighbors, probs, start, length, rng, rows=rows
        )
        np.add.at(visits, path, 1)
        paths.append(path)
    paths = np.asarray(paths, dtype=np.int64)
    return np.concatenate((paths, paths[:, ::-1])) if reverse else paths


class WalkRows(list):
    """Per-state ``(neighbors, weights)`` rows plus cached kernel arrays."""

    csr = None


def prepare_walk_rows(neighbors, neigh_probs):
    """Per-state neighbor lists with unusable weights already zeroed.

    NaN, infinite and nonpositive weights are unavailable transitions.
    Preparing once per graph avoids repeating this on every walk step.

    Parameters
    ----------
    neighbors : list of list of int
        Neighbor state indices for each state.
    neigh_probs : list of list of float
        Transition weights aligned with ``neighbors``.

    Returns
    -------
    WalkRows
        One ``(neighbors, weights)`` pair per state with cleaned weights.
    """
    rows = WalkRows()
    for nbrs, row in zip(neighbors, neigh_probs, strict=True):
        raw = np.asarray(row, dtype=np.float64)
        clean = np.where(np.isfinite(raw) & (raw > 0.0), raw, 0.0)
        rows.append(([int(n) for n in nbrs], clean.tolist()))
    return rows


def _numpy_sum(values):
    """Sum floats in exactly the order ``ndarray.sum`` uses.

    NumPy adds fewer than eight values sequentially and otherwise keeps eight
    running partial sums (pairwise summation). Reproducing that order keeps
    sampled walks bit-identical to ``Generator.choice`` without its overhead.

    Parameters
    ----------
    values : list of float
        Values to sum.

    Returns
    -------
    float
        Sum of ``values``.
    """
    n = len(values)
    if n < 8:
        total = 0.0
        for value in values:
            total += value
        return total
    if n > 128:
        # Longer blocks recurse; defer to NumPy itself.
        return float(np.sum(values))
    r = list(values[:8])
    stop = n - n % 8
    for i in range(8, stop, 8):
        for j in range(8):
            r[j] += values[i + j]
    total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    for i in range(stop, n):
        total += values[i]
    return total


def simulate_nbrw_sparse(
    neighbors: list[list[int]],
    neigh_probs: list[list[float]],
    start: int,
    length: int,
    rng: np.random.Generator,
    rows=None,
) -> list[int]:
    """Non-backtracking weighted walk of ``length`` states.

    Each step draws exactly as ``rng.choice(len(nbrs), p=probs)`` does: one
    uniform sample against the normalized cumulative weights. Plain-float
    arithmetic reproduces NumPy's results bit for bit while avoiding its
    per-call overhead on these short rows. Pass ``rows`` from
    ``prepare_walk_rows`` when sampling several walks on one graph.

    Parameters
    ----------
    neighbors : list of list of int
        Neighbor state indices for each state.
    neigh_probs : list of list of float
        Transition weights aligned with ``neighbors``.
    start : int
        Initial state index.
    length : int
        Number of states in the walk.
    rng : numpy.random.Generator
        Random generator supplying one uniform draw per step.
    rows : WalkRows, optional
        Precomputed output of ``prepare_walk_rows`` (default: None).

    Returns
    -------
    list of int
        Visited state indices, starting with ``start``.
    """
    if rows is None:
        rows = prepare_walk_rows(neighbors, neigh_probs)
    if _numba_available and isinstance(rows, WalkRows):
        # Compiled kernel with the identical arithmetic and random draws.
        if rows.csr is None:
            rows.csr = _numba_walk_csr(rows)
        return _numba_walk(rows.csr, start, length, rng)
    current = int(start)
    prev = -1
    path = [current]
    for _ in range(length - 1):
        nbrs, clean = rows[current]
        probs = (
            [0.0 if n == prev else p for n, p in zip(nbrs, clean, strict=True)]
            if prev >= 0
            else clean
        )
        total = _numpy_sum(probs)
        if not np.isfinite(total) or total <= 0.0:
            # Backtracking is allowed when it is the only usable move.
            probs = clean
            total = _numpy_sum(probs)
            if not np.isfinite(total) or total <= 0.0:
                if nbrs:
                    probs, total = [1.0 / len(nbrs)] * len(nbrs), 1.0
                else:
                    # Defensive only: graph construction normally inserts a
                    # self-loop for isolated states.
                    path.append(current)
                    prev = current
                    continue
        cdf = list(accumulate(p / total for p in probs))
        last = cdf[-1]
        cdf = [c / last for c in cdf]
        nxt = nbrs[bisect_right(cdf, rng.random())]
        prev, current = current, nxt
        path.append(current)
    return path


_numba_available = njit is not None


def _numba_walk_csr(rows):
    """Flatten ``prepare_walk_rows`` output into CSR arrays for the kernel.

    Parameters
    ----------
    rows : list of tuple
        Per-state ``(neighbors, weights)`` pairs from ``prepare_walk_rows``.

    Returns
    -------
    tuple
        ``(indptr, indices, weights, width)`` where ``width`` is the largest
        row length (at least 1).
    """
    lengths = np.fromiter((len(nbrs) for nbrs, _ in rows), np.int64, len(rows))
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    np.cumsum(lengths, out=indptr[1:])
    indices = np.fromiter(
        (n for nbrs, _ in rows for n in nbrs), np.int64, int(indptr[-1])
    )
    weights = np.fromiter(
        (w for _, row in rows for w in row), np.float64, int(indptr[-1])
    )
    width = int(lengths.max()) if len(lengths) else 0
    return indptr, indices, weights, max(width, 1)


if _numba_available:

    @njit(cache=True)
    def _block_sum(values, start, n):
        # NumPy's pairwise_sum base case (blocks of at most 128 values).
        if n < 8:
            total = 0.0
            for i in range(start, start + n):
                total += values[i]
            return total
        r0 = values[start]
        r1 = values[start + 1]
        r2 = values[start + 2]
        r3 = values[start + 3]
        r4 = values[start + 4]
        r5 = values[start + 5]
        r6 = values[start + 6]
        r7 = values[start + 7]
        stop = n - n % 8
        for i in range(8, stop, 8):
            base = start + i
            r0 += values[base]
            r1 += values[base + 1]
            r2 += values[base + 2]
            r3 += values[base + 3]
            r4 += values[base + 4]
            r5 += values[base + 5]
            r6 += values[base + 6]
            r7 += values[base + 7]
        total = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        for i in range(stop, n):
            total += values[start + i]
        return total

    @njit(cache=True)
    def _pairwise_sum(values, start, n):
        """NumPy's float64 pairwise sum, with its recursion made explicit.

        Numba cannot reliably load cached recursive functions, so the
        recursive halving is replayed with a stack in the same order.
        """
        if n <= 128:
            return _block_sum(values, start, n)
        frames = np.empty((128, 3), dtype=np.int64)
        partial = np.empty(128, dtype=np.float64)
        frames[0, 0], frames[0, 1], frames[0, 2] = start, n, 0
        top, filled = 1, 0
        while top > 0:
            top -= 1
            first, size, phase = frames[top, 0], frames[top, 1], frames[top, 2]
            if size <= 128:
                partial[filled] = _block_sum(values, first, size)
                filled += 1
            elif phase == 0:
                half = size // 2
                half -= half % 8
                frames[top, 0], frames[top, 1], frames[top, 2] = first, size, 1
                frames[top + 1, 0], frames[top + 1, 1] = (
                    first + half,
                    size - half,
                )
                frames[top + 1, 2] = 0
                frames[top + 2, 0], frames[top + 2, 1], frames[top + 2, 2] = (
                    first,
                    half,
                    0,
                )
                top += 3
            else:
                partial[filled - 2] = partial[filled - 2] + partial[filled - 1]
                filled -= 1
        return partial[0]

    @njit(cache=True)
    def _walk(indptr, indices, weights, width, start, length, rng):
        path = np.empty(length, dtype=np.int64)
        probs = np.empty(width, dtype=np.float64)
        cdf = np.empty(width, dtype=np.float64)
        current = start
        previous = -1
        path[0] = current
        for step in range(1, length):
            lo = indptr[current]
            n = indptr[current + 1] - lo
            for j in range(n):
                probs[j] = (
                    0.0 if indices[lo + j] == previous else weights[lo + j]
                )
            total = _pairwise_sum(probs, 0, n)
            if not (np.isfinite(total) and total > 0.0):
                # Backtracking is allowed when it is the only usable move.
                for j in range(n):
                    probs[j] = weights[lo + j]
                total = _pairwise_sum(probs, 0, n)
                if not (np.isfinite(total) and total > 0.0):
                    if n == 0:
                        path[step] = current
                        previous = current
                        continue
                    for j in range(n):
                        probs[j] = 1.0 / n
                    total = 1.0
            running = 0.0
            for j in range(n):
                running = (
                    probs[j] / total if j == 0 else running + probs[j] / total
                )
                cdf[j] = running
            last = cdf[n - 1]
            for j in range(n):
                cdf[j] = cdf[j] / last
            draw = rng.random()
            index = 0
            while index < n and cdf[index] <= draw:
                index += 1
            previous = current
            current = indices[lo + index]
            path[step] = current
        return path


def _numba_walk(csr, start, length, rng):
    """Sample one walk with the compiled kernel.

    Parameters
    ----------
    csr : tuple
        ``(indptr, indices, weights, width)`` arrays from ``_numba_walk_csr``.
    start : int
        Initial state index.
    length : int
        Number of states in the walk.
    rng : numpy.random.Generator
        Random generator supplying one uniform draw per step.

    Returns
    -------
    list of int
        Visited state indices, starting with ``start``.
    """
    indptr, indices, weights, width = csr
    return _walk(
        indptr, indices, weights, width, int(start), int(length), rng
    ).tolist()


# ---------------------------------------------------------------------------
# Base TRAWL backbone
# ---------------------------------------------------------------------------


def upload(array, device):
    """Copy a host array to ``device`` without making the host wait.

    Integer arrays become int64 on every platform. CUDA copies go through
    pinned memory with ``non_blocking`` so they never synchronize.

    Parameters
    ----------
    array : array_like
        Host array to copy.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor
        Tensor on ``device``; integer and boolean arrays become int64.
    """
    array = np.ascontiguousarray(array)
    if array.dtype.kind in "iub":
        array = array.astype(np.int64, copy=False)
    tensor = torch.from_numpy(array)
    if torch.device(device).type != "cuda":
        return tensor.to(device)
    return tensor.pin_memory().to(device, non_blocking=True)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Batch norm that uses running statistics for single-row batches."""

    def forward(self, x):
        """Normalize ``x``; a one-row training batch uses running stats.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``[rows, features]``.

        Returns
        -------
        torch.Tensor
            Normalized input.
        """
        if self.training and len(x) < 2:
            return nn.functional.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                0.0,
                self.eps,
            )
        return super().forward(x)


class NeighborhoodFusion(nn.Module):
    """Fuse available neighborhood embeddings, preserving empty-slot semantics.

    Parameters
    ----------
    dim : int
        Dimension of each neighborhood embedding.
    count : int
        Number of neighborhoods.
    mode : str, optional
        Fusion mode: ``"mean"``, ``"concat"``, ``"learned"`` (a global
        learnable logit per neighborhood, softmax over available ones),
        ``"attention"`` (a score per neighborhood embedding) or
        ``"gate_mlp"`` (per-graph softmax gate from a small MLP on the
        concatenated, availability-masked embeddings) (default: "mean").
    """

    GATED = ("learned", "attention", "gate_mlp")

    def __init__(self, dim, count, mode="mean"):
        super().__init__()
        if mode not in {"mean", "concat", "learned", "attention", "gate_mlp"}:
            raise ValueError(f"Unknown neighborhood fusion {mode}")
        self.mode = mode
        self.weights = (
            nn.Parameter(torch.zeros(count)) if mode == "learned" else None
        )
        self.attention = nn.Linear(dim, 1) if mode == "attention" else None
        self.project = (
            nn.Linear(count * (dim + 1), dim) if mode == "concat" else None
        )
        if mode == "gate_mlp":
            self.gate = nn.Sequential(
                nn.Linear(count * (dim + 1), dim),
                nn.GELU(),
                nn.Linear(dim, count),
            )

    def forward(self, values, available, fallback, return_weights=False):
        """Fuse neighborhood embeddings, ignoring unavailable slots.

        Parameters
        ----------
        values : torch.Tensor
            Neighborhood embeddings of shape ``[count, dim]``.
        available : torch.Tensor
            Boolean mask of shape ``[count]`` marking neighborhoods with walks.
        fallback : torch.Tensor
            Embedding of shape ``[dim]`` returned when none is available.

        return_weights : bool, optional
            Also return the softmax weights over neighborhoods (zeros if
            none is available, None for ``"concat"``) (default: False).

        Returns
        -------
        torch.Tensor
            Fused embedding of shape ``[dim]`` (and the weights).
        """
        if not bool(available.any()):
            return (
                (fallback, values.new_zeros(len(values)))
                if return_weights
                else fallback
            )
        if self.mode == "concat":
            slots = torch.cat(
                (values * available[:, None], available[:, None].to(values)),
                dim=1,
            )
            fused = self.project(slots.flatten())
            return (fused, None) if return_weights else fused
        if self.mode == "mean":
            scores = values.new_zeros(len(values))
        elif self.mode == "learned":
            scores = self.weights
        elif self.mode == "gate_mlp":
            slots = torch.cat(
                (values * available[:, None], available[:, None].to(values)),
                dim=1,
            )
            scores = self.gate(slots.flatten())
        else:
            scores = self.attention(values).squeeze(-1)
        weights = scores.masked_fill(~available, -torch.inf).softmax(dim=0)
        fused = (values * weights[:, None]).sum(dim=0)
        return (fused, weights) if return_weights else fused


# ---------------------------------------------------------------------------
# Host walk sampling, optionally in a process pool
# ---------------------------------------------------------------------------

_WORKER_SAMPLER = None
_POOLS = {}


def _init_sampling_worker(cache_size):
    """Give a sampling worker its own transition cache.

    Parameters
    ----------
    cache_size : int
        Transition distributions cached per worker.
    """
    global _WORKER_SAMPLER
    _WORKER_SAMPLER = WalkSampler(cache_size)
    torch.set_num_threads(1)


def _sampling_pool(workers, cache_size):
    """Return a persistent fork pool of sampling workers.

    Parameters
    ----------
    workers : int
        Number of processes.
    cache_size : int
        Transition cache size per worker.

    Returns
    -------
    multiprocessing.pool.Pool
        The pool (created on first use).
    """
    key = (int(workers), int(cache_size))
    if key not in _POOLS:
        import multiprocessing as mp

        _POOLS[key] = mp.get_context("fork").Pool(
            workers,
            initializer=_init_sampling_worker,
            initargs=(cache_size,),
        )
    return _POOLS[key]


def _sample_graph(spec, item, sampler=None):
    """Sample every family and view of one graph (host side, numpy only).

    Each graph is seeded independently (``walk_seed`` of the model seed,
    graph identity, sampling step, family and view), so sampling graphs in
    worker processes gives exactly the walks of the serial loop.

    Parameters
    ----------
    spec : dict
        Sampling options of the backbone for this batch.
    item : tuple
        ``(graph_id, total, first, identity, edges, weights, signs, pivots,
        cell_vertices, count_row)`` with graph-local indices.
    sampler : WalkSampler, optional
        Sampler with its transition cache; the worker's own if None
        (default: None).

    Returns
    -------
    list
        Per family, ten lists of arrays in the slot order of ``sampled``
        (paths, owners, views, sign codes, running codes, pivots, vertex
        ids, crossed, pointers, vertex states); paths, pivots and vertex
        states are graph-local (-1 for none).
    """
    sampler = sampler if sampler is not None else _WORKER_SAMPLER
    (
        graph_id,
        total,
        first,
        identity,
        edges,
        weights,
        edge_signs,
        edge_pivots,
        cell_vertices,
        count_row,
    ) = item
    groups, views = spec["groups"], spec["views"]
    out = [[[] for _ in range(10)] for _ in groups]
    use_pivots, use_signs = spec["use_pivots"], spec["use_signs"]
    if use_pivots:
        pivot_table = np.full((total, total), -1, dtype=np.int64)
        pivot_table[edges[:, 0], edges[:, 1]] = edge_pivots
    if use_signs:
        # Code per ordered state pair: 0 none, 1 for +1, 2 for -1.
        sign_table = np.zeros((total, total), dtype=np.int64)
        sign_table[edges[:, 0], edges[:, 1]] = np.where(
            edge_signs > 0, 1, np.where(edge_signs < 0, 2, 0)
        )
    for group_id, relation in enumerate(groups):
        select = (
            np.ones(len(edges), dtype=bool)
            if relation is None
            else edges[:, 2] == relation
        )
        if spec["fallback_family"] is not None and not select.any():
            select = edges[:, 2] == spec["fallback_family"]
        chosen = edges[select]
        matrix = sp.csr_matrix(
            (weights[select], (chosen[:, 0], chosen[:, 1])),
            shape=(total, total),
        )
        for view in range(views):
            seed = walk_seed(spec["seed"], identity, spec["step"], group_id, view)
            walk_options = (
                spec["walks"]
                if spec["walks_per_family"] is None
                else {**spec["walks"], "k": spec["walks_per_family"][group_id]}
            )
            if spec["straight_p"] > 0:
                paths = TRAWL.straight_walks(
                    matrix,
                    pivot_table,
                    cell_vertices,
                    int(walk_options.get("k", 32)),
                    int(walk_options.get("length", 32)),
                    spec["straight_p"],
                    np.random.default_rng([seed, 2]),
                )
            else:
                paths = sampler(matrix, seed=seed, **walk_options)
            if not len(paths):
                continue
            slots = out[group_id]
            slots[0].append(paths)
            slots[1].append(np.full(len(paths), graph_id))
            slots[2].append(np.full(len(paths), view))
            if use_signs:
                codes = np.zeros_like(paths)
                codes[:, 1:] = sign_table[paths[:, :-1], paths[:, 1:]]
                slots[3].append(codes)
            if use_pivots:
                step_pivots = np.full_like(paths, -1)
                step_pivots[:, 1:] = pivot_table[paths[:, :-1], paths[:, 1:]]
                slots[5].append(step_pivots)
            if spec["vertex_ids"]:
                vertex_info = TRAWL.anonymous_vertex_ids(
                    cell_vertices,
                    paths,
                    np.where(step_pivots >= 0, step_pivots, -1)
                    if use_pivots
                    else None,
                    np.random.default_rng([seed, 1]),
                    spec["max_vertex_ids"],
                    pointers=spec["back_pointers"],
                )
                slots[6].append(vertex_info[0])
                slots[7].append(vertex_info[1])
                if spec["back_pointers"]:
                    slots[8].append(vertex_info[2])
                if spec["vertex_features"]:
                    # Rank-0 states lead each graph's state block.
                    slots[9].append(cell_vertices[paths])
            if spec["running_orientation"]:
                local_ranks = np.repeat(np.arange(spec["num_ranks"]), count_row)
                slots[4].append(
                    TRAWL.running_orientation_codes(
                        paths, local_ranks, codes, spec["num_ranks"] - 1
                    )
                )
    return out


class TRAWL(nn.Module):
    """Encode random walks over lifted cells with configurable sequence layers.

    Consumes the output of ``TRAWLTransform`` and returns contextual cell
    features ``x_r`` plus walk and graph embeddings for ``TRAWLReadout``.
    Input widths left as None are inferred by ``initialize``.

    Parameters
    ----------
    hidden_dim : int, optional
        Width of walk states (default: 128).
    max_rank : int, optional
        Highest cell rank (default: 2).
    in_channels : list of int, optional
        Feature width per rank; inferred if None (default: None).
    pe_dim : int, optional
        Positional encoding width; inferred if None (default: None).
    layers : list, optional
        Explicit ``make_layer`` configs; overrides ``architecture`` and
        ``depth`` (default: None).
    architecture : str, optional
        ``"mamba"``, ``"sisa"``, ``"hybrid"`` (alternating), ``"gru"``,
        ``"transformer"`` or ``"mlp"`` (default: "hybrid").
    depth : int, optional
        Number of preset layers (default: 5).
    layer_options : dict, optional
        Options per layer kind for presets (default: None).
    walks : dict, optional
        ``sample_neighbors`` and guidance options
        (default: ``{"k": 32, "length": 32}``).
    walk_scope : str, optional
        ``"union"`` walks all neighborhoods jointly, ``"separate"`` walks
        each one (default: "union").
    num_neighborhoods : int, optional
        Number of relations built by the transform (default: 2).
    encoder_sharing : str, optional
        ``"shared"`` or ``"independent"`` encoders for separate walks
        (default: "shared").
    fusion : str, optional
        ``NeighborhoodFusion`` mode for separate walks (default: "mean").
    pooling : str, optional
        Time pooling: ``"mean"``, ``"max"`` or ``"mean_max"``
        (default: "mean_max").
    rank_embedding : bool, optional
        Add a learned rank embedding (default: True).
    num_colors : int, optional
        Number of color embeddings; 0 disables them (default: 0).
    move_embedding : bool, optional
        Embed rank moves (down, same, up) along walks (default: True).
    dropout : float, optional
        Dropout on walk inputs (default: 0.0).
    seed : int, optional
        Base seed for walk sampling (default: 42).
    eval_views : int, optional
        Walk views averaged at evaluation (default: 1).
    checkpoint_layers : bool, optional
        Activation checkpointing during training (default: False).
    walk_refresh : str, optional
        ``"train"`` resamples walks every step, ``"fixed"`` never does
        (default: "train").
    occurrence_pooling : str, optional
        Pooling of a cell's walk occurrences: ``"mean"`` or
        ``"attention"`` (default: "mean").
    graph_readout : str, optional
        Graph embedding from ``"walks"`` or ``"cells"`` (default: "walks").
    transition_cache_size : int, optional
        Transition distributions cached by the sampler (default: 1024).
    revisit_embedding : bool, optional
        Embed, per walk step, the anonymous-walk ID (rank of the cell's first
        occurrence in its walk) and the log2-bucketed number of steps since
        the walk last visited the cell (default: False).
    max_anonymous_ids : int, optional
        Anonymous-walk IDs at or above this are clipped (default: 32).
    orientation_embedding : bool, optional
        Embed the signed boundary coefficient (+1/-1) of the incidence
        crossed by each step; needs ``TRAWLTransform(signs=True)``
        (default: False).
    pe_skip : bool, optional
        Add an MLP of per-graph, per-rank pooled encodings (mean, std, min,
        max of ``trawl_pe`` plus the log cell count) to the graph embedding
        (default: False).
    pe_norm : bool, optional
        Batch-normalize ``trawl_pe`` before its projection (default: False).
    running_orientation : bool, optional
        Embed, at each top-rank (triangle) step, the walk's running relative
        orientation: the product of the local relative-orientation signs of
        the triangle-to-triangle moves since the walk's reference triangle
        (``-s(e, T) * s(e, T')`` across a shared face, or the signed
        down-adjacency edge). Passing through a lower-rank cell other than a
        shared face starts a new reference, marked separately. Needs
        ``TRAWLTransform(signs=True)`` (default: False).
    pivot_embedding : bool, optional
        Add, at every adjacency step, a projection of the input state of the
        cell the step pivots across (shared face or coface); needs
        ``TRAWLTransform(pivots=True)`` (default: False).
    cell_memory : bool, optional
        Two-pass cross-rank memory. After the first encoding pass, every
        token writes its state to a slot keyed by the cell it visits and by
        the pivot cell it crossed (scatter-mean over all walks of every
        family, O(tokens)); a second encoder reads both slots back into the
        tokens. Needs ``TRAWLTransform(pivots=True)`` (default: False).
    walk_attention : bool, optional
        Self-attention over all walk summaries of a graph, across walk
        families, before pooling (O(walks^2) per graph) (default: False).
    vertex_ids : bool, optional
        Add, per step, embeddings of walk-relative anonymous ids of the
        current cell's vertices (order of first appearance in the walk;
        vertices first met together are ordered randomly per walk, never by
        label) and, separately, of those vertices that belong to the cell
        crossed by the step (the pivot face). Unsigned structure only;
        needs ``TRAWLTransform(cell_vertices=True)`` and, for the crossed-face
        part, ``pivots=True`` (default: False).
    walks_per_family : list of int, optional
        With ``walk_scope="separate"``, the number of walks per family,
        overriding ``walks.k`` (default: None).
    back_pointers : bool, optional
        After the encoder pass, every token reads the states of earlier
        steps of its own walk that touched the same lower-rank cells: for
        each of its vertices, the last earlier step containing that vertex,
        and the last earlier visit of its own cell. Each pointed-to state is
        combined with an embedding of the step offset and pointer slot, an
        MLP, summed, and followed by one causal GRU pass. Cost
        O(tokens x pointers); needs ``vertex_ids`` (default: False).
    straight_p : float, optional
        If positive, sample walks with a straight (zig-zag) bias instead of
        the default sampler: from the second crossing on, with probability
        ``straight_p`` the walk crosses to the neighbor whose shared cell
        (pivot) avoids the vertex shared by the last two crossed pivots, so
        consecutive turns alternate and the walk runs straight across the
        complex; otherwise it picks a uniform non-backtracking neighbor.
        Starts are uniform over states with neighbors. Uses only unsigned
        local structure; needs ``TRAWLTransform(cell_vertices=True,
        pivots=True)`` (default: 0.0, off).
    use_pe : bool, optional
        If False, the positional/structural encodings ``trawl_pe`` are left
        out of the walk tokens (an ablation; the projection still exists)
        (default: True).
    max_cell_vertices : int, optional
        Largest number of vertices of a cell, i.e. the column width of
        ``trawl_cell_vertices``; sizes the back-pointer slot embedding (one
        slot per vertex column plus one for the cell). None means
        ``max_rank + 1`` (simplices); cell complexes from a cycle lifting
        need ``max_cell_length`` (default: None).
    vertex_features : bool, optional
        With ``vertex_ids``, bind every vertex of a step's cell to its own
        input features: per vertex, the sum of its anonymous-id embedding,
        crossed-face embedding and a projection of its rank-0 input features
        goes through a small MLP, and the results are summed into the token.
        Back-pointers through a shared vertex then also carry what the
        vertex is (e.g. its atom type) (default: False).
    fallback_family : int, optional
        With ``walk_scope="separate"``, the index of a transform relation
        that is not walked as its own family but replaces, per graph, every
        other family whose relation has no edges in that graph (e.g. edges
        across shared vertices for graphs without adjacent 2-cells). None
        walks every relation as its own family (default: None).
    sampling_workers : int, optional
        Host walk sampling (walks, anonymous ids, pointers) runs graph by
        graph in this many forked worker processes; 0 samples in the main
        process. Graphs are seeded independently, so the walks are the same
        either way (default: 0).
    gate_entropy : float, optional
        With a learned family gate, add ``-gate_entropy`` times the mean
        per-graph entropy of the gate weights to the training loss (an
        entropy bonus against gate collapse), returned as ``aux_loss``
        (default: 0.0, off).
    family_dropout : float, optional
        With a learned family gate, during training drop each available
        family of each graph with this probability before the softmax (the
        gate renormalizes over the rest; a graph keeps at least one family)
        (default: 0.0, off).
    eval_chunk_walks : int, optional
        At evaluation, encode walks in chunks of at most this many walks to
        bound memory (walks are encoded independently); None encodes all
        walks of a family at once (default: None).
    """

    def __init__(
        self,
        hidden_dim=128,
        max_rank=2,
        in_channels=None,
        pe_dim=None,
        layers=None,
        architecture="hybrid",
        depth=5,
        layer_options=None,
        walks=None,
        walk_scope="union",
        num_neighborhoods=2,
        encoder_sharing="shared",
        fusion="mean",
        pooling="mean_max",
        rank_embedding=True,
        num_colors=0,
        move_embedding=True,
        dropout=0.0,
        seed=42,
        eval_views=1,
        checkpoint_layers=False,
        walk_refresh="train",
        occurrence_pooling="mean",
        graph_readout="walks",
        transition_cache_size=1024,
        revisit_embedding=False,
        max_anonymous_ids=32,
        orientation_embedding=False,
        pe_skip=False,
        pe_norm=False,
        running_orientation=False,
        pivot_embedding=False,
        cell_memory=False,
        walk_attention=False,
        vertex_ids=False,
        walks_per_family=None,
        use_pe=True,
        back_pointers=False,
        straight_p=0.0,
        max_cell_vertices=None,
        vertex_features=False,
        fallback_family=None,
        eval_chunk_walks=None,
        gate_entropy=0.0,
        family_dropout=0.0,
        sampling_workers=0,
    ):
        super().__init__()
        self.sampling_workers = int(sampling_workers)
        self.gate_entropy = float(gate_entropy)
        self.family_dropout = float(family_dropout)
        if (self.gate_entropy or self.family_dropout) and (
            fusion not in NeighborhoodFusion.GATED or walk_scope != "separate"
        ):
            raise ValueError(
                "gate_entropy/family_dropout need a learned family gate"
            )
        self.eval_chunk_walks = (
            None if eval_chunk_walks is None else int(eval_chunk_walks)
        )
        choices = {
            "walk_scope": (walk_scope, {"union", "separate"}),
            "encoder_sharing": (encoder_sharing, {"shared", "independent"}),
            "pooling": (pooling, {"mean", "max", "mean_max"}),
            "walk_refresh": (walk_refresh, {"train", "fixed"}),
            "occurrence_pooling": (occurrence_pooling, {"mean", "attention"}),
            "graph_readout": (graph_readout, {"walks", "cells"}),
        }
        for name, (value, allowed) in choices.items():
            if value not in allowed:
                raise ValueError(f"{name} must be one of {sorted(allowed)}")
        if eval_views < 1 or depth < 1 or num_neighborhoods < 1:
            raise ValueError(
                "Depth, neighborhood count and eval_views must be positive"
            )
        self.hidden_dim, self.max_rank = hidden_dim, max_rank
        self.walk_sampler = WalkSampler(transition_cache_size)
        self.walk_scope, self.pooling = walk_scope, pooling
        self.num_neighborhoods = num_neighborhoods
        self.walks = dict(walks or {"k": 32, "length": 32})
        self.seed, self.eval_views = seed, eval_views
        self.checkpoint_layers = checkpoint_layers
        self.walk_refresh = walk_refresh
        self.occurrence_pooling = occurrence_pooling
        self.graph_readout = graph_readout
        self.occurrence_attention = (
            nn.Linear(hidden_dim, 1)
            if occurrence_pooling == "attention"
            else None
        )
        self.register_buffer(
            "sampling_step", torch.zeros((), dtype=torch.long)
        )
        # Host mirror of ``sampling_step``; reading the buffer would wait for
        # the device on every forward pass.
        self._host_sampling_step = None
        self.features = nn.ModuleList(
            [
                nn.Linear(in_channels[r], hidden_dim)
                if in_channels is not None
                else nn.LazyLinear(hidden_dim)
                for r in range(max_rank + 1)
            ]
        )
        self.position = (
            nn.LazyLinear(hidden_dim, bias=False)
            if pe_dim is None
            else nn.Linear(pe_dim, hidden_dim, bias=False)
        )
        self.rank = (
            nn.Embedding(max_rank + 1, hidden_dim) if rank_embedding else None
        )
        self.color = (
            nn.Embedding(num_colors, hidden_dim) if num_colors else None
        )
        self.move = nn.Embedding(3, hidden_dim) if move_embedding else None
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        options = dict(layer_options or {})
        if layers is None:
            presets = {"mamba", "sisa", "hybrid", "gru", "transformer", "mlp"}
            if architecture not in presets:
                raise ValueError(
                    "Custom architecture requires an explicit layer list"
                )
            kinds = [
                ("mamba" if i % 2 == 0 else "sisa")
                if architecture == "hybrid"
                else architecture
                for i in range(depth)
            ]
            layers = [
                {"kind": kind, **dict(options.get(kind, {}))} for kind in kinds
            ]
        if not layers:
            raise ValueError("Provide at least one sequence layer")
        encoder = nn.ModuleList(
            [make_layer(config, hidden_dim) for config in layers]
        )
        count = (
            num_neighborhoods
            if walk_scope == "separate" and encoder_sharing == "independent"
            else 1
        )
        self.encoders = nn.ModuleList(
            [encoder] + [copy.deepcopy(encoder) for _ in range(count - 1)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * (2 if pooling == "mean_max" else 1)
        self.fallback_family = (
            None if fallback_family is None else int(fallback_family)
        )
        if self.fallback_family is not None and (
            walk_scope != "separate"
            or not 0 <= self.fallback_family < num_neighborhoods
            or num_neighborhoods < 2
        ):
            raise ValueError(
                "fallback_family needs walk_scope=separate and an index of "
                "one of at least two relations"
            )
        self.fusion = NeighborhoodFusion(
            self.output_dim,
            num_neighborhoods - (self.fallback_family is not None),
            fusion,
        )
        # Optional modules are created only when enabled, so default models
        # draw exactly the same initial parameters as before.
        self.revisit_embedding = bool(revisit_embedding)
        self.orientation_embedding = bool(orientation_embedding)
        self.pe_skip, self.pe_norm = bool(pe_skip), bool(pe_norm)
        if self.revisit_embedding:
            self.max_anonymous_ids = int(max_anonymous_ids)
            self.anonymous = nn.Embedding(self.max_anonymous_ids, hidden_dim)
            # 0: first visit; b >= 1: gap in [2 ** (b - 1), 2 ** b).
            self.revisit_gap = nn.Embedding(self.REVISIT_BUCKETS, hidden_dim)
        if self.orientation_embedding:
            # 0: no incidence crossed (or first step), 1: +1, 2: -1.
            self.orientation = nn.Embedding(3, hidden_dim)
        self.running_orientation = bool(running_orientation)
        if self.running_orientation:
            # 0: not a top-rank step, 1: same, 2: flipped, 3: new reference.
            self.running = nn.Embedding(4, hidden_dim)
        self.pivot_embedding = bool(pivot_embedding)
        self.cell_memory = bool(cell_memory)
        self.walk_attention = bool(walk_attention)
        if self.pivot_embedding:
            self.pivot_projection = nn.Linear(hidden_dim, hidden_dim)
        if self.cell_memory:
            self.memory_self = nn.Linear(hidden_dim, hidden_dim)
            self.memory_pivot = nn.Linear(hidden_dim, hidden_dim)
            self.memory_encoders = copy.deepcopy(self.encoders)
            for module in self.memory_encoders.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        self.vertex_ids = bool(vertex_ids)
        self.use_pe = bool(use_pe)
        self.straight_p = float(straight_p)
        if not 0.0 <= self.straight_p <= 1.0:
            raise ValueError("straight_p must be in [0, 1]")
        self.back_pointers = bool(back_pointers)
        if self.back_pointers:
            if not vertex_ids or cell_memory:
                raise ValueError(
                    "back_pointers needs vertex_ids and excludes cell_memory"
                )
            # One slot per vertex column of a cell, plus one for the cell.
            self.max_cell_vertices = (
                max_rank + 1
                if max_cell_vertices is None
                else int(max_cell_vertices)
            )
            slots = self.max_cell_vertices + 1
            self.pointer_offset = nn.Embedding(self.REVISIT_BUCKETS, hidden_dim)
            self.pointer_slot = nn.Embedding(slots, hidden_dim)
            self.pointer_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.pointer_layer = make_layer({"kind": "gru"}, hidden_dim)
        self.walks_per_family = (
            None if walks_per_family is None else [int(k) for k in walks_per_family]
        )
        if self.walks_per_family is not None and (
            walk_scope != "separate"
            or len(self.walks_per_family) != num_neighborhoods
        ):
            raise ValueError(
                "walks_per_family needs walk_scope=separate and one count per family"
            )
        if self.vertex_ids:
            self.max_vertex_ids = int(max_anonymous_ids)
            self.vertex_anonymous = nn.Embedding(
                self.max_vertex_ids, hidden_dim
            )
            self.vertex_crossed = nn.Embedding(self.max_vertex_ids, hidden_dim)
        if self.walk_attention:
            self.family = nn.Embedding(num_neighborhoods, self.output_dim)
            self.walk_mixer = nn.TransformerEncoderLayer(
                d_model=self.output_dim,
                nhead=4,
                dim_feedforward=2 * self.output_dim,
                dropout=0.0,
                batch_first=True,
                norm_first=True,
            )
        self.vertex_features = bool(vertex_features)
        if self.vertex_features:
            if not self.vertex_ids:
                raise ValueError("vertex_features needs vertex_ids")
            self.vertex_feature = nn.Linear(hidden_dim, hidden_dim)
            self.vertex_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        if (self.pe_skip or self.pe_norm) and pe_dim is not None:
            self._build_pe_modules(pe_dim)

    REVISIT_BUCKETS = 12

    def compile_layers(self, dynamic=True):
        """Compile every sequence layer, with and without checkpointing.

        Compiling the layer modules themselves fails when they are called
        inside ``torch.utils.checkpoint``; compiling the checkpointed call as
        one function works and keeps the math (and dropout RNG replay)
        unchanged. Walk sampling stays eager on the host.

        Parameters
        ----------
        dynamic : bool, optional
            Mark shapes dynamic so varying walk counts do not recompile
            (default: True).
        """
        from torch.utils.checkpoint import checkpoint

        if getattr(self, "_compiled_calls", None) is not None:
            return
        self._compiled_calls = {}
        for stack in list(self.encoders) + list(
            getattr(self, "memory_encoders", [])
        ):
            for layer in stack:
                self._compiled_calls[id(layer)] = (
                    torch.compile(
                        lambda x, layer=layer: checkpoint(
                            layer, x, use_reentrant=False
                        ),
                        dynamic=dynamic,
                    ),
                    torch.compile(layer, dynamic=dynamic),
                )

    def _build_pe_modules(self, pe_dim):
        """Create the encoding normalization and skip-readout modules.

        Parameters
        ----------
        pe_dim : int
            Width of ``trawl_pe``.
        """
        if self.pe_norm:
            self.pe_batch_norm = SafeBatchNorm1d(pe_dim)
        if self.pe_skip:
            width = (self.max_rank + 1) * (4 * pe_dim + 1)
            self.pe_skip_net = nn.Sequential(
                SafeBatchNorm1d(width),
                nn.Linear(width, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.output_dim),
            )

    def initialize(self, data_list):
        """Materialize projections from training data, including absent ranks.

        Parameters
        ----------
        data_list : list of torch_geometric.data.Data
            Transformed training graphs used to infer input widths.
        """
        if not data_list:
            raise ValueError(
                "Cannot infer TRAWL input widths from an empty dataset"
            )
        with torch.no_grad():
            for rank, projection in enumerate(self.features):
                if not isinstance(projection, nn.LazyLinear):
                    continue
                key = f"trawl_signal_{rank}"
                widths = {
                    data[key].shape[1] for data in data_list if len(data[key])
                }
                if len(widths) > 1:
                    raise ValueError(
                        f"Inconsistent rank-{rank} feature widths: {widths}"
                    )
                width = next(iter(widths), data_list[0][key].shape[1])
                projection(
                    torch.zeros(1, width, device=projection.weight.device)
                )
            if (self.pe_skip or self.pe_norm) and not (
                hasattr(self, "pe_batch_norm") or hasattr(self, "pe_skip_net")
            ):
                device = self.sampling_step.device
                self._build_pe_modules(data_list[0].trawl_pe.shape[1])
                self.to(device)
            if isinstance(self.position, nn.LazyLinear):
                self.position(
                    torch.zeros(
                        1,
                        data_list[0].trawl_pe.shape[1],
                        device=self.position.weight.device,
                    )
                )

    # Walk metadata read on the host (see ``TBModel.on_before_batch_transfer``).
    host_fields = (
        "trawl_counts",
        "trawl_edges",
        "trawl_weights",
        "trawl_identity",
        "trawl_relations",
        "trawl_colors",
        "trawl_signs",
        "trawl_pivots",
        "trawl_cell_vertices",
    )

    def _host(self, batch):
        """Return CPU copies of ``host_fields`` without waiting for the device.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch, optionally carrying precomputed ``trawl_host`` arrays.

        Returns
        -------
        dict
            Mapping from field name to NumPy array.
        """
        host = getattr(batch, "trawl_host", None)
        if host is None:
            host = {
                key: batch[key].detach().cpu().numpy()
                for key in self.host_fields
                if key in batch
            }
        return host

    def next_sampling_step(self):
        """Return the training sampling step and advance it.

        Returns
        -------
        int
            Sampling step before the increment.
        """
        if self._host_sampling_step is None:
            self._host_sampling_step = int(self.sampling_step)
        step = self._host_sampling_step
        self._host_sampling_step += 1
        self.sampling_step.add_(1)
        return step

    def reset_sampling_step(self):
        """Restart walk refreshes, e.g. before supervised fitting."""
        self.sampling_step.zero_()
        self._host_sampling_step = 0

    def _load_from_state_dict(self, *args, **kwargs):
        """Load state and invalidate the host sampling-step mirror.

        Parameters
        ----------
        *args : tuple
            Positional arguments forwarded to ``nn.Module``.
        **kwargs : dict
            Keyword arguments forwarded to ``nn.Module``.
        """
        # A restored buffer invalidates the host mirror.
        self._host_sampling_step = None
        super()._load_from_state_dict(*args, **kwargs)

    def _pool(self, x):
        """Pool walk states over time.

        Parameters
        ----------
        x : torch.Tensor
            States of shape ``[walk, time, hidden]``.

        Returns
        -------
        torch.Tensor
            Pooled embeddings of shape ``[walk, output_dim]``.
        """
        if self.pooling == "mean":
            return x.mean(dim=1)
        if self.pooling == "max":
            return x.amax(dim=1)
        return torch.cat((x.amax(dim=1), x.mean(dim=1)), dim=-1)

    def _revisit_codes(self, paths):
        """Anonymous-walk IDs and revisit-gap buckets of every walk step.

        Parameters
        ----------
        paths : torch.Tensor
            Walk cell indices of shape ``[walk, time]``.

        Returns
        -------
        tuple of torch.Tensor
            ``(anonymous, gap)``, both ``[walk, time]``: the order of each
            cell's first appearance in its own walk, and 0 for a first visit
            or ``1 + floor(log2(steps since the last visit))``.
        """
        length = paths.shape[1]
        time = torch.arange(length, device=paths.device)
        same = paths.unsqueeze(2) == paths.unsqueeze(1)
        # First occurrence: smallest matching time; previous: largest earlier.
        first = torch.where(same, time, length).amin(dim=2)
        earlier = same & (time.view(1, 1, -1) < time.view(1, -1, 1))
        previous = torch.where(earlier, time, -1).amax(dim=2)
        new = first == time
        anonymous = (new.long().cumsum(dim=1) - 1).gather(1, first)
        anonymous = anonymous.clamp_max(self.max_anonymous_ids - 1)
        gap = (time - previous).clamp_min(1).float()
        gap = torch.where(
            previous >= 0,
            1 + torch.floor(torch.log2(gap)).long(),
            torch.zeros_like(previous),
        ).clamp_max(self.REVISIT_BUCKETS - 1)
        return anonymous, gap

    @staticmethod
    def running_orientation_codes(paths, ranks, codes, top):
        """Running relative orientation of top-rank cells along walks.

        Parameters
        ----------
        paths : np.ndarray
            Graph-local walk states ``[walk, time]``.
        ranks : np.ndarray
            Rank of each graph-local state.
        codes : np.ndarray
            Sign code of each step (0 none, 1 for +1, 2 for -1),
            ``[walk, time]``.
        top : int
            Rank whose cells carry the orientation (2 for surfaces).

        Returns
        -------
        np.ndarray
            ``[walk, time]`` codes: 0 not a top-rank step, 1 same orientation
            as the reference, 2 flipped, 3 new reference.
        """
        sign = np.where(codes == 1, 1, np.where(codes == 2, -1, 0))
        rank = ranks[paths]
        out = np.zeros_like(paths)
        state = np.zeros(len(paths), dtype=np.int64)
        for t in range(paths.shape[1]):
            here = rank[:, t] == top
            if t >= 1:
                direct = here & (rank[:, t - 1] == top)
                factor = sign[:, t]
            else:
                direct = np.zeros_like(here)
                factor = np.zeros(len(paths), dtype=np.int64)
            if t >= 2:
                across = (
                    here & (rank[:, t - 1] == top - 1) & (rank[:, t - 2] == top)
                )
                factor = np.where(
                    across, -sign[:, t - 1] * sign[:, t], factor
                )
            else:
                across = np.zeros_like(here)
            chained = (direct | across) & (state != 0) & (factor != 0)
            fresh = here & ~chained
            state = np.where(chained, state * factor, state)
            state = np.where(fresh, 1, state)
            out[:, t] = np.where(
                fresh, 3, np.where(chained, np.where(state > 0, 1, 2), 0)
            )
        return out

    @staticmethod
    def straight_walks(matrix, pivot_table, cell_vertices, k, length, p, rng):
        """Sample straight-biased non-backtracking walks.

        Parameters
        ----------
        matrix : scipy.sparse.spmatrix
            Graph-local relation matrix (states x states).
        pivot_table : np.ndarray
            ``[states, states]`` graph-local pivot state per edge, -1 none.
        cell_vertices : np.ndarray
            ``[states, R]`` graph-local vertices per state, -1 padded.
        k : int
            Number of walks.
        length : int
            States per walk.
        p : float
            Probability of the straight move when it is defined.
        rng : np.random.Generator
            Source of all random draws.

        Returns
        -------
        np.ndarray
            ``[k, length]`` graph-local walk states.
        """
        csr = matrix.tocsr()
        degree = np.diff(csr.indptr)
        active = np.flatnonzero(degree > 0)
        if not len(active):
            return np.empty((0, length), dtype=np.int64)
        width = int(degree.max())
        rows = np.repeat(np.arange(csr.shape[0]), degree)
        slots = np.arange(csr.nnz) - csr.indptr[rows]
        neighbors = np.full((csr.shape[0], width), -1, dtype=np.int64)
        neighbors[rows, slots] = csr.indices
        faces = np.full((csr.shape[0], width), -1, dtype=np.int64)
        faces[rows, slots] = pivot_table[rows, csr.indices]
        face_vertices = np.where(
            (faces >= 0)[..., None], cell_vertices[np.maximum(faces, 0)], -1
        )  # [states, width, R]
        walks = np.arange(k)
        current = active[rng.integers(len(active), size=k)]
        previous = np.full(k, -1, dtype=np.int64)
        last_face = np.full(k, -1, dtype=np.int64)
        before_face = np.full(k, -1, dtype=np.int64)
        out = np.empty((k, length), dtype=np.int64)
        out[:, 0] = current
        for t in range(1, length):
            candidates = neighbors[current]
            valid = (candidates >= 0) & (candidates != previous[:, None])
            stuck = ~valid.any(axis=1)
            valid = np.where(stuck[:, None], candidates >= 0, valid)
            # Pivot vertex: the single vertex shared by the last two pivots.
            a = np.where((last_face >= 0)[:, None], cell_vertices[np.maximum(last_face, 0)], -1)
            b = np.where((before_face >= 0)[:, None], cell_vertices[np.maximum(before_face, 0)], -2)
            shared = (a[:, :, None] == b[:, None, :]) & (a[:, :, None] >= 0)
            count = shared.any(axis=2).sum(axis=1)
            vertex = np.where(
                count == 1,
                np.take_along_axis(a, shared.any(axis=2).argmax(axis=1)[:, None], 1)[:, 0],
                -1,
            )
            avoids = ~(face_vertices[current] == vertex[:, None, None]).any(axis=2)
            straight = valid & avoids & (vertex >= 0)[:, None]
            use = (rng.random(k) < p) & straight.any(axis=1)
            pool = np.where(use[:, None], straight, valid)
            choice = np.where(pool, rng.random(pool.shape), -1.0).argmax(axis=1)
            following = candidates[walks, choice]
            before_face = last_face
            last_face = faces[current, choice]
            previous = current
            current = following
            out[:, t] = current
        return out

    @staticmethod
    def anonymous_vertex_ids(
        cell_vertices, paths, pivots, rng, cap, pointers=False
    ):
        """Walk-relative anonymous vertex ids of every step.

        Parameters
        ----------
        cell_vertices : np.ndarray
            ``[states, R]`` graph-local vertex indices per state, -1 padded.
        paths : np.ndarray
            Graph-local walk states ``[walk, time]``.
        pivots : np.ndarray or None
            Graph-local pivot state per step, -1 for none.
        rng : np.random.Generator
            Orders vertices met at the same step.
        cap : int
            Ids at or above ``cap - 1`` are clipped.
        pointers : bool, optional
            Also return, per step and vertex column, the last earlier step
            of the walk whose cell contains that vertex (-1 if none)
            (default: False).

        Returns
        -------
        tuple of np.ndarray
            ``ids`` ``[walk, time, R]`` (-1 for padding) and ``crossed``
            ``[walk, time, R]`` booleans marking vertices of the pivot cell,
            both aligned with the columns of ``cell_vertices``.
        """
        walks, length = paths.shape
        verts = cell_vertices[paths]  # [W, L, R]
        width = verts.shape[2]
        order = np.argsort(rng.random(verts.shape), axis=2)
        verts = np.take_along_axis(verts, order, axis=2)
        flat = verts.reshape(walks, length * width)
        valid = flat >= 0
        size = length * width
        position = np.arange(size)
        # One stable sort by (walk, vertex) groups every vertex's
        # occurrences in walk order, so the first occurrence and the
        # previous one are a scan away: O(P log P) instead of the [W, P, P]
        # comparison. Padding (-1) forms its own group and is masked below.
        span = int(flat.max()) + 2 if flat.size else 1
        key = (np.arange(walks)[:, None] * span + flat + 1).ravel()
        sort = np.argsort(key, kind="stable")
        sorted_key = key[sort]
        head = np.ones(len(sort), dtype=bool)
        head[1:] = sorted_key[1:] != sorted_key[:-1]
        sorted_pos = sort % size if size else sort
        group_head = np.maximum.accumulate(
            np.where(head, np.arange(len(sort)), 0)
        )
        first = np.empty(len(sort), dtype=np.int64)
        first[sort] = sorted_pos[group_head]
        first = first.reshape(walks, size)
        new = valid & (first == position)
        rank = np.cumsum(new, axis=1) - 1
        ids = np.take_along_axis(rank, first, axis=1)
        ids = np.where(valid, np.minimum(ids, cap - 1), -1)
        ids = ids.reshape(walks, length, width)
        crossed = np.zeros(ids.shape, dtype=bool)
        if pivots is not None:
            face = np.where(
                (pivots >= 0)[..., None],
                cell_vertices[np.maximum(pivots, 0)],
                -1,
            )  # [W, L, R]
            crossed = (verts[..., :, None] == face[..., None, :]).any(-1) & (
                verts >= 0
            )
        # Back to the cell's column order (ids themselves stay random).
        inverse = np.argsort(order, axis=2)
        ids = np.take_along_axis(ids, inverse, axis=2)
        crossed = np.take_along_axis(crossed, inverse, axis=2)
        if not pointers:
            return ids, crossed
        # The previous occurrence of a vertex in its sorted group is the
        # last earlier step containing it (a cell lists a vertex once).
        sorted_prev = np.full(len(sort), -1, dtype=np.int64)
        sorted_prev[1:] = np.where(head[1:], -1, sorted_pos[:-1] // width)
        previous = np.empty(len(sort), dtype=np.int64)
        previous[sort] = sorted_prev
        previous = previous.reshape(walks, size)
        previous = np.where(valid, previous, -1).reshape(walks, length, width)
        previous = np.take_along_axis(previous, inverse, axis=2)
        return ids, crossed, previous

    def _vertex_tokens(self, ids, crossed, vertex_states=None, features=None):
        """Sum of anonymous vertex-id embeddings per token.

        Parameters
        ----------
        ids : torch.Tensor
            ``[walk, time, R]`` anonymous ids, -1 for padding.
        crossed : torch.Tensor
            ``[walk, time, R]`` crossed-face membership.
        vertex_states : torch.Tensor, optional
            ``[walk, time, R]`` global state index of each vertex, -1 for
            padding; used by ``vertex_features`` (default: None).
        features : torch.Tensor, optional
            Projected input features of every state ``[states, hidden]``;
            used by ``vertex_features`` (default: None).

        Returns
        -------
        torch.Tensor
            ``[walk, time, hidden]``.
        """
        if ids.shape[2] <= self.max_rank + 1 and not self.vertex_features:
            # Simplicial width: the original dense computation.
            valid = (ids >= 0).unsqueeze(-1)
            safe = ids.clamp_min(0)
            own = (self.vertex_anonymous(safe) * valid).sum(dim=2)
            face = (self.vertex_crossed(safe) * crossed.unsqueeze(-1)).sum(
                dim=2
            )
            return own + face
        # Wide cells (cell complexes) are mostly padding: embed only the
        # valid vertex entries and sum them per token, so memory follows the
        # number of real vertices, not walks x time x width x hidden.
        walk, time, column = torch.nonzero(ids >= 0, as_tuple=True)
        entry = ids[walk, time, column]
        value = self.vertex_anonymous(entry) + self.vertex_crossed(
            entry
        ) * crossed[walk, time, column].unsqueeze(-1)
        if self.vertex_features:
            # Bind each vertex's id to what it is.
            value = value + self.vertex_mlp(
                value
                + self.vertex_feature(
                    features[vertex_states[walk, time, column]]
                )
            )
        slot = walk * ids.shape[1] + time
        out = value.new_zeros(ids.shape[0] * ids.shape[1], value.shape[-1])
        return out.index_add(0, slot, value).view(*ids.shape[:2], -1)

    def _memory_pass(self, x, ranks, prepared, num_states, extras=None):
        """Second encoding pass reading a per-cell memory of the first.

        Every first-pass token writes its state to the slot of the cell it
        visits and of the pivot cell its step crossed; slots average over all
        walks of every family. Each token of the second pass reads its cell's
        and its pivot's slot. Cost is linear in the number of tokens.

        Parameters
        ----------
        x : torch.Tensor
            State inputs ``[states, hidden]``.
        ranks : torch.Tensor
            Rank of each state.
        prepared : list
            Per group ``(paths, owners, views, codes, running, pivots,
            encoded)`` or None.
        num_states : int
            Number of states in the batch.
        extras : list, optional
            Per group additional token inputs reused in the second pass
            (default: None).

        Returns
        -------
        list
            ``prepared`` with second-pass encodings.
        """
        memory = x.new_zeros(num_states, x.shape[1], dtype=torch.float32)
        count = memory.new_zeros(num_states, 1)
        for item in prepared:
            if item is None:
                continue
            paths, _, _, _, _, pivots, encoded = item
            states = encoded.flatten(0, 1).float()
            ones = count.new_ones(len(states), 1)
            memory.index_add_(0, paths.flatten(), states)
            count.index_add_(0, paths.flatten(), ones)
            flat = pivots.flatten()
            has = flat >= 0
            memory.index_add_(0, flat[has], states[has])
            count.index_add_(0, flat[has], ones[has])
        memory = memory / count.clamp_min(1)
        out = []
        for group_id, item in enumerate(prepared):
            if item is None:
                out.append(None)
                continue
            paths, owners, view_ids, codes, running, pivots, _ = item
            has = (pivots >= 0).unsqueeze(-1)
            extra = self.memory_self(memory[paths].to(x.dtype)) + has * (
                self.memory_pivot(memory[pivots.clamp_min(0)].to(x.dtype))
            )
            if extras is not None and extras[group_id] is not None:
                extra = extra + extras[group_id]
            encoded = self._encode(
                x,
                paths,
                ranks,
                group_id,
                codes,
                running,
                pivots,
                extra=extra,
                encoders=self.memory_encoders,
            )
            out.append(
                (paths, owners, view_ids, codes, running, pivots, encoded)
            )
        return out

    def _attend_walks(self, walks, owners, families, num_graphs, fallback):
        """Mix all walk summaries of each graph with self-attention.

        Parameters
        ----------
        walks : torch.Tensor
            Pooled walk embeddings ``[walks, output_dim]``.
        owners : torch.Tensor
            Graph of each walk.
        families : torch.Tensor
            Walk family (neighborhood group) of each walk.
        num_graphs : int
            Number of graphs.
        fallback : torch.Tensor
            Graph embeddings used for graphs without walks.

        Returns
        -------
        torch.Tensor
            Graph embeddings ``[graphs, output_dim]``.
        """
        values = walks + self.family(families).to(walks.dtype)
        order = torch.argsort(owners, stable=True)
        values, owners = values[order], owners[order]
        counts = torch.bincount(owners, minlength=num_graphs)
        width = int(counts.max())
        first = torch.cumsum(counts, 0) - counts
        slot = torch.arange(len(owners), device=owners.device) - first[owners]
        padded = values.new_zeros(num_graphs, width, values.shape[1])
        padded[owners, slot] = values
        pad = (
            torch.arange(width, device=owners.device)[None] >= counts[:, None]
        )
        pad[:, 0] &= counts > 0  # keep one slot open for empty graphs
        mixed = self.walk_mixer(padded, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1) & (counts > 0)[:, None, None]
        pooled = (mixed * keep).sum(dim=1) / counts.clamp_min(1)[:, None]
        return torch.where(
            (counts > 0)[:, None], pooled.to(fallback.dtype), fallback
        )

    def _encode(
        self,
        x,
        paths,
        ranks,
        relation,
        signs=None,
        running=None,
        pivots=None,
        extra=None,
        encoders=None,
        pointers=None,
    ):
        """Encode sampled walks with the sequence layers.

        Parameters
        ----------
        x : torch.Tensor
            Cell states of shape ``[cells, hidden]``.
        paths : torch.Tensor
            Walk cell indices of shape ``[walk, time]``.
        ranks : torch.Tensor
            Rank of each cell, shape ``[cells]``.
        relation : int
            Neighborhood group index selecting the encoder.
        signs : torch.Tensor, optional
            Orientation code per step (0 none, 1 for +1, 2 for -1), shape
            ``[walk, time]``; used by ``orientation_embedding``
            (default: None).
        running : torch.Tensor, optional
            Running-orientation code per step (default: None).
        pivots : torch.Tensor, optional
            Global pivot state per step, -1 for none (default: None).
        extra : torch.Tensor, optional
            Additional token input ``[walk, time, hidden]`` (default: None).
        encoders : nn.ModuleList, optional
            Encoder stacks replacing ``self.encoders`` (default: None).
        pointers : torch.Tensor, optional
            ``[walk, time, R]`` last earlier step touching each vertex of
            the step's cell, -1 for none; used by ``back_pointers``
            (default: None).

        Returns
        -------
        torch.Tensor
            Encoded walk states of shape ``[walk, time, hidden]``.
        """
        tokens = x[paths]
        if self.move is not None:
            path_ranks = ranks[paths]
            movement = torch.zeros_like(path_ranks)
            movement[:, 1:] = torch.sign(
                path_ranks[:, 1:] - path_ranks[:, :-1]
            )
            tokens = tokens + self.move(movement + 1)
        if self.revisit_embedding:
            anonymous, gap = self._revisit_codes(paths)
            tokens = tokens + self.anonymous(anonymous) + self.revisit_gap(gap)
        if self.orientation_embedding:
            tokens = tokens + self.orientation(signs)
        if self.running_orientation:
            tokens = tokens + self.running(running)
        if self.pivot_embedding:
            has = (pivots >= 0).unsqueeze(-1)
            tokens = tokens + has * self.pivot_projection(
                x[pivots.clamp_min(0)]
            )
        if extra is not None:
            tokens = tokens + extra
        tokens = self.dropout(self.input_norm(tokens))
        stacks = self.encoders if encoders is None else encoders
        encoder = stacks[0 if len(stacks) == 1 else relation]
        compiled = getattr(self, "_compiled_calls", None)
        for layer in encoder:
            if self.checkpoint_layers and self.training:
                if compiled is not None:
                    # Compiled "checkpoint(layer, x)": dynamo handles the
                    # checkpoint as one higher-order op (RNG state included).
                    tokens = compiled[id(layer)][0](tokens)
                else:
                    from torch.utils.checkpoint import checkpoint

                    tokens = checkpoint(layer, tokens, use_reentrant=False)
            elif compiled is not None:
                tokens = compiled[id(layer)][1](tokens)
            else:
                tokens = layer(tokens)
        if self.back_pointers:
            if (
                self.checkpoint_layers
                and self.training
                and pointers.shape[2] > self.max_rank + 1
            ):
                # Wide cells: also checkpoint the pointer pass (memory only).
                from torch.utils.checkpoint import checkpoint

                tokens = checkpoint(
                    self._follow_pointers,
                    tokens,
                    paths,
                    pointers,
                    use_reentrant=False,
                )
            else:
                tokens = self._follow_pointers(tokens, paths, pointers)
        return self.output_norm(tokens)

    @staticmethod
    def previous_visit(paths):
        """Last earlier step of each walk visiting the same state.

        A stable sort of every walk by state keeps visits of one state in
        time order, so each visit's predecessor in the sort is its previous
        visit: O(L log L) per walk instead of an ``[W, L, L]`` comparison.

        Parameters
        ----------
        paths : torch.Tensor
            Walk states ``[walk, time]``.

        Returns
        -------
        torch.Tensor
            ``[walk, time]`` step of the previous visit, -1 for none.
        """
        values, order = torch.sort(paths, dim=1, stable=True)
        previous = torch.full_like(order, -1)
        repeat = values[:, 1:] == values[:, :-1]
        previous[:, 1:] = torch.where(repeat, order[:, :-1], -1)
        return torch.full_like(order, -1).scatter_(1, order, previous)

    def _follow_pointers(self, tokens, paths, pointers):
        """Read the states of earlier steps that touched the same cells.

        Parameters
        ----------
        tokens : torch.Tensor
            Encoded states ``[walk, time, hidden]``.
        paths : torch.Tensor
            Walk states ``[walk, time]``.
        pointers : torch.Tensor
            ``[walk, time, R]`` vertex back-pointers (-1 for none).

        Returns
        -------
        torch.Tensor
            Updated states ``[walk, time, hidden]``.
        """
        walks, length = paths.shape
        time = torch.arange(length, device=paths.device)
        cell = self.previous_visit(paths)  # [W, L]
        targets = torch.cat((pointers, cell.unsqueeze(-1)), dim=2)  # [W,L,P]
        valid = targets >= 0
        if targets.shape[2] > self.max_rank + 2:
            # Wide cells: only valid pointers are gathered and transformed,
            # so memory follows the number of real pointers.
            walk, step, slot = torch.nonzero(valid, as_tuple=True)
            target = targets[walk, step, slot]
            offset = (step - target).clamp_min(1).float()
            bucket = (1 + torch.floor(torch.log2(offset)).long()).clamp_max(
                self.REVISIT_BUCKETS - 1
            )
            gathered = tokens[walk, target]
            message = self.pointer_mlp(
                gathered
                + self.pointer_offset(bucket).to(gathered.dtype)
                + self.pointer_slot(slot).to(gathered.dtype)
            )
            summed = message.new_zeros(walks * length, message.shape[-1])
            summed = summed.index_add(0, walk * length + step, message)
            return self.pointer_layer(
                tokens + summed.view(walks, length, -1).to(tokens.dtype)
            )
        index = targets.clamp_min(0)
        gathered = torch.gather(
            tokens.unsqueeze(2).expand(-1, -1, index.shape[2], -1),
            1,
            index.unsqueeze(-1).expand(-1, -1, -1, tokens.shape[-1]),
        )
        offset = (time.view(1, -1, 1) - targets).clamp_min(1).float()
        bucket = (1 + torch.floor(torch.log2(offset)).long()).clamp_max(
            self.REVISIT_BUCKETS - 1
        )
        slot = torch.arange(index.shape[2], device=paths.device)
        message = self.pointer_mlp(
            gathered
            + self.pointer_offset(bucket).to(gathered.dtype)
            + self.pointer_slot(slot).to(gathered.dtype)
        )
        message = (message * valid.unsqueeze(-1)).sum(dim=2)
        return self.pointer_layer(tokens + message)

    def _pe_summary(self, pe, ranks, state_graph, counts):
        """Per-graph, per-rank mean/std/min/max of encodings and log counts.

        Parameters
        ----------
        pe : torch.Tensor
            State encodings ``[states, pe_dim]``.
        ranks : torch.Tensor
            Rank of each state.
        state_graph : torch.Tensor
            Graph of each state.
        counts : np.ndarray
            Cells per graph and rank, ``[graphs, ranks]``.

        Returns
        -------
        torch.Tensor
            Summary of shape ``[graphs, ranks * (4 * pe_dim + 1)]``; ranks
            without cells get zeros.
        """
        num_graphs, num_ranks = counts.shape
        pe = pe.float()
        key = state_graph * num_ranks + ranks
        size = num_graphs * num_ranks
        mean = scatter(pe, key, 0, size, reduce="mean")
        square = scatter(pe * pe, key, 0, size, reduce="mean")
        std = (square - mean * mean).clamp_min(0).sqrt()
        high = scatter(pe, key, 0, size, reduce="max")
        low = -scatter(-pe, key, 0, size, reduce="max")
        present = torch.bincount(key, minlength=size).gt(0).unsqueeze(-1)
        stats = torch.cat((mean, std, low, high), dim=-1) * present
        log_counts = torch.log1p(
            torch.as_tensor(counts, dtype=pe.dtype, device=pe.device)
        ).view(size, 1)
        return torch.cat((stats, log_counts), dim=-1).view(num_graphs, -1)

    def forward(self, batch):
        """Sample and encode walks for every graph in the batch.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batch produced by ``TRAWLTransform``.

        Returns
        -------
        dict
            Contextual cell features ``x_r`` and ``batch_r``, cell, walk and
            graph embeddings with their batch indices, labels and readout
            metadata.
        """
        host = self._host(batch)
        counts = host["trawl_counts"].reshape(-1, self.max_rank + 1)
        edge_slices = getattr(batch, "_slice_dict", {}).get("trawl_edges")
        num_graphs, num_ranks = counts.shape
        views = 1 if self.training else self.eval_views
        step = self.next_sampling_step() if self.training else 0
        if self.walk_refresh != "train":
            step = 0
        device = batch.trawl_pe.device
        if (host["trawl_relations"][:num_graphs] != self.num_neighborhoods).any():
            raise ValueError("Transform/model neighborhood counts differ")

        # States are graph-major, rank-minor; trawl_signal_r rows are stored
        # per rank in graph order. Encode each rank once, then reorder.
        totals = counts.sum(axis=1)
        starts = np.concatenate(([0], np.cumsum(totals)[:-1]))
        rank_starts = np.concatenate(([0], np.cumsum(counts.sum(axis=0))[:-1]))
        within = np.cumsum(counts, axis=0) - counts
        order = np.concatenate(
            [
                np.arange(counts[g, r]) + rank_starts[r] + within[g, r]
                for g in range(num_graphs)
                for r in range(num_ranks)
            ]
            or [np.empty(0, dtype=int)]
        )
        features = torch.cat(
            [
                self.features[rank](batch[f"trawl_signal_{rank}"])
                for rank in range(num_ranks)
            ]
        )[upload(order, device)]
        num_states = int(totals.sum())
        ranks = upload(
            np.repeat(np.tile(np.arange(num_ranks), num_graphs), counts.ravel()),
            device,
        )
        state_graph = upload(np.repeat(np.arange(num_graphs), totals), device)
        pe = batch.trawl_pe[:num_states]
        if self.pe_norm:
            pe = self.pe_batch_norm(pe)
        x = features + self.position(pe) if self.use_pe else features
        if self.rank is not None:
            x = x + self.rank(ranks)
        if self.color is not None:
            host_colors = host["trawl_colors"][:num_states]
            if (
                host_colors.size
                and int(host_colors.max()) >= self.color.num_embeddings
            ):
                raise ValueError(
                    "num_colors is smaller than a transformed color ID"
                )
            x = x + self.color(batch.trawl_colors[:num_states])

        def pool_by_graph(values):
            """Pool state rows per graph, as ``_pool`` does over time.

            Parameters
            ----------
            values : torch.Tensor
                State rows of shape ``[states, hidden]``.

            Returns
            -------
            torch.Tensor
                Pooled rows of shape ``[graphs, output_dim]``.
            """
            mean = scatter(values, state_graph, 0, num_graphs, reduce="mean")
            if self.pooling == "mean":
                return mean
            amax = scatter(values, state_graph, 0, num_graphs, reduce="max")
            if self.pooling == "max":
                return amax
            return torch.cat((amax, mean), dim=-1)

        fallback = pool_by_graph(features)

        # Sample on the host per graph, then encode each group's walks from
        # the whole batch in one pass.
        groups = (
            [None]
            if self.walk_scope == "union"
            else [
                r
                for r in range(self.num_neighborhoods)
                if r != self.fallback_family
            ]
        )
        sampled = [([], [], [], [], [], [], [], [], [], []) for _ in groups]
        if self.vertex_ids and "trawl_cell_vertices" not in host:
            raise ValueError(
                "vertex_ids needs TRAWLTransform(cell_vertices=True)"
            )
        use_pivots = (
            self.pivot_embedding
            or self.cell_memory
            or self.straight_p > 0
            or (self.vertex_ids and "trawl_pivots" in host)
        )
        if self.straight_p > 0 and "trawl_cell_vertices" not in host:
            raise ValueError(
                "straight_p needs TRAWLTransform(cell_vertices=True)"
            )
        if use_pivots and "trawl_pivots" not in host:
            raise ValueError("pivot mechanisms need TRAWLTransform(pivots=True)")
        use_signs = self.orientation_embedding or self.running_orientation
        if use_signs and "trawl_signs" not in host:
            raise ValueError(
                "orientation embeddings need TRAWLTransform(signs=True)"
            )
        spec = {
            "groups": groups,
            "fallback_family": self.fallback_family,
            "views": views,
            "seed": self.seed,
            "step": step,
            "walks": self.walks,
            "walks_per_family": self.walks_per_family,
            "straight_p": self.straight_p,
            "use_pivots": use_pivots,
            "use_signs": use_signs,
            "vertex_ids": self.vertex_ids,
            "max_vertex_ids": getattr(self, "max_vertex_ids", None),
            "back_pointers": self.back_pointers,
            "vertex_features": getattr(self, "vertex_features", False),
            "running_orientation": self.running_orientation,
            "num_ranks": num_ranks,
        }
        items = []
        for graph_id in range(num_graphs):
            total = int(totals[graph_id])
            if edge_slices is None:
                lo, hi = 0, len(host["trawl_edges"])
            else:
                lo, hi = (
                    int(edge_slices[graph_id]),
                    int(edge_slices[graph_id + 1]),
                )
            first = int(starts[graph_id])
            items.append(
                (
                    graph_id,
                    total,
                    first,
                    int(host["trawl_identity"][graph_id]),
                    host["trawl_edges"][lo:hi],
                    host["trawl_weights"][lo:hi],
                    host["trawl_signs"][lo:hi] if "trawl_signs" in host else None,
                    host["trawl_pivots"][lo:hi] if "trawl_pivots" in host else None,
                    host["trawl_cell_vertices"][first : first + total]
                    if "trawl_cell_vertices" in host
                    else None,
                    counts[graph_id],
                )
            )
        if self.sampling_workers > 0 and num_graphs > 1:
            pool = _sampling_pool(
                self.sampling_workers, self.walk_sampler.cache_size
            )
            chunk = max(1, -(-num_graphs // (2 * self.sampling_workers)))
            results = pool.map(
                functools.partial(_sample_graph, spec), items, chunksize=chunk
            )
        else:
            results = [
                _sample_graph(spec, item, self.walk_sampler) for item in items
            ]
        # Same order as sampling graph by graph: graphs, then groups/views.
        for graph_id, result in enumerate(results):
            first = int(starts[graph_id])
            for group_id, slots in enumerate(result):
                for slot, values in enumerate(slots):
                    for value in values:
                        if slot == 0:
                            value = value + first
                        elif slot in (5, 9):
                            # Graph-local states (-1 = none) to batch states.
                            value = np.where(value >= 0, value + first, -1)
                        sampled[group_id][slot].append(value)

        # Accumulate occurrences in full precision under autocast. Embedding
        # additions can promote encoded states independently of projections.
        accumulation_dtype = (
            torch.float32
            if features.dtype in {torch.float16, torch.bfloat16}
            else features.dtype
        )
        contextual = torch.zeros_like(features, dtype=accumulation_dtype)
        visits = contextual.new_zeros(num_states, 1)
        occurrence_values, occurrence_ids = [], []
        walk_embeddings, walk_batches, walk_views = [], [], []
        per_group, availability = [], []
        prepared, vertex_extras = [], []
        for group_id, (
            paths,
            owners,
            view_ids,
            codes,
            running,
            step_pivots,
            vertex_ids,
            vertex_crossed,
            vertex_pointers,
            vertex_states,
        ) in enumerate(sampled):
            if not paths:
                prepared.append(None)
                vertex_extras.append(None)
                continue
            paths = upload(np.concatenate(paths), device)
            owners = upload(np.concatenate(owners), device)
            view_ids = upload(np.concatenate(view_ids), device)
            codes = (
                upload(np.concatenate(codes), device)
                if self.orientation_embedding
                else None
            )
            running = (
                upload(np.concatenate(running), device)
                if self.running_orientation
                else None
            )
            step_pivots = (
                upload(np.concatenate(step_pivots), device)
                if use_pivots
                else None
            )
            vertex_extra = (
                self._vertex_tokens(
                    upload(np.concatenate(vertex_ids), device),
                    upload(np.concatenate(vertex_crossed), device),
                    upload(np.concatenate(vertex_states), device)
                    if self.vertex_features
                    else None,
                    features,
                )
                if self.vertex_ids
                else None
            )
            vertex_extras.append(vertex_extra)
            encode_args = (
                paths,
                codes,
                running,
                step_pivots if self.pivot_embedding or self.cell_memory else None,
                vertex_extra,
                upload(np.concatenate(vertex_pointers), device)
                if self.back_pointers
                else None,
            )
            chunk = self.eval_chunk_walks
            if self.training or chunk is None or len(paths) <= chunk:
                encoded = self._encode(
                    x,
                    encode_args[0],
                    ranks,
                    group_id,
                    *encode_args[1:4],
                    extra=encode_args[4],
                    pointers=encode_args[5],
                )
            else:
                # Walks are encoded independently: chunks bound eval memory.
                encoded = torch.cat(
                    [
                        self._encode(
                            x,
                            parts[0],
                            ranks,
                            group_id,
                            *parts[1:4],
                            extra=parts[4],
                            pointers=parts[5],
                        )
                        for parts in zip(
                            *[
                                (None,) * -(-len(paths) // chunk)
                                if value is None
                                else value.split(chunk)
                                for value in encode_args
                            ],
                            strict=True,
                        )
                    ]
                )
            prepared.append(
                (paths, owners, view_ids, codes, running, step_pivots, encoded)
            )
        if self.cell_memory:
            prepared = self._memory_pass(
                x, ranks, prepared, num_states, vertex_extras
            )
        walk_families = []
        for group_id, item in enumerate(prepared):
            if item is None:
                per_group.append(fallback * 0)
                availability.append(fallback.new_zeros(num_graphs).bool())
                continue
            paths, owners, view_ids, _, _, _, encoded = item
            pooled = self._pool(encoded)
            if self.walk_attention:
                walk_families.append(torch.full_like(owners, group_id))
            walk_embeddings.append(pooled)
            walk_batches.append(owners)
            walk_views.append(view_ids)
            contextual.index_add_(
                0, paths.flatten(), encoded.flatten(0, 1).to(contextual)
            )
            if self.occurrence_attention is not None:
                occurrence_values.append(encoded.flatten(0, 1))
                occurrence_ids.append(paths.flatten())
            visits.index_add_(
                0, paths.flatten(), visits.new_ones(paths.numel(), 1)
            )
            # Mean over each view's walks, then over the views that sampled.
            key = owners * views + view_ids
            per_view = scatter(
                pooled, key, 0, num_graphs * views, reduce="mean"
            ).view(num_graphs, views, -1)
            sampled_views = (
                torch.bincount(key, minlength=num_graphs * views)
                .view(num_graphs, views)
                .gt(0)
            )
            count = sampled_views.sum(dim=1, keepdim=True)
            per_group.append(
                (per_view * sampled_views.unsqueeze(-1)).sum(dim=1)
                / count.clamp_min(1)
            )
            availability.append(count.squeeze(1) > 0)
        if self.walk_scope == "union":
            graph_embedding = torch.where(
                availability[0].unsqueeze(-1), per_group[0], fallback
            )
        else:
            stacked = torch.stack(per_group, dim=1)
            available = torch.stack(availability, dim=1)
            if self.fusion.mode in NeighborhoodFusion.GATED:
                # Learned family gates: also report the softmax weights.
                gated = available
                if self.training and self.family_dropout > 0:
                    keep = (
                        torch.rand(available.shape, device=available.device)
                        >= self.family_dropout
                    )
                    dropped = available & keep
                    # A graph that would lose every family keeps them all.
                    gated = torch.where(
                        dropped.any(dim=1, keepdim=True), dropped, available
                    )
                fused = [
                    self.fusion(stacked[g], gated[g], fallback[g], True)
                    for g in range(num_graphs)
                ]
                graph_embedding = torch.stack([f for f, _ in fused])
                live_weights = torch.stack([w for _, w in fused])
                family_weights = live_weights.detach()
                if self.training and self.gate_entropy > 0:
                    entropy = -(
                        live_weights * torch.log(live_weights.clamp_min(1e-12))
                    ).sum(dim=1)
                    gate_aux_loss = -self.gate_entropy * entropy.mean()
            else:
                graph_embedding = torch.stack(
                    [
                        self.fusion(stacked[g], available[g], fallback[g])
                        for g in range(num_graphs)
                    ]
                )
        if self.walk_attention and walk_embeddings:
            graph_embedding = self._attend_walks(
                torch.cat(walk_embeddings),
                torch.cat(walk_batches),
                torch.cat(walk_families),
                num_graphs,
                graph_embedding,
            )
        contextual = torch.where(
            visits > 0, contextual / visits.clamp_min(1), features
        )
        if occurrence_values:
            from torch_geometric.utils import softmax

            values, ids = (
                torch.cat(occurrence_values),
                torch.cat(occurrence_ids),
            )
            weights = softmax(
                self.occurrence_attention(values), ids, num_nodes=num_states
            )
            attended = torch.zeros_like(contextual).index_add(
                0, ids, (values * weights).to(contextual)
            )
            contextual = torch.where(visits > 0, attended, features)
        if self.graph_readout == "cells":
            graph_embedding = pool_by_graph(contextual)
        if self.pe_skip:
            graph_embedding = graph_embedding + self.pe_skip_net(
                self._pe_summary(
                    batch.trawl_pe[:num_states], ranks, state_graph, counts
                )
            ).to(graph_embedding.dtype)
        result = {}
        if (
            self.walk_scope == "separate"
            and self.fusion.mode in NeighborhoodFusion.GATED
        ):
            result["family_weights"] = family_weights.float()
            if self.training and self.gate_entropy > 0:
                result["aux_loss"] = gate_aux_loss.float()
        result |= {
            "cell_embeddings": contextual,
            "cell_ranks": ranks,
            "graph_embedding": graph_embedding,
            "labels": batch.get("y"),
            "walk_readout_compatible": self.walk_scope == "union"
            and self.graph_readout == "walks",
            "num_views": views,
        }
        for rank in range(self.max_rank + 1):
            mask = ranks == rank
            result[f"x_{rank}"] = contextual[mask]
            result[f"batch_{rank}"] = state_graph[mask]
        result["walk_embedding"] = (
            torch.cat(walk_embeddings)
            if walk_embeddings
            else graph_embedding.new_empty((0, self.output_dim))
        )
        result["walk_batch"] = (
            torch.cat(walk_batches)
            if walk_batches
            else result["batch_0"].new_empty(0)
        )
        result["walk_view"] = (
            torch.cat(walk_views)
            if walk_views
            else result["batch_0"].new_empty(0)
        )
        return result


# ---------------------------------------------------------------------------
# Feature adapter and task head
# ---------------------------------------------------------------------------


class TRAWLFeatureEncoder(AbstractFeatureEncoder):
    """Synchronize a native encoder's x_r outputs with TRAWL state features.

    When using this adapter, set backbone.in_channels to the encoder's output
    widths; input shape inference runs before learned feature encoding.

    Parameters
    ----------
    encoder : torch.nn.Module
        TopoBench feature encoder producing ``x_r`` outputs.
    ranks : iterable of int
        Ranks whose encoded features replace ``trawl_signal_r``.
    """

    def __init__(self, encoder, ranks):
        super().__init__()
        self.encoder, self.ranks = encoder, list(ranks)

    def forward(self, batch):
        """Encode features and copy them into TRAWL state signals.

        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Input batch.

        Returns
        -------
        torch_geometric.data.Batch
            Batch with encoded ``x_r`` and matching ``trawl_signal_r``.
        """
        batch = self.encoder(batch)
        for rank in self.ranks:
            batch[f"trawl_signal_{rank}"] = batch[f"x_{rank}"]
        return batch


class TRAWLReadout(nn.Module):
    """Predict from graph embeddings, mean walk logits, or contextual nodes.

    Parameters
    ----------
    hidden_dim : int
        Width of node embeddings, and of graph embeddings if ``graph_dim`` is
        None.
    out_channels : int
        Number of output channels.
    task_level : str, optional
        ``"node"`` or ``"graph"`` (default: "graph").
    graph_dim : int, optional
        Width of graph and walk embeddings (default: None).
    head_hidden : int, optional
        Hidden width of an MLP head; a linear head is used if None
        (default: None).
    dropout : float, optional
        Dropout probability inside the head (default: 0.0).
    aggregation : str, optional
        Graph aggregation: ``"embedding"`` or ``"walk_logits"``
        (default: "embedding").
    input_dropout : float, optional
        Dropout probability on head inputs; defaults to ``dropout`` if None
        (default: None).
    walk_reduce : str, optional
        With ``walk_logits``, how per-walk logits are combined per graph:
        ``"mean"``, ``"max"`` or ``"logsumexp"`` (minus the log walk count,
        so it lies between mean and max) (default: "mean").
    """

    def __init__(
        self,
        hidden_dim,
        out_channels,
        task_level="graph",
        graph_dim=None,
        head_hidden=None,
        dropout=0.0,
        aggregation="embedding",
        input_dropout=None,
        walk_reduce="mean",
    ):
        super().__init__()
        if walk_reduce not in {"mean", "max", "logsumexp"}:
            raise ValueError("walk_reduce must be mean, max or logsumexp")
        self.walk_reduce = walk_reduce
        if task_level not in {"node", "graph"}:
            raise ValueError("TRAWL supports TopoBench's node and graph tasks")
        if aggregation not in {"embedding", "walk_logits"}:
            raise ValueError("Unknown readout aggregation")
        self.task_level, self.aggregation = task_level, aggregation
        width = (
            hidden_dim if task_level == "node" else (graph_dim or hidden_dim)
        )
        input_dropout = dropout if input_dropout is None else input_dropout
        if head_hidden:
            self.head = nn.Sequential(
                nn.Dropout(input_dropout),
                nn.Linear(width, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, out_channels),
            )
        else:
            # A linear head has a single dropout, applied to its input.
            self.head = nn.Sequential(
                nn.Dropout(input_dropout), nn.Linear(width, out_channels)
            )

    def forward(self, model_out, batch):
        """Compute logits from the backbone output.

        Parameters
        ----------
        model_out : dict
            Backbone output dictionary.
        batch : torch_geometric.data.Batch
            Input batch (unused).

        Returns
        -------
        dict
            ``model_out`` with ``logits`` added.
        """
        if self.task_level == "node":
            logits = self.head(model_out["x_0"])
        else:
            if self.aggregation != "embedding" and not model_out.get(
                "walk_readout_compatible", True
            ):
                raise ValueError(
                    "Use embedding aggregation with separate neighborhoods or cell readout to preserve their fusion"
                )
            if self.aggregation == "walk_logits" and len(
                model_out["walk_batch"]
            ):
                predictions = self.head(model_out["walk_embedding"])
                shape = (
                    len(model_out["graph_embedding"]),
                    predictions.shape[-1],
                )
                sums = predictions.new_zeros(shape).index_add(
                    0, model_out["walk_batch"], predictions
                )
                count = predictions.new_zeros(shape[0], 1).index_add(
                    0,
                    model_out["walk_batch"],
                    predictions.new_ones(len(predictions), 1),
                )
                logits = sums / count.clamp_min(1)
                if self.walk_reduce != "mean":
                    owners = model_out["walk_batch"]
                    high = scatter(
                        predictions, owners, 0, shape[0], reduce="max"
                    )
                    if self.walk_reduce == "max":
                        logits = high
                    else:
                        total = predictions.new_zeros(shape).index_add(
                            0, owners, (predictions - high[owners]).exp()
                        )
                        logits = (
                            high
                            + total.clamp_min(1e-30).log()
                            - count.clamp_min(1).log()
                        )
                missing = count.squeeze(-1) == 0
                if missing.any():
                    logits[missing] = self.head(
                        model_out["graph_embedding"][missing]
                    )
            else:
                logits = self.head(model_out["graph_embedding"])
        model_out["logits"] = logits
        return model_out
