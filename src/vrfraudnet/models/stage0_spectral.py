"""Stage 0: time-conditioned spectral graph encoder (manuscript S4.1, Figure 3).

Specification taken from the manuscript
---------------------------------------
* Input: ``G_t = (V_t, E_t, X_t, T_t)`` observable at evaluation time ``t``
  (Eq. 1). Only nodes, edges, features and timestamps observable by ``t`` are
  used; future edges, future labels and post-split graph information are
  excluded.
* Normalised Laplacian ``L_t = I - D_t^(-1/2) A_t D_t^(-1/2)`` (Eq. 2).
* ``K = 4`` Beta-kernel spectral filters (S4.1, Figure 3, Table 3).
* The evaluation time is embedded into a 32-dimensional representation.
* A two-layer MLP with GELU activation produces time-conditioned coefficients
  ``theta_k(t)`` for the spectral filters.
* Spectral convolution combines filtered structure, node features and temporal
  coefficients into a 64-dimensional per-node representation ``z_g``.
* Optimiser AdamW, lr 1e-3, weight decay 1e-4, max 100 epochs, early-stopping
  patience 15 on validation AUPRC, LR halved after 5 stagnant epochs with floor
  1e-5, node mini-batch 2048, two-hop neighbour sampling with fan-outs 25 and
  10, gradient-norm clipping at 1.0 (S4.8.1).

NOT specified by the manuscript (ASSUMPTION, see docs/KNOWN_LIMITATIONS.md)
---------------------------------------------------------------------------
* L-08: the Beta-kernel order. The Beta wavelet basis of Bo et al. (BWGNN) is
  used, with ``W_{p,q} = c * (L/2)^p (I - L/2)^q`` and ``(p, q)`` ranging over
  ``p + q = K``, which yields exactly ``K + 1`` band-pass filters; the four
  filters used are ``p = 1..K``. The manuscript says "four Beta-kernel spectral
  filters" without giving the order.
* L-09: the hidden width of the coefficient MLP, and whether the temporal
  coefficients are shared across feature channels. A single scalar coefficient
  per filter is used here (the reading that matches "contributions adjusted by a
  learned temporal representation").
* L-10: the time-embedding parameterisation. A sinusoidal-plus-linear embedding
  is used, which is deterministic and requires no additional hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrfraudnet.errors import ConfigurationError


@dataclass(frozen=True)
class Stage0Config:
    """Stage 0 hyperparameters. Defaults are the manuscript values."""

    n_filters: int = 4  # MANUSCRIPT S4.1 / Table 3: K = 4
    time_embedding_dim: int = 32  # MANUSCRIPT S4.1 / Figure 3
    output_dim: int = 64  # MANUSCRIPT S4.1 / Table 3: z_g in R^64
    mlp_hidden_dim: int = 64  # ASSUMPTION L-09
    beta_order: int = 4  # ASSUMPTION L-08
    dropout: float = 0.0  # ASSUMPTION L-09
    fanouts: tuple[int, int] = (25, 10)  # MANUSCRIPT S4.8.1
    batch_size: int = 2048  # MANUSCRIPT S4.8.1
    learning_rate: float = 1e-3  # MANUSCRIPT S4.8.1
    weight_decay: float = 1e-4  # MANUSCRIPT S4.8.1
    max_epochs: int = 100  # MANUSCRIPT S4.8.1
    patience: int = 15  # MANUSCRIPT S4.8.1
    lr_reduce_factor: float = 0.5  # MANUSCRIPT S4.8.1
    lr_reduce_patience: int = 5  # MANUSCRIPT S4.8.1
    min_learning_rate: float = 1e-5  # MANUSCRIPT S4.8.1
    grad_clip_norm: float = 1.0  # MANUSCRIPT S4.8.1


def normalised_laplacian(
    edge_index: np.ndarray, num_nodes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the sparse normalised Laplacian ``L = I - D^-1/2 A D^-1/2`` (Eq. 2).

    Returned as ``(indices, values)`` in COO form with shape ``(2, nnz)`` and
    ``(nnz,)``. Self-loops from the identity term are included explicitly.

    Isolated nodes get ``d = 0``; their normalisation term is defined as 0,
    which leaves the identity contribution and is the standard convention.
    """
    if edge_index.shape[0] != 2:
        raise ConfigurationError("edge_index must have shape (2, E)")

    src, dst = edge_index
    # Symmetrise: the manuscript's money-flow / account graphs are directed, but
    # the normalised Laplacian of Eq. 2 is defined for a symmetric adjacency.
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])

    degree = np.bincount(src_sym, minlength=num_nodes).astype(np.float64)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(np.maximum(degree, 1e-12)), 0.0)

    off_values = -d_inv_sqrt[src_sym] * d_inv_sqrt[dst_sym]
    diag_index = np.arange(num_nodes, dtype=np.int64)
    diag_values = np.ones(num_nodes, dtype=np.float64)

    indices = np.concatenate(
        [np.vstack([src_sym, dst_sym]), np.vstack([diag_index, diag_index])], axis=1
    )
    values = np.concatenate([off_values, diag_values])
    return indices, values


def beta_kernel_coefficients(p: int, q: int) -> np.ndarray:
    """Polynomial coefficients of the Beta wavelet ``(L/2)^p (I - L/2)^q``.

    Returns coefficients ``c`` such that the filter equals ``sum_j c[j] L^j``,
    normalised by ``1 / (2^(p+q) * B(p+1, q+1))`` as in the Beta wavelet
    construction. Working in the monomial basis keeps the implementation
    dependency-free and exactly testable.
    """
    if p < 0 or q < 0:
        raise ValueError("Beta kernel orders must be non-negative")
    from math import comb, factorial

    # (I - L/2)^q = sum_i C(q, i) (-1/2)^i L^i ; multiply by (L/2)^p.
    coeffs = np.zeros(p + q + 1, dtype=np.float64)
    for i in range(q + 1):
        coeffs[p + i] += comb(q, i) * ((-0.5) ** i)
    coeffs *= 0.5**p

    # Beta normalisation B(p+1, q+1) = p! q! / (p + q + 1)!
    beta = factorial(p) * factorial(q) / factorial(p + q + 1)
    return coeffs / beta


def apply_polynomial_filter(
    indices: np.ndarray,
    values: np.ndarray,
    num_nodes: int,
    features: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Evaluate ``sum_j c[j] L^j X`` by repeated sparse matrix-vector products."""
    out = np.zeros_like(features, dtype=np.float64)
    current = features.astype(np.float64)
    for j, coefficient in enumerate(coefficients):
        if j > 0:
            current = _spmm(indices, values, num_nodes, current)
        if coefficient != 0.0:
            out += coefficient * current
    return out


def _spmm(
    indices: np.ndarray, values: np.ndarray, num_nodes: int, dense: np.ndarray
) -> np.ndarray:
    """Sparse (COO) times dense product without a SciPy dependency."""
    rows, cols = indices
    out = np.zeros((num_nodes, dense.shape[1]), dtype=np.float64)
    np.add.at(out, rows, values[:, None] * dense[cols])
    return out


def time_embedding(t: np.ndarray, dim: int) -> np.ndarray:
    """Deterministic sinusoidal time embedding (ASSUMPTION L-10).

    ``emb(t)[2i]   = sin(t / 10000^(2i/dim))``
    ``emb(t)[2i+1] = cos(t / 10000^(2i/dim))``

    Chosen because it introduces no free parameters, so the only learned part of
    the temporal pathway is the MLP the manuscript actually describes.
    """
    if dim % 2 != 0:
        raise ValueError("time embedding dimension must be even")
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    freqs = np.exp(-np.log(10000.0) * np.arange(0, dim, 2, dtype=np.float64) / dim)
    angles = t * freqs
    return np.concatenate([np.sin(angles), np.cos(angles)], axis=1)


def build_torch_module(config: Stage0Config, in_dim: int):
    """Construct the trainable Stage 0 module.

    Kept behind a function so that the NumPy reference operators above (which
    the unit tests exercise) do not require torch to be installed.
    """
    import torch
    from torch import nn

    class BetaSpectralEncoder(nn.Module):
        """Beta-kernel spectral filters with time-conditioned mixing coefficients."""

        def __init__(self) -> None:
            super().__init__()
            self.config = config
            self.kernels = [
                beta_kernel_coefficients(p, config.beta_order - p)
                for p in range(1, config.n_filters + 1)
            ]
            self.filter_projections = nn.ModuleList(
                nn.Linear(in_dim, config.output_dim) for _ in range(config.n_filters)
            )
            # Two-layer MLP with GELU producing one coefficient per filter.
            self.coefficient_mlp = nn.Sequential(
                nn.Linear(config.time_embedding_dim, config.mlp_hidden_dim),
                nn.GELU(),
                nn.Linear(config.mlp_hidden_dim, config.n_filters),
            )
            self.output_norm = nn.LayerNorm(config.output_dim)
            self.dropout = nn.Dropout(config.dropout)

        def forward(
            self,
            filtered_stack: "torch.Tensor",
            time_features: "torch.Tensor",
        ) -> "torch.Tensor":
            """Combine pre-filtered node features with time-conditioned weights.

            Parameters
            ----------
            filtered_stack:
                Shape ``(K, N, in_dim)``: the node features after each of the K
                Beta filters. Precomputing the filters outside the autograd graph
                keeps memory bounded on the 3.7M-node D5 graph.
            time_features:
                Shape ``(N, time_embedding_dim)`` time embeddings.
            """
            theta = torch.softmax(self.coefficient_mlp(time_features), dim=-1)  # (N, K)
            out = None
            for k, projection in enumerate(self.filter_projections):
                contribution = projection(filtered_stack[k]) * theta[:, k : k + 1]
                out = contribution if out is None else out + contribution
            return self.output_norm(self.dropout(out))

    return BetaSpectralEncoder()


def precompute_filter_bank(
    edge_index: np.ndarray,
    num_nodes: int,
    features: np.ndarray,
    config: Stage0Config,
) -> np.ndarray:
    """Apply the K Beta filters to node features; returns shape ``(K, N, F)``.

    Callers are responsible for passing an edge set that has already been
    restricted to the evaluation time (use
    :meth:`vrfraudnet.data.types.TemporalGraph.snapshot`). This function has no
    way to check that, so :func:`assert_no_future_edges` exists as the explicit
    guard used by the pipeline and the tests.
    """
    indices, values = normalised_laplacian(edge_index, num_nodes)
    stack = np.empty((config.n_filters, num_nodes, features.shape[1]), dtype=np.float64)
    for k, p in enumerate(range(1, config.n_filters + 1)):
        coefficients = beta_kernel_coefficients(p, config.beta_order - p)
        stack[k] = apply_polynomial_filter(indices, values, num_nodes, features, coefficients)
    return stack


def assert_no_future_edges(edge_time: np.ndarray, evaluation_time: float) -> None:
    """Guard for manuscript S4.1: no edge later than ``t`` may be visible.

    Raises :class:`~vrfraudnet.errors.LeakageError` on violation.
    """
    from vrfraudnet.errors import LeakageError

    if edge_time.size and float(np.max(edge_time)) > evaluation_time:
        offending = int(np.sum(edge_time > evaluation_time))
        raise LeakageError(
            f"Stage 0 received {offending} edge(s) with a timestamp later than the "
            f"evaluation time {evaluation_time!r}; the maximum observed edge time is "
            f"{float(np.max(edge_time))!r}. Snapshot the graph before encoding."
        )
