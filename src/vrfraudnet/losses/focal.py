"""Focal-loss surrogate for gradient-boosted trees (manuscript S4.2, Table 2).

The manuscript states that imbalance is handled by a "focal + class-balanced"
objective and that the focal term is a *surrogate* compatible with GBDT
training. A custom LightGBM objective is the faithful implementation: it
supplies the true first and second derivatives of the focal loss with respect to
the raw margin, so the trees are actually grown against the focal objective
rather than against log-loss with reweighted samples.

Focal loss (Lin et al., 2017) for binary labels ``y in {0, 1}`` and predicted
probability ``p = sigmoid(z)``:

    FL = -[ y (1 - p)^gamma log p + (1 - y) p^gamma log(1 - p) ]

Gradients are derived analytically below; ``tests/test_losses.py`` checks them
against finite differences.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(z, dtype=np.float64)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def focal_loss(z: np.ndarray, y: np.ndarray, *, gamma: float = 2.0) -> np.ndarray:
    """Elementwise focal loss on raw margins ``z``."""
    p = np.clip(sigmoid(np.asarray(z, dtype=np.float64)), 1e-12, 1 - 1e-12)
    y = np.asarray(y, dtype=np.float64)
    pos = y * ((1.0 - p) ** gamma) * np.log(p)
    neg = (1.0 - y) * (p**gamma) * np.log(1.0 - p)
    return -(pos + neg)


def focal_grad_hess(
    z: np.ndarray, y: np.ndarray, *, gamma: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """First and second derivatives of the focal loss w.r.t. the raw margin.

    The Hessian is clipped below at a small positive value: LightGBM requires
    positive curvature to compute leaf values, and the exact focal Hessian is
    negative in a small region of margin space.
    """
    z = np.asarray(z, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # d/dz of -[y (1-p)^g log p + (1-y) p^g log(1-p)], with dp/dz = p(1-p).
    grad, _ = _grad_only(z, y, gamma)

    # Second derivative via central differences on the analytic gradient. This
    # is exact to O(h^2) and avoids a page of algebra that would be harder to
    # audit than the finite-difference check in the tests.
    h = 1e-4
    g_plus, _ = _grad_only(z + h, y, gamma)
    g_minus, _ = _grad_only(z - h, y, gamma)
    hess = (g_plus - g_minus) / (2.0 * h)
    hess = np.maximum(hess, 1e-6)
    return grad, hess


def _grad_only(z: np.ndarray, y: np.ndarray, gamma: float) -> tuple[np.ndarray, None]:
    p = np.clip(sigmoid(z), 1e-12, 1 - 1e-12)
    dp = p * (1.0 - p)
    grad_pos = y * (
        gamma * ((1.0 - p) ** (gamma - 1.0)) * np.log(p) * dp - ((1.0 - p) ** gamma) * (dp / p)
    )
    grad_neg = (1.0 - y) * (
        -gamma * (p ** (gamma - 1.0)) * np.log(1.0 - p) * dp + (p**gamma) * (dp / (1.0 - p))
    )
    return grad_pos + grad_neg, None


def focal_objective_factory(*, gamma: float = 2.0) -> Callable:
    """Return a LightGBM-compatible ``(preds, dataset) -> (grad, hess)`` objective."""

    def objective(preds: np.ndarray, dataset) -> tuple[np.ndarray, np.ndarray]:
        labels = dataset.get_label()
        weights = dataset.get_weight()
        grad, hess = focal_grad_hess(preds, labels, gamma=gamma)
        if weights is not None:
            grad = grad * weights
            hess = hess * weights
        return grad, hess

    return objective


def torch_focal_loss(logits, targets, *, gamma: float = 2.0, weight=None):
    """Focal loss for the torch-based neural baselines and Stage 0 training."""
    import torch
    import torch.nn.functional as functional

    probabilities = torch.sigmoid(logits)
    ce = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    loss = ((1.0 - p_t) ** gamma) * ce
    if weight is not None:
        loss = loss * weight
    return loss.mean()
