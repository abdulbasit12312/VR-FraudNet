"""Holm-Bonferroni correction and bootstrap confidence intervals (S5.2).

The manuscript applies Holm-Bonferroni across the nine baseline comparisons and
reports Holm-adjusted p-values. It does not report confidence intervals; the
bootstrap helper here is provided because a Q1 reviewer will ask for one, and it
is clearly marked as an addition rather than a reproduction.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


def holm_bonferroni(p_values: Mapping[str, float]) -> dict[str, float]:
    """Step-down Holm-Bonferroni adjustment.

    Sort ascending, multiply the k-th smallest by ``m - k``, then enforce
    monotonicity and cap at 1. Returns a mapping with the same keys.
    """
    if not p_values:
        return {}
    items = sorted(p_values.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: list[tuple[str, float]] = []
    running = 0.0
    for k, (key, p) in enumerate(items):
        value = (m - k) * float(p)
        running = max(running, value)
        adjusted.append((key, min(1.0, running)))
    return dict(adjusted)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for a statistic of ``values``.

    Returns ``(point_estimate, lower, upper)``. Not a manuscript quantity; see
    the module docstring.
    """
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        raise ValueError("bootstrap requires at least two observations")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, data.size, size=(n_resamples, data.size))
    estimates = np.array([statistic(data[idx]) for idx in draws])
    alpha = 1.0 - confidence
    return (
        float(statistic(data)),
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    )


def paired_bootstrap_ci(
    model: Sequence[float],
    baseline: Sequence[float],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Percentile bootstrap interval for the paired mean difference, in pp."""
    a = np.asarray(model, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have identical shape")
    return bootstrap_ci(
        (a - b) * 100.0, n_resamples=n_resamples, confidence=confidence, seed=seed
    )
