"""DeLong test for two correlated ROC curves (manuscript Section 5.2, Tables 6b-6d).

Implements the fast DeLong algorithm (Sun and Xu, 2014) for the covariance of
two AUCs estimated on the *same* set of observations, and the resulting z-test.

What DeLong requires, and what the manuscript does not supply (audit A-03)
--------------------------------------------------------------------------
The DeLong test consumes two score vectors over one labelled sample. It is not
defined for a pair of "mean +/- standard deviation" summaries. Manuscript
Section 4.8.1 reports ten seeded runs per model, so there are ten score vectors
per model per dataset, and the manuscript does not say which of them produced
the reported ``Z``. Three readings are possible and give different numbers:

1. one designated seed's predictions;
2. the seed-averaged score per transaction, then a single DeLong test;
3. ten DeLong tests, then some pooling of the statistics.

This module implements reading (1) and (2) explicitly and requires the caller to
choose, recording the choice in the result. It will not silently pick one.

Because a DeLong ``Z`` cannot be recomputed from published summary statistics,
Tables 6(b)-6(d) cannot be regenerated from the manuscript alone; they require
the per-transaction prediction files. See docs/KNOWN_LIMITATIONS.md L-01.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import stats

from vrfraudnet.errors import ConfigurationError

SeedPolicy = Literal["single_seed", "seed_mean_scores"]


@dataclass
class DeLongResult:
    """One DeLong comparison between VR-FraudNet and a baseline."""

    dataset: str
    baseline: str
    auc_model: float
    auc_baseline: float
    delta_pp: float
    z: float
    p_value_two_sided: float
    n: int
    n_positive: int
    seed_policy: SeedPolicy
    seed: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "baseline": self.baseline,
            "auc_model": self.auc_model,
            "auc_baseline": self.auc_baseline,
            "delta_roc_auc_pp": self.delta_pp,
            "delong_z": self.z,
            "p_raw_two_sided": self.p_value_two_sided,
            "n": self.n,
            "n_positive": self.n_positive,
            "seed_policy": self.seed_policy,
            "seed": self.seed,
        }


def _midrank(x: np.ndarray) -> np.ndarray:
    """Mid-ranks with ties averaged, as required by the DeLong construction."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = x.size
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def fast_delong(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong AUC estimates and covariance matrix for ``k`` score vectors.

    Parameters
    ----------
    scores:
        Shape ``(k, n)``: one row per model, columns aligned to the same
        observations.
    labels:
        Shape ``(n,)`` binary labels, fraud encoded as 1.

    Returns
    -------
    (aucs, covariance) with shapes ``(k,)`` and ``(k, k)``.
    """
    labels = np.asarray(labels).astype(int)
    positive = labels == 1
    negative = ~positive
    m, n = int(positive.sum()), int(negative.sum())
    if m == 0 or n == 0:
        raise ConfigurationError("DeLong requires both classes to be present")

    pos = scores[:, positive]
    neg = scores[:, negative]
    k = scores.shape[0]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(np.concatenate([pos[r], neg[r]]))

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    s01 = np.cov(v01)
    s10 = np.cov(v10)
    s01 = np.atleast_2d(s01)
    s10 = np.atleast_2d(s10)
    covariance = s01 / m + s10 / n
    return aucs, covariance


def delong_test(
    model_scores: np.ndarray,
    baseline_scores: np.ndarray,
    labels: np.ndarray,
    *,
    dataset: str,
    baseline: str,
    seed_policy: SeedPolicy,
    seed: int | None = None,
) -> DeLongResult:
    """Two-sided DeLong z-test between two correlated AUCs on one sample.

    ``seed_policy`` must be stated explicitly; see the module docstring and
    audit finding A-03.
    """
    if seed_policy not in ("single_seed", "seed_mean_scores"):
        raise ConfigurationError(
            "seed_policy must be 'single_seed' or 'seed_mean_scores'; the manuscript "
            "does not state which prediction vectors produced its DeLong statistics "
            "(audit finding A-03), so the choice must be recorded explicitly"
        )
    if seed_policy == "single_seed" and seed is None:
        raise ConfigurationError("seed_policy='single_seed' requires the seed to be named")

    stacked = np.vstack(
        [np.asarray(model_scores, dtype=float), np.asarray(baseline_scores, dtype=float)]
    )
    aucs, covariance = fast_delong(stacked, labels)
    variance = covariance[0, 0] + covariance[1, 1] - 2.0 * covariance[0, 1]
    if variance <= 0:
        z = 0.0
        p = 1.0
    else:
        z = float((aucs[0] - aucs[1]) / np.sqrt(variance))
        p = float(2.0 * stats.norm.sf(abs(z)))

    return DeLongResult(
        dataset=dataset,
        baseline=baseline,
        auc_model=float(aucs[0]),
        auc_baseline=float(aucs[1]),
        delta_pp=float((aucs[0] - aucs[1]) * 100.0),
        z=z,
        p_value_two_sided=p,
        n=int(labels.shape[0]),
        n_positive=int(np.asarray(labels).sum()),
        seed_policy=seed_policy,
        seed=seed,
    )
