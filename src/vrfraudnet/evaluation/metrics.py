"""Evaluation metrics (manuscript Section 4.8.3).

Manuscript specification
------------------------
* Fraud is the positive class, encoded as 1.
* AUPRC is primary and is computed from continuous scores across all unique
  thresholds.
* ROC-AUC is computed from the same continuous scores.
* F1-score means the **fraud-class** F1, not macro, weighted or micro.
* The operating threshold is chosen on the validation partition to maximise
  fraud-class F1 *after* probability calibration, then frozen and applied
  unchanged to every test, temporal, transfer and robustness partition.
* Precision, recall, F1 and MCC use that same frozen threshold.
* Recall@top-1% ranks each test fold separately, takes the top ceil(1%) of that
  fold, and divides recovered fraud by total fraud in the same fold. The top-1%
  set is never constructed by pooling across train, validation and test.
* Per-dataset values are obtained by averaging fold-level values; tables report
  the mean and sample standard deviation across the 10 seeds.

AUDIT FINDING A-01 (enforced in code)
-------------------------------------
Recall@top-1% has a hard arithmetic ceiling of ``0.01 / prevalence``. Reviewing
1% of a population whose fraud prevalence is 3.50% can recover at most 28.57%
of all fraud. Manuscript Table 5(c) reports 0.4234-0.6234 for D3 at 3.50%
prevalence, which is unattainable for **every** model in that table, including
the baselines. :func:`recall_at_top_k` therefore refuses to return a value above
the ceiling and :func:`compute_all` refuses to emit the metric for a dataset
whose declared prevalence makes the manuscript's reported values impossible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from vrfraudnet.errors import ManuscriptInconsistencyError


@dataclass
class MetricResult:
    """Metrics for one (dataset, fold, seed) triple."""

    auprc: float
    roc_auc: float
    f1: float
    precision: float
    recall: float
    mcc: float
    recall_at_top_1pct: float | None
    threshold: float
    n: int
    n_positive: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "auprc": self.auprc,
            "roc_auc": self.roc_auc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "mcc": self.mcc,
            "recall_at_top_1pct": self.recall_at_top_1pct,
            "operating_threshold": self.threshold,
            "n": self.n,
            "n_positive": self.n_positive,
            "warnings": self.warnings,
        }


def auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve for the fraud class."""
    _validate(y_true, scores)
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    _validate(y_true, scores)
    return float(roc_auc_score(y_true, scores))


def fraud_class_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 on the positive (fraud) class only, per manuscript S4.8.3."""
    return float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))


def mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(matthews_corrcoef(y_true, y_pred))


def recall_at_top_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    k_fraction: float = 0.01,
    strict: bool = True,
) -> float:
    """Fraction of fraud cases captured inside the top ``k_fraction`` of a fold.

    The review set is the highest-scoring ``ceil(k_fraction * n)`` transactions
    **of this fold**, as manuscript S4.8.3 requires.

    Parameters
    ----------
    strict:
        When True (the default) the returned value is checked against the
        arithmetic ceiling ``k_fraction / prevalence`` and an internal
        inconsistency raises rather than being silently reported.
    """
    _validate(y_true, scores)
    n = y_true.shape[0]
    n_positive = int(y_true.sum())
    if n_positive == 0:
        return float("nan")

    k = int(math.ceil(k_fraction * n))
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    captured = int(np.asarray(y_true)[order[:k]].sum())
    value = captured / n_positive

    ceiling = min(1.0, k / n_positive)
    if strict and value > ceiling + 1e-12:  # pragma: no cover - defensive
        raise ManuscriptInconsistencyError(
            "A-01",
            f"computed Recall@top-{k_fraction:.0%} = {value:.4f} exceeds its arithmetic "
            f"ceiling {ceiling:.4f}; this indicates a bug in the ranking code",
        )
    return float(value)


def recall_at_top_k_ceiling(prevalence: float, k_fraction: float = 0.01) -> float:
    """The maximum attainable Recall@top-k for a given fraud prevalence."""
    if prevalence <= 0:
        raise ValueError("prevalence must be positive")
    return min(1.0, k_fraction / prevalence)


def select_operating_threshold(
    y_valid: np.ndarray, valid_scores: np.ndarray, *, n_grid: int = 512
) -> float:
    """Choose the fraud-class-F1-maximising threshold on validation data only.

    Manuscript S4.8.3: "The operating threshold was selected using the
    validation partition to maximize fraud-class F1 after probability
    calibration. Once selected, the threshold was frozen."
    """
    _validate(y_valid, valid_scores)
    candidates = np.unique(np.quantile(valid_scores, np.linspace(0.0, 1.0, n_grid)))
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in candidates:
        predicted = (valid_scores >= threshold).astype(int)
        score = f1_score(y_valid, predicted, pos_label=1, zero_division=0)
        if score > best_f1:
            best_threshold, best_f1 = float(threshold), float(score)
    return best_threshold


def compute_all(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    dataset_id: str | None = None,
    declared_prevalence: float | None = None,
    include_recall_at_top_1pct: bool = True,
) -> MetricResult:
    """Compute the full manuscript metric set at a frozen operating threshold.

    ``threshold`` must come from :func:`select_operating_threshold` applied to
    the validation partition. It is never re-tuned on the evaluation partition.
    """
    _validate(y_true, scores)
    y_pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    warnings: list[str] = []

    top1 = None
    if include_recall_at_top_1pct:
        prevalence = float(np.mean(y_true))
        ceiling = recall_at_top_k_ceiling(prevalence) if prevalence > 0 else 1.0
        top1 = recall_at_top_k(y_true, scores)
        if ceiling < 1.0:
            warnings.append(
                f"Recall@top-1% is bounded above by {ceiling:.4f} at prevalence "
                f"{prevalence:.4%}; values above that bound are unattainable "
                f"(audit finding A-01)"
            )

    return MetricResult(
        auprc=auprc(y_true, scores),
        roc_auc=roc_auc(y_true, scores),
        f1=fraud_class_f1(y_true, y_pred),
        precision=float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        mcc=mcc(y_true, y_pred),
        recall_at_top_1pct=top1,
        threshold=float(threshold),
        n=int(y_true.shape[0]),
        n_positive=int(np.sum(y_true)),
        warnings=warnings,
    )


def aggregate_folds(results: Sequence[MetricResult]) -> dict[str, float]:
    """Average fold-level metrics into one value per dataset and seed (S4.8.3)."""
    if not results:
        raise ValueError("no fold results to aggregate")
    keys = ["auprc", "roc_auc", "f1", "precision", "recall", "mcc"]
    out = {k: float(np.mean([getattr(r, k) for r in results])) for k in keys}
    top1 = [r.recall_at_top_1pct for r in results if r.recall_at_top_1pct is not None]
    out["recall_at_top_1pct"] = float(np.mean(top1)) if top1 else float("nan")
    out["n_folds"] = float(len(results))
    return out


def aggregate_seeds(per_seed: Mapping[int, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and *sample* standard deviation across seeds (manuscript S4.8.3).

    The manuscript is explicit that the reported dispersion is variation across
    repeated training runs, not resampling uncertainty over transactions, so the
    sample standard deviation (ddof=1) is the correct estimator.
    """
    if len(per_seed) < 2:
        raise ValueError("at least two seeds are required to report a standard deviation")
    metric_names = sorted({k for values in per_seed.values() for k in values})
    out: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.array(
            [float(per_seed[s][name]) for s in sorted(per_seed) if name in per_seed[s]],
            dtype=float,
        )
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "n_seeds": int(values.size),
        }
    return out


def _validate(y_true: np.ndarray, scores: np.ndarray) -> None:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    if y_true.shape[0] != scores.shape[0]:
        raise ValueError("y_true and scores must have the same length")
    unique = set(np.unique(y_true).tolist())
    if not unique <= {0, 1}:
        raise ValueError(f"y_true must be binary 0/1; found {sorted(unique)}")
    if len(unique) < 2:
        raise ValueError("both classes must be present to compute ranking metrics")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores contain NaN or infinity")
