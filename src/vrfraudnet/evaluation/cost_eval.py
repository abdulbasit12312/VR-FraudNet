"""Cost-sensitive operational evaluation (Supplementary Table S1).

The manuscript decomposes operational cost into false-negative fraud loss,
false-positive review cost and true-positive review cost, and compares
VR-FraudNet against LightGBM, TabTransformer and the no-cost-sensitive-loss
ablation.

The cost values themselves are not in the main text (audit finding A-10), so
this module computes the decomposition from an operator-supplied cost matrix and
never from a built-in default. With no cost file, it raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vrfraudnet.losses.cost_sensitive import CostMatrix


@dataclass
class CostDecomposition:
    """Operational cost of one model at one frozen operating threshold."""

    model: str
    dataset: str
    threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    fraud_loss: float
    review_cost_false_positive: float
    review_cost_true_positive: float
    currency: str

    @property
    def total_loss(self) -> float:
        return self.fraud_loss + self.review_cost_false_positive + self.review_cost_true_positive

    @property
    def review_rate(self) -> float:
        n = self.true_positives + self.false_positives + self.false_negatives + self.true_negatives
        return float((self.true_positives + self.false_positives) / n) if n else float("nan")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "operating_threshold": self.threshold,
            "confusion": {
                "tp": self.true_positives,
                "fp": self.false_positives,
                "fn": self.false_negatives,
                "tn": self.true_negatives,
            },
            "fraud_loss": self.fraud_loss,
            "review_cost_false_positive": self.review_cost_false_positive,
            "review_cost_true_positive": self.review_cost_true_positive,
            "total_loss": self.total_loss,
            "review_rate": self.review_rate,
            "currency": self.currency,
        }


def evaluate_cost(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
    costs: CostMatrix,
    model: str,
    dataset: str,
) -> CostDecomposition:
    """Compute the manuscript's three-way cost decomposition."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    return CostDecomposition(
        model=model,
        dataset=dataset,
        threshold=float(threshold),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        fraud_loss=fn * costs.false_negative_fraud_cost,
        review_cost_false_positive=fp * costs.false_positive_review_cost,
        review_cost_true_positive=tp * costs.true_positive_review_cost,
        currency=costs.currency,
    )


def savings_against(
    candidate: CostDecomposition, reference: CostDecomposition
) -> dict[str, float]:
    """Absolute and relative saving of ``candidate`` against ``reference``."""
    absolute = reference.total_loss - candidate.total_loss
    relative = absolute / reference.total_loss if reference.total_loss else float("nan")
    return {
        "absolute_saving": float(absolute),
        "relative_saving": float(relative),
        "candidate_total_loss": float(candidate.total_loss),
        "reference_total_loss": float(reference.total_loss),
    }
