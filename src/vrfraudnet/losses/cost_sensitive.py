"""Cost-sensitive weighting and class-balanced reweighting (manuscript S4.6.1).

Manuscript specification
------------------------
* The cost-sensitive loss is applied to the Stage 1 LightGBM triage classifier.
* It uses "the dataset-specific false-negative fraud cost and false-positive
  review cost reported". For each dataset both monetary costs are divided by the
  larger of the two, preserving their ratio while removing currency scale.
* The true-positive review cost is deliberately excluded from the classification
  penalty (a detected fraud is not a prediction error) and enters only
  validation-based threshold selection and the operational cost analysis.
* The final cost-sensitive weight is 0.50, selected from {0.25, 0.50, 0.75,
  1.00} on validation AUPRC with validation expected operational cost as the
  secondary criterion.

MISSING FROM THE MANUSCRIPT (audit finding A-10)
------------------------------------------------
The actual monetary values of the false-negative fraud cost, false-positive
review cost and true-positive review cost are *not* given anywhere in the main
text; they live in Supplementary Table S1, which was not supplied. Without them
neither the cost-sensitive weights nor Supplementary Table S1 can be
reproduced. :func:`load_cost_matrix` therefore refuses to invent numbers and
raises unless the operator supplies a cost file explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from vrfraudnet.errors import ConfigurationError, MissingArtefactError


@dataclass(frozen=True)
class CostMatrix:
    """Operational costs for one dataset, in that dataset's own currency unit."""

    dataset_id: str
    false_negative_fraud_cost: float
    false_positive_review_cost: float
    true_positive_review_cost: float
    currency: str = "unspecified"
    source: str = "operator-supplied"

    def normalised(self) -> tuple[float, float]:
        """Return (FN, FP) costs divided by the larger of the two (S4.6.1)."""
        scale = max(self.false_negative_fraud_cost, self.false_positive_review_cost)
        if scale <= 0:
            raise ConfigurationError(f"{self.dataset_id}: costs must be positive")
        return (
            self.false_negative_fraud_cost / scale,
            self.false_positive_review_cost / scale,
        )

    def total_loss(self, tp: int, fp: int, fn: int) -> float:
        """Total operational loss under the manuscript's cost decomposition.

        True negatives cost nothing. True positives incur the review cost but
        are not a classification error, which is exactly the distinction the
        manuscript draws in S4.6.1.
        """
        return (
            fn * self.false_negative_fraud_cost
            + fp * self.false_positive_review_cost
            + tp * self.true_positive_review_cost
        )


def load_cost_matrix(dataset_id: str, path: str | Path | None) -> CostMatrix:
    """Load a cost matrix from an operator-supplied YAML file.

    Raises
    ------
    vrfraudnet.errors.MissingArtefactError
        When no cost file is supplied. The manuscript's Supplementary Table S1
        values were not available when this repository was written, and
        fabricating them would make the operational-cost analysis meaningless.
    """
    if path is None:
        raise MissingArtefactError(
            "no cost matrix supplied for "
            f"{dataset_id}. The manuscript's false-negative fraud cost, "
            "false-positive review cost and true-positive review cost are not "
            "stated in the main text (audit finding A-10); they appear only in "
            "Supplementary Table S1.\n"
            "Supply them with --costs configs/costs.yaml after filling in the "
            "placeholder entries, or omit the cost-sensitive analysis."
        )
    path = Path(path)
    if not path.exists():
        raise MissingArtefactError(f"cost file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    entry = (document.get("datasets") or {}).get(dataset_id)
    if not entry:
        raise ConfigurationError(f"{path} contains no cost entry for {dataset_id}")
    for field in (
        "false_negative_fraud_cost",
        "false_positive_review_cost",
        "true_positive_review_cost",
    ):
        if entry.get(field) is None:
            raise ConfigurationError(
                f"{path}: {dataset_id}.{field} is null. Fill in the real value from "
                f"Supplementary Table S1; this repository will not guess it."
            )
    return CostMatrix(
        dataset_id=dataset_id,
        false_negative_fraud_cost=float(entry["false_negative_fraud_cost"]),
        false_positive_review_cost=float(entry["false_positive_review_cost"]),
        true_positive_review_cost=float(entry["true_positive_review_cost"]),
        currency=str(entry.get("currency", "unspecified")),
        source=str(entry.get("source", "operator-supplied")),
    )


def class_balanced_weights(y: np.ndarray, *, beta: float = 0.9999) -> np.ndarray:
    """Class-balanced reweighting (Cui et al., 2019), as named in manuscript Table 2.

    ``w_c = (1 - beta) / (1 - beta^{n_c})``, normalised so the mean weight is 1.
    ``beta`` is not specified by the manuscript; 0.9999 is the value used by the
    original paper for extreme imbalance and is recorded as ASSUMPTION L-12.
    """
    y = np.asarray(y).astype(int)
    weights = np.ones(y.shape[0], dtype=np.float64)
    for label in np.unique(y):
        n_c = int((y == label).sum())
        effective = (1.0 - beta**n_c) / (1.0 - beta) if beta < 1 else float(n_c)
        weights[y == label] = 1.0 / max(effective, 1e-12)
    return weights / weights.mean()


def cost_sensitive_weights(
    y: np.ndarray, costs: CostMatrix, *, cost_weight: float = 0.50
) -> np.ndarray:
    """Blend class-balanced weights with normalised misclassification costs.

    ``w = (1 - lambda) * w_balanced + lambda * w_cost``, with ``lambda`` the
    cost-sensitive weight (manuscript S4.6.1 final value 0.50).
    """
    if not 0.0 <= cost_weight <= 1.0:
        raise ConfigurationError("cost_weight must lie in [0, 1]")
    fn_cost, fp_cost = costs.normalised()
    balanced = class_balanced_weights(y)
    cost = np.where(np.asarray(y) == 1, fn_cost, fp_cost)
    cost = cost / cost.mean()
    blended = (1.0 - cost_weight) * balanced + cost_weight * cost
    return blended / blended.mean()
