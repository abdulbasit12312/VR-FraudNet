"""Cross-dataset transfer evaluation (manuscript Section 5.4, Table 8).

What the manuscript reports
---------------------------
A 5x5 zero-shot AUPRC matrix for VR-FraudNet, a 5x5 few-shot matrix using 1% of
target labels, and a 5x5 zero-shot matrix for TabTransformer.

AUDIT FINDING A-14: transfer across heterogeneous schemas is under-specified
---------------------------------------------------------------------------
The five benchmarks do not share a feature space, a label semantics, or a graph
semantics:

===  ===========================================  ==========================
D1   30 account-application attributes            account-opening fraud
D2   9 graph attributes + entity metadata         money-laundering typology
D3   109 anonymised transaction/identity columns  card-payment fraud
D4   94 anonymised transaction features           illicit Bitcoin flows
D5   17 anonymised node attributes                P2P borrower default risk
===  ===========================================  ==========================

A model trained on D1 has a first layer (or a set of tree splits) defined over
D1's 30 columns. Applying it to D3's 109 columns is not defined until an
alignment is specified. The manuscript describes no alignment mechanism - no
shared encoder, no column-name matching, no feature-space projection, no
adapter. It is therefore impossible to say what operation produced Table 8, and
the numbers cannot be regenerated from the paper alone.

This module refuses to invent an alignment. It requires an explicit
:class:`AlignmentStrategy` and records which one was used in the result file, so
any transfer number this repository produces is interpretable. Running with
``strategy=None`` raises :class:`~vrfraudnet.errors.ManuscriptInconsistencyError`.

A second, independent problem: D5 DGraph-Fin's label is *loan default*, not
fraud in the sense of D1-D4. Transferring a fraud detector to a default-risk
target is a different task, not a harder instance of the same one. That is
recorded as audit note A-15.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from vrfraudnet.errors import ConfigurationError, ManuscriptInconsistencyError


class AlignmentStrategy(str, Enum):
    """How a source model is made applicable to a target feature space."""

    #: Match columns by name; unmatched target columns are zero-filled and
    #: unmatched source columns are dropped. Only meaningful when the two
    #: datasets genuinely share column names (none of D1-D5 do).
    COLUMN_NAME = "column_name"

    #: Project both datasets onto a fixed-width space of rank-normalised
    #: features ordered by training-set mutual information with the label.
    #: Purely positional, so it is defined for any pair, and it is honest about
    #: being an alignment invented by this repository rather than by the paper.
    RANK_PROJECTION = "rank_projection"

    #: Refit only the final decision layer / a fresh head on target labels.
    #: This is the only strategy that makes the *few-shot* column of Table 8(b)
    #: well defined.
    HEAD_REFIT = "head_refit"


@dataclass
class TransferCell:
    """One (source, target) entry of a transfer matrix."""

    source: str
    target: str
    auprc: float
    strategy: str
    n_target_labels_used: int
    is_diagonal: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "auprc": self.auprc,
            "alignment_strategy": self.strategy,
            "n_target_labels_used": self.n_target_labels_used,
            "is_diagonal": self.is_diagonal,
            "note": self.note,
        }


def require_alignment_strategy(strategy: AlignmentStrategy | None) -> AlignmentStrategy:
    """Refuse to run transfer without an explicitly declared alignment."""
    if strategy is None:
        raise ManuscriptInconsistencyError(
            "A-14",
            "cross-dataset transfer requires a feature-space alignment, and the "
            "manuscript specifies none. D1-D5 share no columns, no feature "
            "semantics and no graph semantics, so 'apply the D1 model to D3' has "
            "no defined meaning. Pass --alignment {column_name|rank_projection|"
            "head_refit} to declare one, and it will be recorded in the result "
            "file alongside every number it produces.",
        )
    return strategy


def rank_project(
    x: np.ndarray, *, width: int, order: np.ndarray | None = None
) -> np.ndarray:
    """Project a feature matrix onto ``width`` rank-normalised columns.

    Each retained column is mapped to its within-column empirical rank in
    ``[0, 1]``, which removes scale and unit differences between datasets.
    Columns are selected by ``order`` (computed on the *source training* set) and
    zero-padded when the target has fewer columns.
    """
    x = np.asarray(x, dtype=float)
    if order is not None:
        keep = [i for i in order.tolist() if i < x.shape[1]][:width]
    else:
        keep = list(range(min(width, x.shape[1])))
    out = np.zeros((x.shape[0], width), dtype=float)
    for j, column in enumerate(keep):
        values = x[:, column]
        ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
        out[:, j] = ranks / max(1, values.size - 1)
    return out


def mutual_information_order(x: np.ndarray, y: np.ndarray, *, seed: int = 0) -> np.ndarray:
    """Column indices ordered by mutual information with the label, descending."""
    from sklearn.feature_selection import mutual_info_classif

    scores = mutual_info_classif(
        np.nan_to_num(np.asarray(x, dtype=float)), np.asarray(y).astype(int), random_state=seed
    )
    return np.argsort(-scores, kind="stable")


def few_shot_subset(
    y: np.ndarray, *, fraction: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a stratified ``fraction`` of target labels (manuscript S5.4: 1%).

    Stratified so that a 1% sample of a 0.10%-prevalence dataset still contains
    positives; an unstratified 1% of D2 would often contain none, which would
    make the few-shot column meaningless.
    """
    if not 0 < fraction < 1:
        raise ConfigurationError("few-shot fraction must lie in (0, 1)")
    y = np.asarray(y).astype(int)
    chosen: list[int] = []
    for label in (0, 1):
        idx = np.flatnonzero(y == label)
        k = max(1, int(round(fraction * idx.size)))
        chosen.extend(rng.choice(idx, size=min(k, idx.size), replace=False).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def off_diagonal_mean(matrix: Mapping[str, Mapping[str, float]]) -> float:
    """Grand mean of the off-diagonal entries of a transfer matrix."""
    values = [
        float(v)
        for source, row in matrix.items()
        for target, v in row.items()
        if source != target and np.isfinite(v)
    ]
    if not values:
        return float("nan")
    return float(np.mean(values))


def per_row_off_diagonal_means(
    matrix: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:
    """Per-source off-diagonal means, as reported in manuscript S5.4."""
    out: dict[str, float] = {}
    for source, row in matrix.items():
        values = [float(v) for t, v in row.items() if t != source and np.isfinite(v)]
        out[source] = float(np.mean(values)) if values else float("nan")
    return out


def build_transfer_matrix(
    sources: Sequence[str],
    targets: Sequence[str],
    evaluate: Callable[[str, str], TransferCell],
) -> dict[str, dict[str, TransferCell]]:
    """Fill a transfer matrix by calling ``evaluate`` for every (source, target)."""
    return {s: {t: evaluate(s, t) for t in targets} for s in sources}
