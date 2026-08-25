"""Temporal drift and pre/post-shock evaluation (manuscript Section 5.5, Table 9).

Three protocols
---------------
* **D1 controlled month drift** (Table 9a): AUPRC on test month 6 and month 7,
  with degradation measured in percentage points against a reference.
* **D3 expanding-window CV** (Table 9b): five folds; late-fold degradation is
  Fold 5 minus Fold 3.
* **D4 pre/post shock** (Table 9c): degradation from a reference to the
  pre-shock test window, to the post-shock test window, and the shock-impact
  component (post minus pre).

AUDIT FINDING A-16: the D1 reference column is mislabelled
-----------------------------------------------------------
Table 9(a) heads its reference column "Train AUPRC" and gives 0.5247 for the
full model - which is exactly the value Table 5(a) reports as the **held-out
test** AUPRC on D1. Two consequences:

1. If 0.5247 is the training-period AUPRC, then Table 5(a) is reporting a
   training figure as its primary result.
2. If 0.5247 is the test figure, then Table 9(a) is measuring "drift" from the
   test result to a subset of the same test partition, which is not drift.

The arithmetic settles it: D1's test partition is months 6-7 (Table 1), so the
Table 5(a) value should equal the combined month-6/month-7 performance. The
mean of the Table 9(a) month values is (0.5089 + 0.4945)/2 = 0.5017, not 0.5247.
The three numbers cannot all be right.

This module therefore requires the reference partition to be named explicitly
(:class:`DriftReference`) and records it in the result file. It will not emit a
"drift" figure whose baseline is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from vrfraudnet.errors import ConfigurationError

#: Manuscript S5.5: "The predeclared drift target is 5 pp under the evaluated
#: protocols".
DRIFT_TARGET_PP: float = 5.0


class DriftReference(str, Enum):
    """Which partition the degradation is measured against."""

    TRAINING_PERIOD = "training_period"
    VALIDATION_PERIOD = "validation_period"
    FIRST_TEST_WINDOW = "first_test_window"


@dataclass
class DriftResult:
    """Degradation of one model under one temporal protocol."""

    model: str
    dataset: str
    reference_partition: str
    reference_auprc: float
    windows: dict[str, float]
    deltas_pp: dict[str, float]
    target_pp: float = DRIFT_TARGET_PP

    @property
    def within_target(self) -> dict[str, bool]:
        """Whether each window's degradation stays inside the 5 pp target."""
        return {k: (-v) <= self.target_pp for k, v in self.deltas_pp.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "reference_partition": self.reference_partition,
            "reference_auprc": self.reference_auprc,
            "windows": self.windows,
            "deltas_pp": self.deltas_pp,
            "target_pp": self.target_pp,
            "within_target": self.within_target,
        }


def month_drift(
    model: str,
    reference_auprc: float,
    per_month_auprc: Mapping[int, float],
    *,
    reference: DriftReference,
) -> DriftResult:
    """D1 controlled month drift (manuscript Table 9a)."""
    deltas = {
        f"m{month}": float((value - reference_auprc) * 100.0)
        for month, value in sorted(per_month_auprc.items())
    }
    return DriftResult(
        model=model,
        dataset="D1",
        reference_partition=reference.value,
        reference_auprc=float(reference_auprc),
        windows={f"m{k}": float(v) for k, v in sorted(per_month_auprc.items())},
        deltas_pp=deltas,
    )


def late_fold_degradation(fold_auprc: Sequence[float]) -> float:
    """D3 late-fold degradation, Fold 5 minus Fold 3, in percentage points."""
    if len(fold_auprc) < 5:
        raise ConfigurationError(
            f"expanding-window degradation needs 5 folds, received {len(fold_auprc)}"
        )
    return float((fold_auprc[4] - fold_auprc[2]) * 100.0)


def expanding_window_summary(model: str, fold_auprc: Sequence[float]) -> dict[str, Any]:
    """Per-fold AUPRC, mean, and the late-fold delta (manuscript Table 9b)."""
    values = [float(v) for v in fold_auprc]
    return {
        "model": model,
        "dataset": "D3",
        "folds": {f"fold{i + 1}": v for i, v in enumerate(values)},
        "mean": float(np.mean(values)),
        "late_delta_f5_f3_pp": late_fold_degradation(values),
    }


@dataclass
class ShockResult:
    """D4 pre/post-shock decomposition (manuscript Table 9c)."""

    model: str
    reference_partition: str
    reference_auprc: float
    pre_shock_auprc: float
    post_shock_auprc: float

    @property
    def delta_pre_pp(self) -> float:
        return float((self.pre_shock_auprc - self.reference_auprc) * 100.0)

    @property
    def delta_post_pp(self) -> float:
        return float((self.post_shock_auprc - self.reference_auprc) * 100.0)

    @property
    def shock_impact_pp(self) -> float:
        """The additional degradation attributable to the shock itself."""
        return float((self.post_shock_auprc - self.pre_shock_auprc) * 100.0)

    @property
    def exceeds_target(self) -> bool:
        """Whether absolute post-shock degradation breaches the 5 pp target.

        The manuscript is candid about this for its own model: D4 post-shock
        AUPRC falls from 0.6823 to 0.6087, a 7.36 pp absolute reduction that
        exceeds the target. This property reproduces that judgement mechanically.
        """
        return (-self.delta_post_pp) > DRIFT_TARGET_PP

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "dataset": "D4",
            "reference_partition": self.reference_partition,
            "reference_auprc": self.reference_auprc,
            "pre_shock_auprc": self.pre_shock_auprc,
            "post_shock_auprc": self.post_shock_auprc,
            "delta_pre_pp": self.delta_pre_pp,
            "delta_post_pp": self.delta_post_pp,
            "shock_impact_pp": self.shock_impact_pp,
            "target_pp": DRIFT_TARGET_PP,
            "exceeds_target": self.exceeds_target,
        }


def split_shock_windows(
    timesteps: np.ndarray, test_idx: np.ndarray, *, shock_timestep: int
) -> tuple[np.ndarray, np.ndarray]:
    """Partition the D4 test window into pre-shock and post-shock indices.

    The manuscript refers to "a pre/post dark-market-shutdown shock
    decomposition" but never names the timestep at which the shock occurs. The
    Elliptic literature places the dark-market shutdown at timestep 43; that
    value is an ASSUMPTION here (L-19) and must be supplied explicitly.
    """
    if shock_timestep is None:
        raise ConfigurationError(
            "the D4 shock timestep is not stated in the manuscript; supply it with "
            "--shock-timestep (the Elliptic literature places the dark-market "
            "shutdown at timestep 43; see docs/KNOWN_LIMITATIONS.md L-19)"
        )
    steps = timesteps[test_idx]
    pre = test_idx[steps < shock_timestep]
    post = test_idx[steps >= shock_timestep]
    if pre.size == 0 or post.size == 0:
        raise ConfigurationError(
            f"shock timestep {shock_timestep} does not split the test window "
            f"(pre={pre.size}, post={post.size})"
        )
    return pre, post
