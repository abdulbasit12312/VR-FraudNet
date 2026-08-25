"""Stage 4: isotonic probability mixer and split-conformal calibration (S4.5).

Manuscript specification
------------------------
* The mixer combines the Stage 1 triage probability, the Stage 2 rationale-side
  probability and the Stage 3 verifier status.
* The rationale-side probability contributes **only** when the verifier accepts
  the rationale. On rejection the language-model signal is excluded from
  automated use and the rejection status itself is retained as a mixer input.
* The mixer is fitted with isotonic regression on validation data. Routing
  thresholds, mixer parameters and calibration settings are fixed before final
  evaluation. No test labels are used.
* Split-conformal calibration is then performed on a separate held-out
  calibration set; nonconformity scores derive from the probability assigned to
  the observed class, and the conformal threshold follows the predefined
  significance level (primary 0.05, secondary sensitivity 0.10).
* A verifier-aware variant assigns greater uncertainty to verifier-rejected
  cases; a drift-weighted variant may be applied when temporal change is
  expected.
* The marginal-coverage statement applies to the split-conformal layer only,
  under exchangeability. It certifies nothing about the rest of the framework.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vrfraudnet.errors import ConfigurationError, LeakageError


@dataclass
class MixerInputs:
    """The three signals the Stage 4 mixer consumes."""

    triage_probability: np.ndarray
    rationale_probability: np.ndarray | None
    verifier_status: np.ndarray | None

    def __post_init__(self) -> None:
        n = self.triage_probability.shape[0]
        for name in ("rationale_probability", "verifier_status"):
            arr = getattr(self, name)
            if arr is not None and arr.shape[0] != n:
                raise ConfigurationError(f"{name} length {arr.shape[0]} != triage length {n}")

    def gated_rationale(self) -> np.ndarray:
        """Rationale probability with verifier-rejected entries zeroed out.

        This is the operational meaning of "Verifier-rejected rationales are
        excluded from automated use": a rejected rationale contributes no
        probability mass, only its rejection flag.
        """
        n = self.triage_probability.shape[0]
        if self.rationale_probability is None or self.verifier_status is None:
            return np.zeros(n, dtype=float)
        return np.where(self.verifier_status == 1, self.rationale_probability, 0.0)

    def design_matrix(self) -> np.ndarray:
        """Feature matrix ``[p_triage, gated_rationale, verifier_rejected]``."""
        n = self.triage_probability.shape[0]
        rejected = (
            np.zeros(n, dtype=float)
            if self.verifier_status is None
            else (self.verifier_status == 0).astype(float)
        )
        return np.column_stack([self.triage_probability, self.gated_rationale(), rejected])


class IsotonicMixer:
    """Monotone combination of the verified predictive signals (manuscript S4.5).

    A single isotonic regression is monotone in one variable, so the mixer is
    built in two steps: a linear blend of the three signals produces a scalar
    score, and isotonic regression maps that score onto calibrated probability.
    Monotonicity of the final map is preserved, which is what the manuscript's
    "monotonic calibration method" requires. The blend weights are fitted on
    validation data by non-negative least squares, so no weight can invert the
    direction of a signal.
    """

    def __init__(self) -> None:
        self.weights_: np.ndarray | None = None
        self.isotonic_: Any | None = None
        self._fitted_on: str | None = None

    def fit(self, inputs: MixerInputs, y: np.ndarray, *, partition: str = "validation") -> "IsotonicMixer":
        if partition == "test":
            raise LeakageError(
                "the Stage 4 mixer must be fitted on validation data; manuscript S4.5 "
                "states explicitly that no test labels are used"
            )
        from scipy.optimize import nnls
        from sklearn.isotonic import IsotonicRegression

        design = inputs.design_matrix()
        weights, _ = nnls(design, y.astype(float))
        if not np.any(weights):
            weights = np.array([1.0, 0.0, 0.0])
        self.weights_ = weights
        scores = design @ weights
        self.isotonic_ = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.isotonic_.fit(scores, y.astype(float))
        self._fitted_on = partition
        return self

    def predict(self, inputs: MixerInputs) -> np.ndarray:
        if self.weights_ is None or self.isotonic_ is None:
            raise RuntimeError("IsotonicMixer.predict called before fit")
        scores = inputs.design_matrix() @ self.weights_
        return np.clip(self.isotonic_.predict(scores), 0.0, 1.0)


@dataclass
class ConformalResult:
    """Outcome of split-conformal calibration."""

    alpha: float
    threshold: float
    empirical_miscoverage: float
    n_calibration: int
    verifier_aware: bool

    @property
    def nominal_coverage(self) -> float:
        return 1.0 - self.alpha

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "nominal_coverage": self.nominal_coverage,
            "conformal_threshold": self.threshold,
            "empirical_miscoverage": self.empirical_miscoverage,
            "n_calibration": self.n_calibration,
            "verifier_aware": self.verifier_aware,
        }


class SplitConformalCalibrator:
    """Split-conformal prediction sets over calibrated fraud probabilities.

    Nonconformity score: ``s_i = 1 - p_hat(y_i | x_i)``, the complement of the
    probability assigned to the observed class (manuscript S4.5). The conformal
    threshold is the ``ceil((n + 1)(1 - alpha)) / n`` empirical quantile of the
    calibration scores, which is the finite-sample-valid quantile for
    exchangeable data.

    The verifier-aware variant inflates the score of verifier-rejected cases by
    a fixed penalty, so those cases receive wider prediction sets. The penalty
    magnitude is not stated by the manuscript (ASSUMPTION L-16).
    """

    def __init__(self, alpha: float = 0.05, *, verifier_penalty: float = 0.10) -> None:
        if not 0.0 < alpha < 1.0:
            raise ConfigurationError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.verifier_penalty = float(verifier_penalty)
        self.threshold_: float | None = None
        self.verifier_aware_: bool = False

    def calibrate(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        verifier_status: np.ndarray | None = None,
        drift_weights: np.ndarray | None = None,
    ) -> ConformalResult:
        """Fit the conformal threshold on a held-out calibration partition."""
        scores = self.nonconformity(probabilities, labels, verifier_status=verifier_status)
        n = scores.shape[0]
        if n < 20:
            raise ConfigurationError(
                f"split-conformal calibration with n={n} gives a meaningless quantile; "
                "at least 20 calibration points are required"
            )
        if drift_weights is None:
            level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
            threshold = float(np.quantile(scores, level, method="higher"))
        else:
            threshold = float(_weighted_quantile(scores, drift_weights, 1.0 - self.alpha))

        self.threshold_ = threshold
        self.verifier_aware_ = verifier_status is not None
        miscoverage = float(np.mean(scores > threshold))
        return ConformalResult(
            alpha=self.alpha,
            threshold=threshold,
            empirical_miscoverage=miscoverage,
            n_calibration=n,
            verifier_aware=self.verifier_aware_,
        )

    def nonconformity(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        verifier_status: np.ndarray | None = None,
    ) -> np.ndarray:
        """``1 - p(observed class)``, optionally inflated for rejected rationales."""
        p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        y = np.asarray(labels).astype(int)
        scores = 1.0 - np.where(y == 1, p, 1.0 - p)
        if verifier_status is not None:
            scores = scores + self.verifier_penalty * (np.asarray(verifier_status) == 0)
        return scores

    def prediction_set(
        self, probability: float, *, verifier_status: int | None = None
    ) -> set[int]:
        """Return the conformal prediction set for one transaction.

        A label is included when its nonconformity score does not exceed the
        calibrated threshold. Sets of size 2 signal genuine ambiguity, which is
        exactly the case an escalation policy should act on.
        """
        if self.threshold_ is None:
            raise RuntimeError("SplitConformalCalibrator.prediction_set called before calibrate")
        penalty = (
            self.verifier_penalty if (verifier_status is not None and verifier_status == 0) else 0.0
        )
        out: set[int] = set()
        if (1.0 - probability) + penalty <= self.threshold_:
            out.add(1)
        if probability + penalty <= self.threshold_:
            out.add(0)
        return out

    def evaluate_coverage(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
        *,
        verifier_status: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Empirical coverage and average set size on an evaluation partition."""
        if self.threshold_ is None:
            raise RuntimeError("evaluate_coverage called before calibrate")
        scores = self.nonconformity(probabilities, labels, verifier_status=verifier_status)
        covered = scores <= self.threshold_
        sizes = [
            len(
                self.prediction_set(
                    float(p),
                    verifier_status=None if verifier_status is None else int(verifier_status[i]),
                )
            )
            for i, p in enumerate(np.asarray(probabilities, dtype=float))
        ]
        return {
            "empirical_coverage": float(np.mean(covered)),
            "empirical_miscoverage": float(1.0 - np.mean(covered)),
            "nominal_coverage": 1.0 - self.alpha,
            "mean_prediction_set_size": float(np.mean(sizes)),
        }


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Weighted empirical quantile used by the drift-weighted conformal variant."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], np.asarray(weights, dtype=float)[order]
    cumulative = np.cumsum(w) / max(float(w.sum()), 1e-12)
    idx = int(np.searchsorted(cumulative, level, side="left"))
    return float(v[min(idx, v.size - 1)])
