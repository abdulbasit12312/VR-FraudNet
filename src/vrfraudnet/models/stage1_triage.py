"""Stage 1: LightGBM triage classifier and threshold gate (manuscript S4.2, Figure 4).

Specification taken from the manuscript
---------------------------------------
* Triage input ``H_i = [x_i ; z_g]`` (Eq. 3); ``z_g`` omitted or replaced by
  past-only temporal aggregates for datasets without an explicit graph.
* ``P_t = T_theta(H_i)``, ``P_t in [0, 1]`` (Eq. 4).
* LightGBM, 500 trees, max depth 8, 64 leaves, learning rate 0.05, min 100
  observations per leaf, feature fraction 0.80, bagging fraction 0.80, bagging
  frequency 1, L1 = 0.10, L2 = 1.00, early stopping after 50 stagnant boosting
  rounds on validation AUPRC (S4.2, S4.8.1).
* Focal-loss surrogate with class-balanced reweighting; no synthetic
  oversampling (S4.2).
* Two validation-tuned thresholds ``T_low < T_high`` per dataset. Selected
  pairs: D1 (0.20, 0.80), D2 (0.15, 0.85), D3 (0.22, 0.78), D4 (0.18, 0.82),
  D5 (0.17, 0.83) (S4.8.1).

AUDIT NOTE (A-05). Equation 5 as printed is malformed. It reads

    D_i = accept,   P_t < T_low
    D_i = reject,   P_t < T_high
    D_i = escalate, T_low <= P_t >= T_high

The second line duplicates the accept condition and the third mixes the
inequality directions. The only routing rule consistent with the surrounding
prose ("Low-risk transactions are accepted, high-risk transactions are
rejected, and intermediate-risk transactions are routed to Stage 2") and with
Figure 4 is implemented here:

    accept    if P_t <  T_low
    escalate  if T_low <= P_t <= T_high
    reject    if P_t >  T_high

The corrected form is stated explicitly so the divergence from the printed
equation is visible rather than silently patched.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np

from vrfraudnet.errors import ConfigurationError


class Route(str, Enum):
    """Routing decision ``D_i`` of manuscript Eq. 5 (corrected form, A-05)."""

    ACCEPT = "accept"
    ESCALATE = "escalate"
    REJECT = "reject"


@dataclass(frozen=True)
class ThresholdGate:
    """The two validation-tuned routing thresholds of manuscript S4.2."""

    t_low: float
    t_high: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.t_low < self.t_high <= 1.0:
            raise ConfigurationError(
                f"require 0 <= t_low < t_high <= 1, got t_low={self.t_low}, t_high={self.t_high}"
            )

    def route(self, probabilities: np.ndarray) -> np.ndarray:
        """Vectorised routing. Returns an array of :class:`Route` values."""
        p = np.asarray(probabilities, dtype=float)
        out = np.full(p.shape, Route.ESCALATE.value, dtype=object)
        out[p < self.t_low] = Route.ACCEPT.value
        out[p > self.t_high] = Route.REJECT.value
        return out

    def escalation_rate(self, probabilities: np.ndarray) -> float:
        """Fraction of transactions sent to the Stage 2 rationale model."""
        routes = self.route(probabilities)
        return float(np.mean(routes == Route.ESCALATE.value))


#: Manuscript S4.8.1: the selected (T_low, T_high) pairs, per dataset.
MANUSCRIPT_THRESHOLDS: dict[str, ThresholdGate] = {
    "D1": ThresholdGate(0.20, 0.80),
    "D2": ThresholdGate(0.15, 0.85),
    "D3": ThresholdGate(0.22, 0.78),
    "D4": ThresholdGate(0.18, 0.82),
    "D5": ThresholdGate(0.17, 0.83),
}


@dataclass
class TriageParams:
    """LightGBM hyperparameters exactly as reported in manuscript S4.2 / S4.8.1."""

    n_estimators: int = 500
    max_depth: int = 8
    num_leaves: int = 64
    learning_rate: float = 0.05
    min_child_samples: int = 100
    feature_fraction: float = 0.80
    bagging_fraction: float = 0.80
    bagging_freq: int = 1
    reg_alpha: float = 0.10
    reg_lambda: float = 1.00
    early_stopping_rounds: int = 50
    num_threads: int = 0  # 0 = LightGBM default; pin it for bitwise determinism

    def to_lightgbm(self, seed: int) -> dict[str, Any]:
        return {
            "objective": "binary",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": self.bagging_fraction,
            "bagging_freq": self.bagging_freq,
            "reg_alpha": self.reg_alpha,
            "reg_lambda": self.reg_lambda,
            "random_state": seed,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": self.num_threads,
            "verbose": -1,
        }


class TriageClassifier:
    """Stage 1 triage model with a focal, class-balanced, cost-weighted objective.

    The manuscript is explicit that this is a *surrogate*: "We phrase this as a
    focal-loss-based objective or surrogate because the implementation must
    remain compatible with the underlying gradient-boosted tree training
    procedure." That is implemented here as a custom LightGBM objective, which is
    the only way to obtain a genuine focal gradient inside GBDT training.
    """

    def __init__(
        self,
        params: TriageParams,
        *,
        seed: int,
        focal_gamma: float = 2.0,
        cost_weight: float = 0.50,
        class_balanced_beta: float = 0.9999,
    ) -> None:
        self.params = params
        self.seed = seed
        self.focal_gamma = focal_gamma
        self.cost_weight = cost_weight
        self.class_balanced_beta = class_balanced_beta
        self.booster_: Any | None = None
        self.feature_names_: list[str] | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_valid: np.ndarray,
        y_valid: np.ndarray,
        *,
        feature_names: Sequence[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "TriageClassifier":
        """Train with early stopping on validation average precision (AUPRC)."""
        import lightgbm as lgb

        from vrfraudnet.losses.focal import focal_objective_factory
        from vrfraudnet.losses.cost_sensitive import class_balanced_weights

        weights = (
            sample_weight
            if sample_weight is not None
            else class_balanced_weights(y_train, beta=self.class_balanced_beta)
        )

        train_set = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=False)
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)

        params = self.params.to_lightgbm(self.seed)
        n_estimators = params.pop("n_estimators")
        params["objective"] = focal_objective_factory(gamma=self.focal_gamma)

        self.booster_ = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[valid_set],
            valid_names=["validation"],
            feval=_auprc_eval,
            callbacks=[
                lgb.early_stopping(self.params.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        self.feature_names_ = list(feature_names) if feature_names is not None else None
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return ``P_t`` in [0, 1] (Eq. 4).

        A custom objective makes LightGBM emit raw margins, so the sigmoid is
        applied here explicitly rather than assumed.
        """
        if self.booster_ is None:
            raise RuntimeError("TriageClassifier.predict_proba called before fit")
        raw = self.booster_.predict(x, raw_score=True)
        return 1.0 / (1.0 + np.exp(-raw))


def _auprc_eval(preds: np.ndarray, dataset: Any) -> tuple[str, float, bool]:
    """LightGBM custom metric: average precision on the fraud class."""
    from sklearn.metrics import average_precision_score

    labels = dataset.get_label()
    probabilities = 1.0 / (1.0 + np.exp(-preds))
    if len(np.unique(labels)) < 2:
        return "auprc", 0.0, True
    return "auprc", float(average_precision_score(labels, probabilities)), True


def select_thresholds(
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    *,
    max_escalation_rate: float,
    candidate_lows: Sequence[float] | None = None,
    candidate_highs: Sequence[float] | None = None,
) -> ThresholdGate:
    """Select ``(T_low, T_high)`` on validation data only (manuscript S4.2, S4.8.1).

    Selection criterion, per S4.8.1: "Threshold selection prioritized validation
    AUPRC and expected operational cost while limiting the proportion of
    transactions sent to the language-model pathway." Because AUPRC is
    threshold-free, the routing thresholds cannot change it; the operative
    criteria are therefore (a) the escalation-rate cap and (b) the fraction of
    fraud captured inside the review band, which is what a reviewer would call
    review-band informativeness. That reading is stated here explicitly, and the
    manuscript's own per-dataset values remain available in
    :data:`MANUSCRIPT_THRESHOLDS` for exact reproduction. See L-11.
    """
    lows = candidate_lows or np.round(np.arange(0.05, 0.51, 0.01), 2)
    highs = candidate_highs or np.round(np.arange(0.50, 0.96, 0.01), 2)

    best: tuple[float, ThresholdGate] | None = None
    for t_low in lows:
        for t_high in highs:
            if t_low >= t_high:
                continue
            gate = ThresholdGate(float(t_low), float(t_high))
            band = (validation_probabilities >= t_low) & (validation_probabilities <= t_high)
            rate = float(band.mean())
            if rate > max_escalation_rate:
                continue
            captured = float(validation_labels[band].sum()) / max(
                1.0, float(validation_labels.sum())
            )
            score = captured / max(rate, 1e-9)
            if best is None or score > best[0]:
                best = (score, gate)

    if best is None:
        raise ConfigurationError(
            f"no (T_low, T_high) pair satisfies the escalation cap "
            f"{max_escalation_rate}; widen the candidate grid or raise the cap"
        )
    return best[1]
