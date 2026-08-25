"""Baseline models (manuscript Section 4.7).

    "The baseline suite consists of Logistic Regression, Isolation Forest, MLP,
     1D-CNN, LSTM, Random Forest, XGBoost, LightGBM, and TabTransformer. All
     baselines are evaluated under the same dataset splits, preprocessing rules,
     leakage-mitigation protocol, and metric definitions used for VR-FraudNet."

Hyperparameters
---------------
The manuscript specifies hyperparameters only for the Stage 1 LightGBM triage
model. It gives none for Logistic Regression, Isolation Forest, MLP, 1D-CNN,
LSTM, Random Forest, XGBoost or TabTransformer. Every baseline setting below is
therefore an ASSUMPTION (docs/KNOWN_LIMITATIONS.md L-18) and is exposed in
``configs/baselines.yaml`` so it can be corrected once the authors supply the
real values. Nothing here should be read as reproducing the manuscript's
baseline numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np

from vrfraudnet.errors import ConfigurationError, MissingArtefactError

#: Baseline names in the manuscript's own order (Section 4.7, Table 5).
BASELINE_ORDER: tuple[str, ...] = (
    "logistic_regression",
    "isolation_forest",
    "mlp",
    "cnn1d",
    "lstm",
    "random_forest",
    "xgboost",
    "lightgbm",
    "tabtransformer",
)

#: Baselines whose training is stochastic and therefore seeded per run.
STOCHASTIC_BASELINES: frozenset[str] = frozenset(
    {"isolation_forest", "mlp", "cnn1d", "lstm", "random_forest", "xgboost", "lightgbm",
     "tabtransformer"}
)


class ScoringModel(Protocol):
    """Minimal interface every baseline exposes."""

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: Any) -> "ScoringModel": ...

    def score_samples(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class BaselineSpec:
    """A baseline together with the provenance of its hyperparameters."""

    name: str
    builder: Callable[..., Any]
    params: dict[str, Any]
    provenance: str
    requires: tuple[str, ...] = ()


def build_baseline(name: str, *, seed: int, params: dict[str, Any] | None = None) -> Any:
    """Instantiate one baseline by name."""
    key = name.lower()
    if key not in BASELINE_ORDER:
        raise ConfigurationError(f"unknown baseline {name!r}; expected one of {BASELINE_ORDER}")
    params = dict(params or {})

    if key == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            max_iter=params.pop("max_iter", 1000),
            class_weight=params.pop("class_weight", "balanced"),
            n_jobs=None,
            random_state=seed,
            **params,
        )
    if key == "isolation_forest":
        from sklearn.ensemble import IsolationForest

        return IsolationForest(
            n_estimators=params.pop("n_estimators", 200),
            contamination=params.pop("contamination", "auto"),
            random_state=seed,
            **params,
        )
    if key == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=params.pop("n_estimators", 500),
            max_depth=params.pop("max_depth", None),
            min_samples_leaf=params.pop("min_samples_leaf", 100),
            class_weight=params.pop("class_weight", "balanced_subsample"),
            n_jobs=params.pop("n_jobs", -1),
            random_state=seed,
            **params,
        )
    if key == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise MissingArtefactError("xgboost is required for this baseline") from exc

        return XGBClassifier(
            n_estimators=params.pop("n_estimators", 500),
            max_depth=params.pop("max_depth", 8),
            learning_rate=params.pop("learning_rate", 0.05),
            subsample=params.pop("subsample", 0.8),
            colsample_bytree=params.pop("colsample_bytree", 0.8),
            reg_alpha=params.pop("reg_alpha", 0.1),
            reg_lambda=params.pop("reg_lambda", 1.0),
            eval_metric="aucpr",
            tree_method="hist",
            random_state=seed,
            **params,
        )
    if key == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=params.pop("n_estimators", 500),
            max_depth=params.pop("max_depth", 8),
            num_leaves=params.pop("num_leaves", 64),
            learning_rate=params.pop("learning_rate", 0.05),
            min_child_samples=params.pop("min_child_samples", 100),
            subsample=params.pop("subsample", 0.8),
            colsample_bytree=params.pop("colsample_bytree", 0.8),
            reg_alpha=params.pop("reg_alpha", 0.1),
            reg_lambda=params.pop("reg_lambda", 1.0),
            random_state=seed,
            deterministic=True,
            verbose=-1,
            **params,
        )
    if key in {"mlp", "cnn1d", "lstm", "tabtransformer"}:
        from vrfraudnet.baselines.neural import build_neural_baseline

        return build_neural_baseline(key, seed=seed, **params)

    raise ConfigurationError(f"no builder registered for {name!r}")  # pragma: no cover


def score_of(model: Any, x: np.ndarray) -> np.ndarray:
    """Extract a continuous fraud score from any baseline.

    Isolation Forest is unsupervised and returns an anomaly score whose sign is
    inverted relative to a fraud probability; the conversion is made explicit
    here rather than hidden in the evaluation loop.
    """
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x))[:, 1]
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(x), dtype=float)
        # IsolationForest: lower decision_function means more anomalous.
        return -raw
    if hasattr(model, "score_samples"):
        return -np.asarray(model.score_samples(x), dtype=float)
    raise ConfigurationError(f"{type(model).__name__} exposes no scoring interface")
