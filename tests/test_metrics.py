"""Metric correctness, including the Recall@top-1% ceiling (audit A-01)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from vrfraudnet.data.registry import get_spec
from vrfraudnet.evaluation.metrics import (
    aggregate_folds,
    aggregate_seeds,
    auprc,
    compute_all,
    fraud_class_f1,
    recall_at_top_k,
    recall_at_top_k_ceiling,
    roc_auc,
    select_operating_threshold,
)


@pytest.fixture
def scored():
    rng = np.random.default_rng(7)
    y = (rng.uniform(size=2000) < 0.05).astype(int)
    scores = rng.uniform(size=2000) + 0.4 * y
    return y, scores


def test_auprc_matches_sklearn(scored):
    y, scores = scored
    assert auprc(y, scores) == pytest.approx(average_precision_score(y, scores))


def test_roc_auc_matches_sklearn(scored):
    y, scores = scored
    assert roc_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_f1_is_the_fraud_class_not_macro():
    y = np.array([0, 0, 0, 1])
    pred = np.array([0, 0, 1, 1])
    # fraud-class F1 = 2*P*R/(P+R) with P=0.5, R=1.0 -> 0.6667
    assert fraud_class_f1(y, pred) == pytest.approx(2 / 3)


def test_perfect_ranking_reaches_the_recall_ceiling():
    n = 1000
    y = np.zeros(n, dtype=int)
    y[:100] = 1  # 10% prevalence -> ceiling 0.10
    scores = np.zeros(n)
    scores[:100] = 1.0
    value = recall_at_top_k(y, scores, k_fraction=0.01)
    assert value == pytest.approx(recall_at_top_k_ceiling(0.10), abs=1e-6)


def test_recall_at_top_k_never_exceeds_the_ceiling():
    rng = np.random.default_rng(3)
    for prevalence in (0.001, 0.011, 0.0223, 0.035, 0.10):
        n = 20_000
        y = (rng.uniform(size=n) < prevalence).astype(int)
        if y.sum() == 0:
            continue
        scores = rng.uniform(size=n) + y  # near-perfect ranking
        value = recall_at_top_k(y, scores)
        assert value <= recall_at_top_k_ceiling(float(y.mean())) + 1e-9


def test_d3_manuscript_recall_values_are_above_the_ceiling():
    """Audit finding A-01, expressed as an executable assertion."""
    spec = get_spec("D3")
    ceiling = spec.recall_at_top_1pct_ceiling
    assert ceiling == pytest.approx(0.01 / 0.035)
    manuscript_values = [0.4234, 0.4567, 0.5234, 0.5345, 0.5478, 0.5612, 0.5812,
                         0.5891, 0.6034, 0.6234]
    assert all(v > ceiling for v in manuscript_values)


def test_ceilings_for_the_other_datasets_do_not_bind():
    for dataset in ("D1", "D2"):
        assert get_spec(dataset).recall_at_top_1pct_ceiling >= 0.9


def test_threshold_is_selected_on_validation_only(scored):
    y, scores = scored
    threshold = select_operating_threshold(y, scores)
    assert 0.0 <= threshold <= 1.5
    # Applying the frozen threshold elsewhere must not re-tune it.
    metrics = compute_all(y, scores, threshold=threshold)
    assert metrics.threshold == pytest.approx(threshold)


def test_compute_all_warns_when_the_recall_ceiling_binds():
    rng = np.random.default_rng(11)
    n = 5000
    y = (rng.uniform(size=n) < 0.035).astype(int)
    scores = rng.uniform(size=n) + 0.5 * y
    metrics = compute_all(y, scores, threshold=0.7)
    assert any("A-01" in w for w in metrics.warnings)


def test_metrics_reject_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        auprc(np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]))


def test_metrics_reject_single_class():
    with pytest.raises(ValueError, match="both classes"):
        auprc(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_metrics_reject_non_finite_scores():
    with pytest.raises(ValueError, match="NaN or infinity"):
        auprc(np.array([0, 1]), np.array([0.5, np.inf]))


def test_aggregate_folds_averages_fold_values(scored):
    y, scores = scored
    folds = [compute_all(y, scores, threshold=0.7) for _ in range(3)]
    aggregated = aggregate_folds(folds)
    assert aggregated["auprc"] == pytest.approx(folds[0].auprc)
    assert aggregated["n_folds"] == 3


def test_aggregate_seeds_uses_the_sample_standard_deviation():
    per_seed = {42: {"auprc": 0.50}, 123: {"auprc": 0.60}}
    aggregated = aggregate_seeds(per_seed)
    assert aggregated["auprc"]["mean"] == pytest.approx(0.55)
    # ddof=1 sample std of [0.5, 0.6] is 0.0707..., not the population 0.05
    assert aggregated["auprc"]["std"] == pytest.approx(0.0707107, abs=1e-6)


def test_aggregate_seeds_requires_at_least_two_seeds():
    with pytest.raises(ValueError, match="at least two seeds"):
        aggregate_seeds({42: {"auprc": 0.5}})
