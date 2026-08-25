"""Statistical tests: Wilcoxon, DeLong, Holm, and the effect-size estimator."""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.statistics.delong import delong_test, fast_delong
from vrfraudnet.statistics.holm import bootstrap_ci, holm_bonferroni
from vrfraudnet.statistics.wilcoxon import (
    cliffs_delta,
    paired_wilcoxon,
    pooled_and_per_dataset,
)
from vrfraudnet.errors import ConfigurationError
from vrfraudnet.seeds import wilcoxon_max_statistic


def test_wilcoxon_max_statistic_matches_the_manuscript():
    assert wilcoxon_max_statistic(30) == 465


def test_paired_wilcoxon_detects_a_consistent_improvement():
    model = [0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]
    baseline = [v - 0.02 for v in model]
    result = paired_wilcoxon(model, baseline, baseline_name="b", scope="test")
    assert result.mean_delta_pp == pytest.approx(2.0)
    assert result.statistic_v == wilcoxon_max_statistic(8)
    assert result.p_value_one_sided < 0.01


def test_paired_wilcoxon_refuses_tiny_samples():
    with pytest.raises(ValueError, match="cannot reach conventional significance"):
        paired_wilcoxon([0.1, 0.2], [0.0, 0.1], baseline_name="b", scope="s")


def test_paired_wilcoxon_requires_matching_shapes():
    with pytest.raises(ValueError, match="identical shape"):
        paired_wilcoxon([0.1] * 8, [0.1] * 7, baseline_name="b", scope="s")


def test_cliffs_delta_bounds():
    assert cliffs_delta([1, 2, 3], [0, 0, 0]) == pytest.approx(1.0)
    assert cliffs_delta([0, 0, 0], [1, 2, 3]) == pytest.approx(-1.0)
    assert cliffs_delta([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_pooled_and_per_dataset_reports_both_scopes():
    rng = np.random.default_rng(0)
    model = {d: (0.5 + rng.normal(0, 0.005, size=10)).tolist() for d in ("D1", "D2", "D3")}
    baseline = {d: (np.asarray(v) - 0.02).tolist() for d, v in model.items()}
    results = pooled_and_per_dataset(model, baseline, baseline_name="tabtransformer")
    assert set(results) == {"pooled", "D1", "D2", "D3"}
    assert results["pooled"].n_pairs == 30
    assert results["pooled"].max_statistic_v == 465
    assert "A-04" in results["pooled"].note


def test_holm_is_monotone_and_capped():
    adjusted = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.5})
    assert adjusted["a"] == pytest.approx(0.003)
    assert adjusted["b"] == pytest.approx(0.04)
    assert adjusted["c"] == pytest.approx(0.5)
    assert all(v <= 1.0 for v in adjusted.values())


def test_holm_never_decreases_along_the_ordering():
    raw = {f"m{i}": p for i, p in enumerate([0.001, 0.004, 0.01, 0.03, 0.2])}
    adjusted = holm_bonferroni(raw)
    ordered = [adjusted[k] for k in sorted(raw, key=lambda k: raw[k])]
    assert ordered == sorted(ordered)


def test_delong_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(5)
    y = (rng.uniform(size=800) < 0.2).astype(int)
    scores = rng.uniform(size=800) + 0.6 * y
    aucs, _ = fast_delong(np.vstack([scores, scores]), y)
    assert aucs[0] == pytest.approx(roc_auc_score(y, scores), abs=1e-6)


def test_delong_identical_models_give_zero_z():
    rng = np.random.default_rng(6)
    y = (rng.uniform(size=500) < 0.3).astype(int)
    scores = rng.uniform(size=500) + 0.5 * y
    result = delong_test(scores, scores, y, dataset="D1", baseline="self",
                         seed_policy="single_seed", seed=42)
    assert result.z == pytest.approx(0.0, abs=1e-9)
    assert result.p_value_two_sided == pytest.approx(1.0)


def test_delong_detects_a_real_difference():
    rng = np.random.default_rng(9)
    y = (rng.uniform(size=4000) < 0.2).astype(int)
    strong = rng.uniform(size=4000) + 1.2 * y
    weak = rng.uniform(size=4000) + 0.2 * y
    result = delong_test(strong, weak, y, dataset="D1", baseline="weak",
                         seed_policy="single_seed", seed=42)
    assert result.z > 5
    assert result.p_value_two_sided < 1e-6
    assert result.delta_pp > 0


def test_delong_requires_an_explicit_seed_policy():
    y = np.array([0, 1] * 50)
    scores = np.random.default_rng(0).uniform(size=100)
    with pytest.raises(ConfigurationError, match="seed_policy"):
        delong_test(scores, scores, y, dataset="D1", baseline="b",
                    seed_policy="whatever", seed=None)  # type: ignore[arg-type]


def test_delong_single_seed_policy_requires_the_seed():
    y = np.array([0, 1] * 50)
    scores = np.random.default_rng(0).uniform(size=100)
    with pytest.raises(ConfigurationError, match="requires the seed"):
        delong_test(scores, scores, y, dataset="D1", baseline="b",
                    seed_policy="single_seed", seed=None)


def test_bootstrap_ci_brackets_the_point_estimate():
    values = np.random.default_rng(1).normal(loc=2.0, scale=0.5, size=200)
    point, low, high = bootstrap_ci(values, n_resamples=500)
    assert low < point < high
