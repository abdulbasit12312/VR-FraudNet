"""Focal, cost-sensitive and counterfactual objectives (manuscript S4.6)."""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.errors import ConfigurationError, LeakageError, MissingArtefactError
from vrfraudnet.losses.cost_sensitive import (
    CostMatrix,
    class_balanced_weights,
    cost_sensitive_weights,
    load_cost_matrix,
)
from vrfraudnet.losses.counterfactual import (
    ADMISSIBLE_MODIFICATIONS,
    CounterfactualExample,
    counterfactual_loss,
    generate_counterfactual,
    rationale_stability_penalty,
)
from vrfraudnet.losses.focal import focal_grad_hess, focal_loss, sigmoid


def test_sigmoid_is_stable_at_extremes():
    values = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(np.isfinite(values))
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(0.5)
    assert values[2] == pytest.approx(1.0)


def test_focal_loss_reduces_to_log_loss_at_gamma_zero():
    z = np.array([-2.0, 0.0, 1.5])
    y = np.array([0.0, 1.0, 1.0])
    p = sigmoid(z)
    expected = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert np.allclose(focal_loss(z, y, gamma=0.0), expected)


def test_focal_loss_down_weights_easy_examples():
    easy = focal_loss(np.array([5.0]), np.array([1.0]), gamma=2.0)
    hard = focal_loss(np.array([-1.0]), np.array([1.0]), gamma=2.0)
    assert hard > easy


def test_focal_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    y = (rng.uniform(size=200) < 0.3).astype(float)
    grad, _ = focal_grad_hess(z, y, gamma=2.0)
    eps = 1e-6
    numeric = (focal_loss(z + eps, y) - focal_loss(z - eps, y)) / (2 * eps)
    assert np.allclose(grad, numeric, atol=1e-5)


def test_focal_hessian_is_positive():
    rng = np.random.default_rng(1)
    z = rng.normal(size=200)
    y = (rng.uniform(size=200) < 0.3).astype(float)
    _, hess = focal_grad_hess(z, y)
    assert np.all(hess > 0), "LightGBM requires strictly positive curvature"


def test_class_balanced_weights_favour_the_minority_class():
    y = np.array([0] * 990 + [1] * 10)
    weights = class_balanced_weights(y)
    assert weights[y == 1].mean() > weights[y == 0].mean()
    assert weights.mean() == pytest.approx(1.0)


def test_cost_matrix_normalisation_preserves_the_ratio():
    costs = CostMatrix("D1", 500.0, 25.0, 5.0)
    fn, fp = costs.normalised()
    assert fn == pytest.approx(1.0)
    assert fp == pytest.approx(0.05)
    assert fn / fp == pytest.approx(500.0 / 25.0)


def test_true_positive_review_cost_is_not_a_classification_penalty():
    costs = CostMatrix("D1", 500.0, 25.0, 5.0)
    fn, fp = costs.normalised()
    # Normalisation involves only the FN and FP costs (manuscript S4.6.1).
    assert (fn, fp) == CostMatrix("D1", 500.0, 25.0, 999.0).normalised()


def test_total_loss_decomposition():
    costs = CostMatrix("D1", 500.0, 25.0, 5.0)
    assert costs.total_loss(tp=2, fp=4, fn=3) == pytest.approx(3 * 500 + 4 * 25 + 2 * 5)


def test_cost_matrix_requires_an_explicit_file():
    with pytest.raises(MissingArtefactError, match="Supplementary Table S1"):
        load_cost_matrix("D1", None)


def test_shipped_cost_file_refuses_null_values(repo_root):
    with pytest.raises(ConfigurationError, match="will not guess it"):
        load_cost_matrix("D1", repo_root / "configs" / "costs.yaml")


def test_cost_sensitive_weights_blend_is_bounded():
    y = np.array([0] * 900 + [1] * 100)
    costs = CostMatrix("D1", 500.0, 25.0, 5.0)
    weights = cost_sensitive_weights(y, costs, cost_weight=0.50)
    assert weights.mean() == pytest.approx(1.0)
    assert np.all(weights > 0)


def test_cost_weight_must_be_a_fraction():
    y = np.array([0, 1])
    costs = CostMatrix("D1", 1.0, 1.0, 1.0)
    with pytest.raises(ConfigurationError):
        cost_sensitive_weights(y, costs, cost_weight=1.5)


def test_counterfactual_must_come_from_training_data():
    with pytest.raises(LeakageError, match="refusing to build a counterfactual"):
        generate_counterfactual(
            {"amount": 100.0}, [], rng=np.random.default_rng(0), partition="test"
        )


def test_counterfactual_modifies_exactly_one_field(rng):
    evidence = {"amount": 100.0, "count": 3, "entity": "ACC-1"}
    modified, example = generate_counterfactual(evidence, [], rng=rng, partition="train")
    changed = [k for k in evidence if modified[k] != evidence[k]]
    assert len(changed) == 1
    assert example.modification_kind in ADMISSIBLE_MODIFICATIONS


def test_counterfactual_updates_only_affected_claims(rng):
    evidence = {"amount": 100.0}
    claims = [
        {"type": "numeric_comparison", "field": "amount", "evidence_id": "amount"},
        {"type": "entity_match", "field": "beneficiary", "evidence_id": "beneficiary"},
    ]
    _, example = generate_counterfactual(evidence, claims, rng=rng, partition="train")
    assert example.claims_updated == [0]
    assert example.claims_removed == []


def test_immutable_fields_cannot_be_a_counterfactual_target():
    with pytest.raises(ValueError, match="immutable"):
        CounterfactualExample(
            source_id="x", modified_field="label", modification_kind="numeric_relation",
            original_value=1, counterfactual_value=0, claims_updated=[], claims_removed=[],
        )


def test_stability_penalty_only_bites_when_evidence_is_unchanged():
    assert rationale_stability_penalty(0.8, 0.2, evidence_changed=True) == 0.0
    assert rationale_stability_penalty(0.8, 0.81, evidence_changed=False) == 0.0
    assert rationale_stability_penalty(0.8, 0.2, evidence_changed=False) > 0.0


def test_counterfactual_loss_uses_the_manuscript_weight():
    assert counterfactual_loss(1.0, [0.4], weight=0.25) == pytest.approx(1.1)
    assert counterfactual_loss(1.0, [], weight=0.25) == pytest.approx(1.0)


def test_counterfactual_weight_must_be_a_fraction():
    with pytest.raises(ValueError):
        counterfactual_loss(1.0, [0.1], weight=2.0)
