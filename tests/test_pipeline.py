"""End-to-end wiring of Stages 1, 3 and 4 (manuscript Section 4, Figure 2).

The property under test is the paper's central claim: a rationale-side
probability can reach the Stage 4 mixer only through a Stage 3 acceptance, and a
verifier rejection routes to human review rather than to automated action.
"""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.models.pipeline import VRFraudNet, escalation_statistics
from vrfraudnet.models.stage1_triage import (
    Route,
    ThresholdGate,
    TriageClassifier,
    TriageParams,
)
from vrfraudnet.models.stage4_calibration import IsotonicMixer, MixerInputs
from vrfraudnet.verifier import DeterministicVerifier


@pytest.fixture(scope="module")
def fitted_pipeline():
    rng = np.random.default_rng(11)
    n = 4000
    x = rng.normal(size=(n, 8))
    y = (rng.uniform(size=n) < 0.08).astype(int)
    train, valid = slice(0, 3000), slice(3000, n)

    triage = TriageClassifier(TriageParams(n_estimators=60, num_threads=1), seed=42)
    triage.fit(x[train], y[train], x[valid], y[valid])
    p_valid = triage.predict_proba(x[valid])
    mixer = IsotonicMixer().fit(MixerInputs(p_valid, None, None), y[valid])

    pipeline = VRFraudNet(
        triage=triage,
        gate=ThresholdGate(0.20, 0.80),
        verifier=DeterministicVerifier(),
        mixer=mixer,
    )
    return pipeline, x[valid], y[valid]


def test_routing_covers_the_three_decisions(fitted_pipeline):
    pipeline, x, _ = fitted_pipeline
    probabilities = pipeline.triage_probabilities(x)
    routes = set(pipeline.gate.route(probabilities).tolist())
    assert routes <= {Route.ACCEPT.value, Route.ESCALATE.value, Route.REJECT.value}


def test_decide_returns_one_decision_per_transaction(fitted_pipeline):
    pipeline, x, _ = fitted_pipeline
    ids = [f"TX-{i}" for i in range(x.shape[0])]
    decisions = pipeline.decide(ids, x)
    assert len(decisions) == x.shape[0]
    assert {d.transaction_id for d in decisions} == set(ids)


def test_calibrated_probabilities_are_valid(fitted_pipeline):
    pipeline, x, _ = fitted_pipeline
    decisions = pipeline.decide([f"TX-{i}" for i in range(x.shape[0])], x)
    values = np.array([d.calibrated_probability for d in decisions])
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert np.all(np.isfinite(values))


def test_no_rationale_is_released_without_a_verifier_acceptance(fitted_pipeline):
    pipeline, x, _ = fitted_pipeline
    decisions = pipeline.decide([f"TX-{i}" for i in range(x.shape[0])], x)
    for decision in decisions:
        if decision.rationale_released:
            assert decision.verifier_accepted is True


def test_escalated_transactions_without_acceptance_go_to_human_review(fitted_pipeline):
    pipeline, x, _ = fitted_pipeline
    decisions = pipeline.decide([f"TX-{i}" for i in range(x.shape[0])], x)
    for decision in decisions:
        if decision.route == Route.ESCALATE.value and decision.verifier_accepted is not True:
            assert decision.escalated_to_human


def test_rejected_rationale_contributes_no_probability_mass():
    """The gating property, isolated from the rest of the pipeline."""
    inputs = MixerInputs(
        triage_probability=np.array([0.5, 0.5]),
        rationale_probability=np.array([0.99, 0.99]),
        verifier_status=np.array([1, 0]),
    )
    assert inputs.gated_rationale().tolist() == [0.99, 0.0]


def test_escalation_statistics_sum_correctly(fitted_pipeline):
    pipeline, x, _ = fitted_pipeline
    decisions = pipeline.decide([f"TX-{i}" for i in range(x.shape[0])], x)
    stats = escalation_statistics(decisions)
    total = stats["accept_rate"] + stats["reject_rate"] + stats["escalation_rate"]
    assert total == pytest.approx(1.0)


def test_escalation_statistics_on_an_empty_batch():
    assert escalation_statistics([]) == {}


def test_pipeline_without_stage2_reports_no_verifier_status(fitted_pipeline):
    """Without an adapter there is no rationale, so no verifier verdict exists.

    This is exactly the ablation A6 ('w/o Large LLM') configuration, and the
    decision objects say so rather than implying a verifier ran.
    """
    pipeline, x, _ = fitted_pipeline
    decisions = pipeline.decide([f"TX-{i}" for i in range(x.shape[0])], x)
    escalated = [d for d in decisions if d.route == Route.ESCALATE.value]
    assert escalated, "the fixture should produce at least one escalated transaction"
    assert all(d.verifier_accepted is False for d in escalated)
    assert all(d.rationale is None for d in escalated)
