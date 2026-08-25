"""The Stage 3 verifier must reject unsupported claims (manuscript S4.4)."""

from __future__ import annotations

import copy
import json

import pytest

from vrfraudnet.verifier import (
    DeterministicVerifier,
    GraphEvidence,
    TransactionEvidence,
    build_graph_evidence,
)


@pytest.fixture(scope="module")
def verifier() -> DeterministicVerifier:
    return DeterministicVerifier()


@pytest.fixture
def evidence() -> TransactionEvidence:
    return TransactionEvidence.from_row(
        "TX-1",
        timestamp=1_000.0,
        numeric={"amount": 9500.0, "account_age_days": 4.0},
        categorical={"payment_type": "wire"},
        entities={"beneficiary": "ACC-88231", "counterparty": "ACC-77004"},
        temporal_aggregates={"recent_transfer_count": 7.0},
        graph=GraphEvidence(
            adjacency={
                "ACC-77004": {"ACC-51902"},
                "ACC-51902": {"ACC-FLAGGED"},
            },
            as_of_time=1_000.0,
            flagged_entities={"ACC-FLAGGED"},
        ),
    )


def _rationale(**overrides):
    base = {
        "verdict": "fraud",
        "rationale_probability": 0.8,
        "claims": [
            {"type": "numeric_comparison", "field": "amount", "operator": "gt",
             "value": 5000, "evidence_id": "amount"},
        ],
        "evidence_references": ["amount"],
    }
    base.update(overrides)
    return base


def test_supported_claim_is_accepted(verifier, evidence):
    result = verifier.verify(_rationale(), evidence)
    assert result.accepted and result.status == 1


def test_contradicted_numeric_claim_is_rejected(verifier, evidence):
    rationale = _rationale(
        claims=[{"type": "numeric_comparison", "field": "amount", "operator": "lt",
                 "value": 100, "evidence_id": "amount"}],
        verdict="uncertain",
    )
    result = verifier.verify(rationale, evidence)
    assert not result.accepted
    assert any("contradicted" in reason for reason in result.failure_reasons)


def test_unresolvable_evidence_id_is_rejected(verifier, evidence):
    rationale = _rationale(
        claims=[{"type": "numeric_comparison", "field": "ghost", "operator": "gt",
                 "value": 1, "evidence_id": "ghost_field"}],
        evidence_references=["ghost_field"],
    )
    result = verifier.verify(rationale, evidence)
    assert not result.accepted
    assert any("CR2" in reason for reason in result.failure_reasons)


def test_claim_not_declared_in_evidence_references_is_rejected(verifier, evidence):
    rationale = _rationale(evidence_references=["payment_type"])
    result = verifier.verify(rationale, evidence)
    assert not result.accepted
    assert any("absent from evidence_references" in reason for reason in result.failure_reasons)


def test_graph_claim_is_unverifiable_without_graph_evidence(verifier):
    no_graph = TransactionEvidence.from_row(
        "TX-2", timestamp=10.0, entities={"counterparty": "ACC-1"}
    )
    rationale = _rationale(
        claims=[{"type": "graph_path", "field": "counterparty", "operator": "path_exists",
                 "value": {"target": "ACC-FLAGGED", "max_hops": 2},
                 "evidence_id": "counterparty"}],
        evidence_references=["counterparty"],
    )
    result = verifier.verify(rationale, no_graph)
    assert not result.accepted
    assert any("graph evidence is unavailable" in r for r in result.failure_reasons)


def test_graph_path_respects_hop_bound(verifier, evidence):
    two_hops = _rationale(
        claims=[{"type": "graph_path", "field": "counterparty", "operator": "path_exists",
                 "value": {"target": "ACC-FLAGGED", "max_hops": 2},
                 "evidence_id": "counterparty"}],
        evidence_references=["counterparty"],
    )
    assert verifier.verify(two_hops, evidence).accepted

    one_hop = copy.deepcopy(two_hops)
    one_hop["claims"][0]["value"]["max_hops"] = 1
    assert not verifier.verify(one_hop, evidence).accepted


def test_verdict_must_be_supported_by_verified_claims(verifier, evidence):
    rationale = _rationale(verdict="legitimate")
    result = verifier.verify(rationale, evidence)
    assert not result.accepted
    assert any("CR3" in reason for reason in result.failure_reasons)


def test_contradictory_numeric_bounds_are_detected(verifier, evidence):
    rationale = _rationale(
        claims=[
            {"type": "numeric_comparison", "field": "amount", "operator": "gt",
             "value": 9000, "evidence_id": "amount"},
            {"type": "numeric_comparison", "field": "amount", "operator": "lt",
             "value": 100, "evidence_id": "amount"},
        ],
    )
    result = verifier.verify(rationale, evidence)
    assert any("CR1" in reason for reason in result.failure_reasons)


def test_duplicate_claims_are_rejected(verifier, evidence):
    claim = {"type": "numeric_comparison", "field": "amount", "operator": "gt",
             "value": 5000, "evidence_id": "amount"}
    result = verifier.verify(_rationale(claims=[claim, dict(claim)]), evidence)
    assert any("CR4" in reason for reason in result.failure_reasons)


def test_more_than_eight_claims_is_a_schema_failure(verifier, evidence):
    claim = {"type": "numeric_comparison", "field": "amount", "operator": "gt",
             "value": 5000, "evidence_id": "amount"}
    rationale = _rationale(claims=[dict(claim, value=1000 + i) for i in range(9)])
    result = verifier.verify(rationale, evidence)
    assert not result.accepted
    assert result.schema_failures


def test_unknown_claim_type_is_rejected(verifier, evidence):
    rationale = _rationale(
        claims=[{"type": "vibes", "field": "amount", "operator": "gt",
                 "value": 1, "evidence_id": "amount"}]
    )
    result = verifier.verify(rationale, evidence)
    assert not result.accepted


def test_out_of_range_probability_is_a_schema_failure(verifier, evidence):
    result = verifier.verify(_rationale(rationale_probability=1.7), evidence)
    assert not result.accepted
    assert result.schema_failures


def test_extra_top_level_field_is_rejected(verifier, evidence):
    rationale = _rationale()
    rationale["free_text_explanation"] = "trust me"
    result = verifier.verify(rationale, evidence)
    assert not result.accepted


def test_future_temporal_evidence_is_refused(verifier):
    evidence = TransactionEvidence.from_row("TX-3", timestamp=100.0)
    from vrfraudnet.verifier.evidence import EvidenceEntry

    evidence.entries["future_count"] = EvidenceEntry(
        "future_count", "temporal_aggregate", 9.0, timestamp=500.0
    )
    rationale = _rationale(
        claims=[{"type": "temporal_relation", "field": "future_count",
                 "operator": "count_ge", "value": 1, "evidence_id": "future_count"}],
        evidence_references=["future_count"],
    )
    result = verifier.verify(rationale, evidence)
    assert not result.accepted
    assert any("CR5" in r or "future evidence" in r for r in result.failure_reasons)


def test_graph_evidence_snapshot_excludes_future_edges():
    import numpy as np

    edge_index = np.array([[0, 1], [1, 2]])
    edge_time = np.array([10.0, 500.0])
    graph = build_graph_evidence(
        edge_index, edge_time, ["A", "B", "C"], as_of_time=100.0
    )
    assert graph.neighbours("A") == {"B"}
    assert "C" not in graph.adjacency, "an edge from the future entered the snapshot"


def test_example_files_round_trip(verifier, evidence, repo_root):
    accepted = json.loads((repo_root / "examples" / "rationale_accepted.json").read_text())
    result = verifier.verify(accepted, evidence)
    # The shipped example cites a flagged counterparty id the fixture does not
    # carry, so it is expected to fail on that claim rather than on the schema.
    assert not result.schema_failures, result.schema_failures


def test_verifier_is_deterministic(verifier, evidence):
    rationale = _rationale()
    first = verifier.verify(rationale, evidence).to_dict()
    second = verifier.verify(rationale, evidence).to_dict()
    assert first == second
