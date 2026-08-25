"""The retrieval index must contain training examples only (manuscript S4.3.4)."""

from __future__ import annotations

import pytest

from vrfraudnet.errors import LeakageError
from vrfraudnet.retrieval.index import (
    RetrievalIndex,
    RetrievalRecord,
    hashing_encoder,
)


def _record(record_id: str, timestamp: float, summary: str, partition: str = "train",
            entities: frozenset[str] = frozenset()) -> RetrievalRecord:
    return RetrievalRecord(
        record_id=record_id,
        partition=partition,
        timestamp=timestamp,
        summary=summary,
        probability_band="mid",
        rationale={"verdict": "fraud"},
        entity_ids=entities,
    )


def test_non_training_record_is_refused_at_construction():
    for partition in ("validation", "test", "transfer", "adversarial"):
        with pytest.raises(LeakageError, match="training examples only"):
            _record("r1", 1.0, "s", partition=partition)


def test_verifier_rejected_examples_are_not_indexed():
    index = RetrievalIndex(encoder=hashing_encoder(64))
    index.add(_record("r1", 1.0, "wire transfer high amount"), verifier_accepted=False)
    assert len(index) == 0


def test_only_strictly_earlier_examples_are_eligible():
    index = RetrievalIndex(encoder=hashing_encoder(64))
    index.add(_record("early", 10.0, "wire transfer high amount"), verifier_accepted=True)
    index.add(_record("same", 50.0, "wire transfer high amount"), verifier_accepted=True)
    index.add(_record("late", 90.0, "wire transfer high amount"), verifier_accepted=True)
    index.build()

    retrieved = index.query("wire transfer high amount", query_timestamp=50.0)
    ids = {r.record_id for r in retrieved}
    assert ids == {"early"}, ids


def test_shared_transaction_identifier_is_excluded():
    index = RetrievalIndex(encoder=hashing_encoder(64))
    index.add(_record("TX-1", 1.0, "abc"), verifier_accepted=True)
    index.add(_record("TX-2", 2.0, "abc"), verifier_accepted=True)
    index.build()
    retrieved = index.query("abc", query_timestamp=10.0, query_transaction_id="TX-1")
    assert {r.record_id for r in retrieved} == {"TX-2"}


def test_shared_entity_identifier_is_excluded():
    index = RetrievalIndex(encoder=hashing_encoder(64))
    index.add(_record("A", 1.0, "abc", entities=frozenset({"ACC-9"})), verifier_accepted=True)
    index.add(_record("B", 2.0, "abc", entities=frozenset({"ACC-4"})), verifier_accepted=True)
    index.build()
    retrieved = index.query(
        "abc", query_timestamp=10.0, query_entity_ids=frozenset({"ACC-9"})
    )
    assert {r.record_id for r in retrieved} == {"B"}


def test_at_most_four_examples_are_returned():
    index = RetrievalIndex(encoder=hashing_encoder(64))
    for i in range(20):
        index.add(_record(f"r{i}", float(i), f"transfer {i}"), verifier_accepted=True)
    index.build()
    assert len(index.query("transfer 3", query_timestamp=1000.0)) == 4


def test_retrieval_is_omitted_when_no_example_is_eligible():
    index = RetrievalIndex(encoder=hashing_encoder(64))
    index.add(_record("late", 900.0, "abc"), verifier_accepted=True)
    index.build()
    assert index.query("abc", query_timestamp=10.0) == []


def test_empty_index_returns_nothing():
    index = RetrievalIndex(encoder=hashing_encoder(64)).build()
    assert index.query("anything") == []


def test_results_are_ordered_by_similarity():
    index = RetrievalIndex(encoder=hashing_encoder(128))
    index.add(_record("exact", 1.0, "wire transfer to new beneficiary"), verifier_accepted=True)
    index.add(_record("other", 2.0, "grocery purchase small amount"), verifier_accepted=True)
    index.build()
    retrieved = index.query("wire transfer to new beneficiary", query_timestamp=100.0)
    assert retrieved[0].record_id == "exact"
