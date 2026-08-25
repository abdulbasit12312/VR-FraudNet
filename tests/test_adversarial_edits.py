"""Adversarial edits must be validity-preserving and applicability-checked."""

from __future__ import annotations

import pytest

from vrfraudnet.adversarial.applicability import (
    EDIT_FAMILIES,
    applicable_families,
    check_applicable,
)
from vrfraudnet.adversarial.edits import (
    EditContext,
    apply_budgeted_edits,
    delay_edit,
    feature_perturbation_edit,
    mule_edit,
    split_edit,
)
from vrfraudnet.errors import NotApplicableError


@pytest.fixture
def context() -> EditContext:
    return EditContext(
        dataset_id="D2",
        model_name="vr_fraudnet",
        mutable_columns=("amount", "timestamp", "counterparty", "score"),
        immutable_columns=("label", "transaction_id"),
        feature_bounds={"score": (0.0, 1.0)},
        amount_column="amount",
        timestamp_column="timestamp",
        entity_columns=("counterparty",),
        admissible_entities=("ACC-1", "ACC-2", "ACC-3"),
    )


@pytest.fixture
def row() -> dict:
    return {
        "transaction_id": "TX-1",
        "label": 1,
        "amount": 1000.0,
        "timestamp": 500.0,
        "counterparty": "ACC-1",
        "score": 0.4,
    }


def test_split_preserves_the_total_amount(context, row, rng):
    result = split_edit(row, context, rng)
    assert result.is_valid and len(result.rows) == 2
    total = sum(r["amount"] for r in result.rows)
    assert total == pytest.approx(row["amount"])


def test_delay_never_moves_time_backwards(context, row, rng):
    for _ in range(50):
        result = delay_edit(row, context, rng)
        assert result.is_valid
        assert result.rows[0]["timestamp"] >= row["timestamp"]


def test_mule_edit_uses_only_admissible_entities(context, row, rng):
    for _ in range(20):
        result = mule_edit(row, context, rng)
        assert result.is_valid
        assert result.rows[0]["counterparty"] in context.admissible_entities
        assert result.rows[0]["counterparty"] != row["counterparty"]


def test_mule_edit_refuses_without_a_training_entity_pool(row, rng):
    context = EditContext(
        dataset_id="D2",
        model_name="vr_fraudnet",
        mutable_columns=("counterparty",),
        immutable_columns=(),
        feature_bounds={},
        entity_columns=("counterparty",),
        admissible_entities=(),
    )
    result = mule_edit(row, context, rng)
    assert not result.is_valid
    assert "TRAINING graph" in result.reason


def test_feature_perturbation_stays_inside_training_bounds(context, row, rng):
    for _ in range(100):
        result = feature_perturbation_edit(row, context, rng)
        assert result.is_valid
        assert 0.0 <= result.rows[0]["score"] <= 1.0


def test_immutable_columns_cannot_be_edited(row, rng):
    context = EditContext(
        dataset_id="D2",
        model_name="vr_fraudnet",
        mutable_columns=("label",),
        immutable_columns=("label",),
        feature_bounds={"label": (0.0, 1.0)},
    )
    with pytest.raises(Exception):
        context.assert_mutable("label")


def test_injection_is_not_applicable_to_any_benchmark():
    for dataset in ("D1", "D2", "D3", "D4", "D5"):
        with pytest.raises(NotApplicableError, match="no untrusted free-text field"):
            check_applicable("inject", dataset, "vr_fraudnet_stage2")


def test_injection_is_not_applicable_to_text_free_models():
    # Even with a declared text field, a tabular model cannot consume it.
    with pytest.raises(NotApplicableError, match="consumes\nno text|consumes no text"):
        check_applicable("inject", "D3", "lightgbm", declared_text_field="memo")


def test_injection_is_permitted_once_both_preconditions_hold():
    check_applicable("inject", "D3", "vr_fraudnet_stage2", declared_text_field="memo")


def test_split_is_not_applicable_to_d1():
    with pytest.raises(NotApplicableError, match="transaction splitting is undefined for D1"):
        check_applicable("split", "D1", "vr_fraudnet")


def test_mule_is_not_applicable_without_a_graph():
    for dataset in ("D1", "D3"):
        with pytest.raises(NotApplicableError, match="mule-routing"):
            check_applicable("mule", dataset, "vr_fraudnet")


def test_applicable_families_matrix_is_stable():
    assert applicable_families("D1", "lightgbm") == ("feat",)
    assert set(applicable_families("D2", "lightgbm")) == {"split", "delay", "mule", "feat"}
    assert set(applicable_families("D3", "lightgbm")) == {"split", "delay", "feat"}
    assert set(applicable_families("D4", "lightgbm")) == {"delay", "mule", "feat"}
    for dataset in ("D1", "D2", "D3", "D4", "D5"):
        assert "inject" not in applicable_families(dataset, "lightgbm")


def test_budgeted_edits_apply_the_requested_number(context, row, rng):
    result = apply_budgeted_edits(row, context, budget=2, families=["delay"], rng=rng)
    assert result.applied
    assert result.rows[0]["timestamp"] >= row["timestamp"]


def test_budget_must_be_positive(context, row, rng):
    with pytest.raises(ValueError):
        apply_budgeted_edits(row, context, budget=0, families=["delay"], rng=rng)


def test_all_five_families_are_registered():
    assert set(EDIT_FAMILIES) == {"split", "delay", "mule", "inject", "feat"}
