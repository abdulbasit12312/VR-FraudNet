"""Deterministic seed handling (manuscript S4.8.1)."""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.determinism import derive_subseed, rng_for, set_global_seed
from vrfraudnet.seeds import (
    MANUSCRIPT_SEEDS,
    is_manuscript_seed_set,
    validate_seeds,
    wilcoxon_max_statistic,
)


def test_manuscript_seed_list_is_exact():
    assert MANUSCRIPT_SEEDS == (42, 123, 456, 789, 2024, 3141, 2718, 1001, 1729, 4096)
    assert len(MANUSCRIPT_SEEDS) == 10


def test_thirty_pairs_matches_the_reported_v_max():
    assert wilcoxon_max_statistic(len(MANUSCRIPT_SEEDS) * 3) == 465


def test_duplicate_seeds_are_rejected():
    with pytest.raises(ValueError, match="duplicate seeds"):
        validate_seeds([42, 42, 123])


def test_empty_seed_list_is_rejected():
    with pytest.raises(ValueError, match="at least one seed"):
        validate_seeds([])


def test_manuscript_seed_set_detection_is_order_insensitive():
    assert is_manuscript_seed_set(list(reversed(MANUSCRIPT_SEEDS)))
    assert not is_manuscript_seed_set([42, 123])


def test_seeding_makes_numpy_reproducible():
    set_global_seed(42)
    first = np.random.rand(5)
    set_global_seed(42)
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_subseeds_are_stable_and_distinct():
    a = derive_subseed(42, "adversarial")
    b = derive_subseed(42, "adversarial")
    c = derive_subseed(42, "retrieval")
    d = derive_subseed(123, "adversarial")
    assert a == b
    assert a != c
    assert a != d


def test_named_substreams_do_not_collide():
    x = rng_for(42, "edits").normal(size=100)
    y = rng_for(42, "init").normal(size=100)
    assert not np.allclose(x, y)


def test_named_substream_is_reproducible():
    x = rng_for(42, "edits").normal(size=100)
    y = rng_for(42, "edits").normal(size=100)
    assert np.array_equal(x, y)


def test_determinism_report_records_what_was_pinned():
    report = set_global_seed(2024)
    assert report.seed == 2024
    assert report.python_hash_seed == "2024"
    assert report.numpy_seeded
    payload = report.to_dict()
    assert set(payload) >= {"seed", "numpy_seeded", "torch_seeded", "torch_deterministic"}


def test_triage_training_is_reproducible_for_a_fixed_seed():
    from vrfraudnet.models.stage1_triage import TriageClassifier, TriageParams

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1500, 6))
    y = (rng.uniform(size=1500) < 0.1).astype(int)
    params = TriageParams(n_estimators=40, num_threads=1)

    first = TriageClassifier(params, seed=42).fit(x[:1000], y[:1000], x[1000:], y[1000:])
    second = TriageClassifier(params, seed=42).fit(x[:1000], y[:1000], x[1000:], y[1000:])
    assert np.allclose(first.predict_proba(x[1000:]), second.predict_proba(x[1000:]))
