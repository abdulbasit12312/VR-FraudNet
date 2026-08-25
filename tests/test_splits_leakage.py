"""No future observation may enter a training partition (manuscript S3.1, S3.2)."""

from __future__ import annotations

import numpy as np
import pytest

from vrfraudnet.data.splits import (
    Split,
    assert_temporal_order,
    chronological_fraction_split,
    chronological_holdout_by_group,
    expanding_window_folds,
    node_arrival_split,
    quantile_time_split,
    strict_inductive_split,
)
from vrfraudnet.errors import LeakageError


def test_partitions_are_disjoint():
    months = np.repeat(np.arange(8), 100)
    split = chronological_holdout_by_group(months, (0, 1, 2, 3, 4, 5), (6, 7))
    everything = np.concatenate(
        [split.train, split.validation, split.test, split.calibration]
    )
    assert everything.size == np.unique(everything).size


def test_d1_validation_is_the_chronological_tail_of_training():
    months = np.repeat(np.arange(8), 100)
    split = chronological_holdout_by_group(months, (0, 1, 2, 3, 4, 5), (6, 7))
    assert months[split.train].max() <= months[split.validation].min()
    assert months[split.validation].max() <= months[split.test].min()


def test_d1_validation_fraction_is_ten_percent():
    months = np.repeat(np.arange(8), 100)
    split = chronological_holdout_by_group(
        months, (0, 1, 2, 3, 4, 5), (6, 7), validation_fraction=0.10
    )
    n_train_pool = 600
    n_validation = split.validation.size + split.calibration.size
    assert n_validation == pytest.approx(0.10 * n_train_pool, abs=1)


def test_overlapping_months_are_rejected():
    months = np.repeat(np.arange(8), 10)
    with pytest.raises(LeakageError, match="both train and test"):
        chronological_holdout_by_group(months, (0, 1, 2, 3, 4, 5), (5, 6, 7))


def test_d2_temporal_fractions_respect_order():
    times = np.sort(np.random.default_rng(0).uniform(0, 1000, size=1000))
    split = chronological_fraction_split(times, train_fraction=0.6, validation_fraction=0.2)
    assert times[split.train].max() <= times[split.validation].min()
    assert times[split.validation].max() <= times[split.test].min()
    assert split.train.size == pytest.approx(600, abs=1)


def test_d3_quantile_boundary():
    times = np.arange(1000, dtype=float)
    split = quantile_time_split(times, test_quantile=0.80)
    assert times[split.test].min() >= np.quantile(times, 0.80)
    assert times[split.train].max() < times[split.test].min()


def test_expanding_window_folds_never_train_on_the_future():
    times = np.arange(2000, dtype=float)
    folds = expanding_window_folds(times, n_folds=5)
    assert len(folds) == 5
    previous_train_size = -1
    for fold in folds:
        assert times[fold.train].max() < times[fold.test].min()
        assert times[fold.validation].max() < times[fold.test].min()
        assert fold.train.size > previous_train_size  # the window expands
        previous_train_size = fold.train.size


def test_strict_inductive_split_rejects_overlap():
    steps = np.repeat(np.arange(1, 50), 10)
    with pytest.raises(LeakageError, match="overlap"):
        strict_inductive_split(steps, train_timesteps=(1, 40), test_timesteps=(35, 49))


def test_strict_inductive_split_boundaries():
    steps = np.repeat(np.arange(1, 50), 10)
    split = strict_inductive_split(steps, train_timesteps=(1, 34), test_timesteps=(35, 49))
    assert steps[split.train].max() < steps[split.validation].min()
    assert steps[split.validation].max() < steps[split.test].min()
    assert steps[split.test].min() >= 35


def test_node_arrival_split_uses_labelled_nodes_only():
    arrival = np.sort(np.random.default_rng(1).uniform(0, 100, size=500))
    labelled = np.zeros(500, dtype=bool)
    labelled[::2] = True
    split = node_arrival_split(arrival, labelled, train_fraction=0.6, validation_fraction=0.2)
    combined = np.concatenate([split.train, split.validation, split.test, split.calibration])
    assert labelled[combined].all(), "a background node reached a supervised partition"


def test_assert_temporal_order_catches_violation():
    times = np.arange(10, dtype=float)
    with pytest.raises(LeakageError, match="temporal order violated"):
        assert_temporal_order(times, earlier=np.array([5, 6]), later=np.array([1, 2]), label="x")


def test_split_rejects_duplicate_indices():
    with pytest.raises(LeakageError, match="appears in both"):
        Split(train=np.array([1, 2, 3]), validation=np.array([3, 4]), test=np.array([5]))
