"""Chronological and strict-inductive split construction (manuscript S3.1, Table 1).

Every split function returns integer index arrays and is accompanied by an
assertion that no test-period observation precedes a training-period
observation. The assertions are not decoration: ``tests/test_leakage_splits.py``
depends on them, and they are the cheapest possible defence against the single
most common source of inflated fraud-detection results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrfraudnet.errors import LeakageError


@dataclass(frozen=True)
class Split:
    """Index arrays for one train/validation/test partitioning.

    ``calibration`` is optional and carries the second half of the validation
    segment used exclusively for split-conformal calibration (manuscript S4.8.1:
    "The validation segment was divided chronologically into equal
    model-selection and calibration portions").
    """

    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    calibration: np.ndarray | None = None
    description: str = ""

    def __post_init__(self) -> None:
        parts = {"train": self.train, "validation": self.validation, "test": self.test}
        if self.calibration is not None:
            parts["calibration"] = self.calibration
        seen: dict[int, str] = {}
        for name, idx in parts.items():
            for i in idx.tolist():
                if i in seen:
                    raise LeakageError(
                        f"index {i} appears in both {seen[i]!r} and {name!r} partitions"
                    )
                seen[i] = name

    @property
    def sizes(self) -> dict[str, int]:
        out = {
            "train": int(self.train.size),
            "validation": int(self.validation.size),
            "test": int(self.test.size),
        }
        if self.calibration is not None:
            out["calibration"] = int(self.calibration.size)
        return out


def assert_temporal_order(
    time_values: np.ndarray,
    earlier: np.ndarray,
    later: np.ndarray,
    *,
    label: str,
) -> None:
    """Assert that every observation in ``later`` is at or after all of ``earlier``.

    Raises :class:`LeakageError` on violation. This is the guard that makes
    "no future data enters training" a testable property rather than a claim.
    """
    if earlier.size == 0 or later.size == 0:
        return
    max_earlier = np.max(time_values[earlier])
    min_later = np.min(time_values[later])
    if min_later < max_earlier:
        raise LeakageError(
            f"{label}: temporal order violated. The latest observation in the earlier "
            f"partition has time {max_earlier!r} but the earliest observation in the "
            f"later partition has time {min_later!r}."
        )


def chronological_holdout_by_group(
    group_values: np.ndarray,
    train_groups: tuple[int, ...],
    test_groups: tuple[int, ...],
    *,
    validation_fraction: float = 0.10,
    split_calibration: bool = True,
) -> Split:
    """D1 BAF split: month-based separation (manuscript Table 1, S3.1).

    Months 0-5 train, months 6-7 test, and the final ``validation_fraction`` of
    the training period (in chronological order) held out for validation.

    Parameters
    ----------
    group_values:
        Per-row month index.
    train_groups, test_groups:
        Month values assigned to each side of the boundary.
    validation_fraction:
        Manuscript value 0.10 ("validation = last 10% of train, chronological").
    split_calibration:
        Halve the validation segment into model-selection and calibration
        portions (manuscript S4.8.1).
    """
    overlap = set(train_groups) & set(test_groups)
    if overlap:
        raise LeakageError(f"month(s) {sorted(overlap)} appear in both train and test")

    train_mask = np.isin(group_values, train_groups)
    test_mask = np.isin(group_values, test_groups)

    train_pool = np.flatnonzero(train_mask)
    # Chronological ordering within the training period: sort by month, keeping
    # the original row order stable inside a month.
    train_pool = train_pool[np.argsort(group_values[train_pool], kind="stable")]

    n_val = int(round(validation_fraction * train_pool.size))
    if n_val < 1:
        raise ValueError("validation_fraction too small for this training pool")
    train_idx, val_idx = train_pool[:-n_val], train_pool[-n_val:]
    test_idx = np.flatnonzero(test_mask)

    assert_temporal_order(group_values, train_idx, val_idx, label="D1 train->validation")
    assert_temporal_order(group_values, train_idx, test_idx, label="D1 train->test")
    assert_temporal_order(group_values, val_idx, test_idx, label="D1 validation->test")

    calib_idx = None
    if split_calibration:
        val_idx, calib_idx = _halve_chronologically(val_idx)

    return Split(
        train=train_idx,
        validation=val_idx,
        test=test_idx,
        calibration=calib_idx,
        description=(
            f"group holdout: train={train_groups} test={test_groups} "
            f"validation={validation_fraction:.0%} tail of train"
        ),
    )


def chronological_fraction_split(
    time_values: np.ndarray,
    *,
    train_fraction: float,
    validation_fraction: float,
    split_calibration: bool = True,
) -> Split:
    """D2 AMLworld split: temporal 60/20/20 by transaction timestamp (Table 1)."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("fractions must lie in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave room for a test partition")

    order = np.argsort(time_values, kind="stable")
    n = order.size
    n_train = int(round(train_fraction * n))
    n_val = int(round(validation_fraction * n))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    assert_temporal_order(time_values, train_idx, val_idx, label="D2 train->validation")
    assert_temporal_order(time_values, val_idx, test_idx, label="D2 validation->test")

    calib_idx = None
    if split_calibration:
        val_idx, calib_idx = _halve_chronologically(val_idx)

    return Split(
        train=train_idx,
        validation=val_idx,
        test=test_idx,
        calibration=calib_idx,
        description=(
            f"chronological fractions train={train_fraction:.0%} "
            f"validation={validation_fraction:.0%}"
        ),
    )


def quantile_time_split(
    time_values: np.ndarray,
    *,
    test_quantile: float = 0.80,
    validation_fraction: float = 0.10,
    split_calibration: bool = True,
) -> Split:
    """D3 IEEE-CIS split: ``TransactionDT`` quantile boundary (Table 1, S3.1).

    "observations below the 0.8 quantile are assigned to training, whereas
    observations at or above the 0.8 quantile are reserved for testing."

    The manuscript does not state how the validation partition is carved out of
    the training period for D3. The chronological tail of the training period is
    used here for consistency with D1. See docs/KNOWN_LIMITATIONS.md L-02.
    """
    cut = float(np.quantile(time_values, test_quantile))
    train_pool = np.flatnonzero(time_values < cut)
    test_idx = np.flatnonzero(time_values >= cut)

    train_pool = train_pool[np.argsort(time_values[train_pool], kind="stable")]
    n_val = int(round(validation_fraction * train_pool.size))
    train_idx, val_idx = train_pool[:-n_val], train_pool[-n_val:]

    assert_temporal_order(time_values, train_idx, val_idx, label="D3 train->validation")
    assert_temporal_order(time_values, val_idx, test_idx, label="D3 validation->test")

    calib_idx = None
    if split_calibration:
        val_idx, calib_idx = _halve_chronologically(val_idx)

    return Split(
        train=train_idx,
        validation=val_idx,
        test=test_idx,
        calibration=calib_idx,
        description=f"quantile split at q={test_quantile} of TransactionDT (cut={cut})",
    )


def expanding_window_folds(
    time_values: np.ndarray,
    *,
    n_folds: int = 5,
    initial_fraction: float = 0.40,
) -> list[Split]:
    """D3 expanding-window time-series CV (manuscript Table 1, S3.1, Table 9(b)).

    Fold ``k`` trains on everything before a growing cut point and tests on the
    next contiguous block. Only the number of folds (5) is stated by the
    manuscript; the initial window size is an ASSUMPTION recorded as L-03.
    """
    order = np.argsort(time_values, kind="stable")
    n = order.size
    start = int(round(initial_fraction * n))
    if start < 1 or start >= n:
        raise ValueError("initial_fraction produces an empty or full training window")
    block = (n - start) // n_folds
    if block < 1:
        raise ValueError("not enough samples for the requested number of folds")

    folds: list[Split] = []
    for k in range(n_folds):
        train_end = start + k * block
        test_end = train_end + block if k < n_folds - 1 else n
        train_idx = order[:train_end]
        test_idx = order[train_end:test_end]
        # Validation for early stopping is the chronological tail of the fold's
        # training window, never the fold's test block.
        n_val = max(1, int(round(0.10 * train_idx.size)))
        fold_train, fold_val = train_idx[:-n_val], train_idx[-n_val:]
        assert_temporal_order(time_values, fold_train, fold_val, label=f"D3 fold{k+1} train->val")
        assert_temporal_order(time_values, fold_val, test_idx, label=f"D3 fold{k+1} val->test")
        folds.append(
            Split(
                train=fold_train,
                validation=fold_val,
                test=test_idx,
                description=f"expanding window fold {k + 1}/{n_folds}",
            )
        )
    return folds


def strict_inductive_split(
    timestep_values: np.ndarray,
    *,
    train_timesteps: tuple[int, int],
    test_timesteps: tuple[int, int],
    validation_timesteps: tuple[int, int] | None = None,
    split_calibration: bool = True,
) -> Split:
    """D4 Elliptic++ strict-inductive split (manuscript Table 1, S3.1).

    "timesteps 1-34 assigned to training and timesteps 35-49 assigned to
    testing. Test-period nodes are not included as supervised targets during
    training."

    The manuscript does not name a validation timestep range for D4. The
    chronological tail of the training range is used by default; see L-04.
    """
    lo_tr, hi_tr = train_timesteps
    lo_te, hi_te = test_timesteps
    if hi_tr >= lo_te:
        raise LeakageError(
            f"training timesteps {train_timesteps} overlap test timesteps {test_timesteps}"
        )

    if validation_timesteps is None:
        # Reserve the last 15% of the training timestep range for validation.
        n_train_steps = hi_tr - lo_tr + 1
        n_val_steps = max(1, int(round(0.15 * n_train_steps)))
        validation_timesteps = (hi_tr - n_val_steps + 1, hi_tr)
    lo_va, hi_va = validation_timesteps

    train_mask = (timestep_values >= lo_tr) & (timestep_values < lo_va)
    val_mask = (timestep_values >= lo_va) & (timestep_values <= hi_va)
    test_mask = (timestep_values >= lo_te) & (timestep_values <= hi_te)

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    assert_temporal_order(timestep_values, train_idx, val_idx, label="D4 train->validation")
    assert_temporal_order(timestep_values, val_idx, test_idx, label="D4 validation->test")

    calib_idx = None
    if split_calibration:
        val_idx, calib_idx = _halve_chronologically(val_idx)

    return Split(
        train=train_idx,
        validation=val_idx,
        test=test_idx,
        calibration=calib_idx,
        description=(
            f"strict-inductive: train t in [{lo_tr},{lo_va - 1}], "
            f"validation t in [{lo_va},{hi_va}], test t in [{lo_te},{hi_te}]"
        ),
    )


def node_arrival_split(
    arrival_times: np.ndarray,
    labelled_mask: np.ndarray,
    *,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
    split_calibration: bool = True,
) -> Split:
    """D5 DGraph-Fin temporal node-arrival split (manuscript Table 1, S3.1).

    Only labelled nodes become supervised targets; background nodes are retained
    for message passing but excluded from every returned index array, which is
    what "supervised loss is restricted to labelled nodes" means operationally.

    The 60/20/20 proportions are an ASSUMPTION (L-05): the manuscript states a
    "temporal node-arrival split" without naming proportions for D5.
    """
    labelled = np.flatnonzero(labelled_mask)
    order = labelled[np.argsort(arrival_times[labelled], kind="stable")]
    n = order.size
    n_train = int(round(train_fraction * n))
    n_val = int(round(validation_fraction * n))
    train_idx = order[:n_train]
    val_idx = order[n_train : n_train + n_val]
    test_idx = order[n_train + n_val :]

    assert_temporal_order(arrival_times, train_idx, val_idx, label="D5 train->validation")
    assert_temporal_order(arrival_times, val_idx, test_idx, label="D5 validation->test")

    calib_idx = None
    if split_calibration:
        val_idx, calib_idx = _halve_chronologically(val_idx)

    return Split(
        train=train_idx,
        validation=val_idx,
        test=test_idx,
        calibration=calib_idx,
        description="temporal node-arrival split over labelled nodes only",
    )


def _halve_chronologically(idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split an already chronologically ordered index array into two halves.

    Manuscript S4.8.1: the first half is used for model selection, early
    stopping, routing-threshold selection and isotonic fitting; the second half
    is held out exclusively for split-conformal calibration.
    """
    half = idx.size // 2
    if half < 1:
        raise ValueError("validation segment too small to split into model-selection/calibration")
    return idx[:half], idx[half:]
