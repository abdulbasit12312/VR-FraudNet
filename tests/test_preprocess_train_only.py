"""Scalers, encoders and PCA must be fitted on training rows only (S3.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from vrfraudnet.data.preprocess_base import (
    HashEncoder,
    MedianImputer,
    SklearnScalerAdapter,
    SmoothedTargetEncoder,
    TrainFittedPCA,
    TrainOnlyPipeline,
)
from vrfraudnet.errors import LeakageError


def _frame(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": rng.choice(["a", "b", "c"], size=n),
        }
    )


def test_pipeline_refuses_rows_outside_the_training_partition():
    frame = _frame()
    train_index = np.arange(60)
    pipeline = TrainOnlyPipeline(
        steps=[("scale", SklearnScalerAdapter(StandardScaler(), ["num"]))],
        train_index=train_index,
    )
    with pytest.raises(LeakageError, match="outside the declared training partition"):
        pipeline.fit(frame, np.zeros(len(frame)))


def test_pipeline_refuses_a_second_fit():
    frame = _frame()
    train_index = np.arange(60)
    pipeline = TrainOnlyPipeline(
        steps=[("scale", SklearnScalerAdapter(StandardScaler(), ["num"]))],
        train_index=train_index,
    )
    pipeline.fit(frame.iloc[train_index], np.zeros(60))
    with pytest.raises(LeakageError, match="already been fitted"):
        pipeline.fit(frame.iloc[train_index], np.zeros(60))


def test_transform_before_fit_is_refused():
    pipeline = TrainOnlyPipeline(steps=[], train_index=np.arange(10))
    with pytest.raises(LeakageError, match="before fit"):
        pipeline.transform(_frame(10))


def test_scaler_statistics_come_from_training_rows_only():
    frame = _frame(100)
    # Make the held-out half wildly different so leakage would be visible.
    frame.loc[50:, "num"] = frame.loc[50:, "num"] * 1000 + 5000
    train_index = np.arange(50)

    adapter = SklearnScalerAdapter(StandardScaler(), ["num"])
    adapter.fit(frame.iloc[train_index])
    transformed = adapter.transform(frame)

    train_mean = float(transformed.iloc[train_index]["num"].mean())
    assert abs(train_mean) < 1e-9, "training rows should be centred by their own statistics"
    assert transformed.iloc[50:]["num"].mean() > 100, "held-out rows must not be re-centred"


def test_target_encoder_uses_training_prior_for_unseen_categories():
    train = pd.DataFrame({"cat": ["a", "a", "b", "b"]})
    y = np.array([1, 0, 1, 1])
    encoder = SmoothedTargetEncoder(["cat"], smoothing=1.0).fit(train, y)
    unseen = pd.DataFrame({"cat": ["zzz"]})
    encoded = encoder.transform(unseen)
    assert encoded["cat"].iloc[0] == pytest.approx(float(y.mean()))


def test_target_encoder_requires_labels():
    with pytest.raises(ValueError, match="requires the training labels"):
        SmoothedTargetEncoder(["cat"]).fit(pd.DataFrame({"cat": ["a"]}), None)


def test_hash_encoder_is_stable_across_calls():
    frame = pd.DataFrame({"cat": ["alpha", "beta", "alpha"]})
    encoder = HashEncoder(["cat"], n_buckets=64)
    first = encoder.transform(frame)["cat"].tolist()
    second = encoder.transform(frame)["cat"].tolist()
    assert first == second
    assert first[0] == first[2], "the same category must hash to the same bucket"


def test_median_imputer_uses_training_medians():
    train = pd.DataFrame({"num": [1.0, 2.0, 3.0]})
    imputer = MedianImputer(["num"]).fit(train)
    out = imputer.transform(pd.DataFrame({"num": [np.nan, 1000.0]}))
    assert out["num"].iloc[0] == pytest.approx(2.0)


def test_pca_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame(rng.normal(size=(100, 8)), columns=[f"V{i}" for i in range(8)])
    train_index = np.arange(50)
    pca = TrainFittedPCA([f"V{i}" for i in range(8)], n_components=3, prefix="pc")
    pca.fit(frame.iloc[train_index])
    out = pca.transform(frame)
    assert [c for c in out.columns if c.startswith("pc_")] == ["pc_000", "pc_001", "pc_002"]
    assert not any(c.startswith("V") for c in out.columns)
    assert pca.pca_.n_samples_ == 50


def test_pca_transform_before_fit_is_refused():
    pca = TrainFittedPCA(["V0"], n_components=1)
    pca.used_columns_ = ["V0"]
    with pytest.raises(LeakageError, match="before fit"):
        pca.transform(pd.DataFrame({"V0": [1.0]}))
