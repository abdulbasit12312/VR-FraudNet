"""Preprocessing primitives with hard train-only fitting guards.

Manuscript Section 3.2: "all preprocessing operations were defined separately
for each dataset, fitted only on the corresponding training partition, and
applied unchanged to validation and test data. The same leakage-controlled
protocol was used for VR-FraudNet and all baselines."

The guard implemented here makes that statement mechanically enforceable: a
:class:`TrainOnlyPipeline` records the exact index set it was fitted on and
refuses to be fitted twice or to be fitted on indices outside the declared
training partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

import numpy as np
import pandas as pd

from vrfraudnet.errors import LeakageError


class Transformer(Protocol):
    """Minimal transformer protocol used by :class:`TrainOnlyPipeline`."""

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = ...) -> "Transformer": ...

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame: ...


@dataclass
class TrainOnlyPipeline:
    """A sequence of transformers that may only ever be fitted on training rows.

    Parameters
    ----------
    steps:
        Ordered ``(name, transformer)`` pairs.
    train_index:
        The row labels that constitute the training partition. Any attempt to
        fit on a row outside this set raises :class:`LeakageError`.
    """

    steps: list[tuple[str, Transformer]]
    train_index: np.ndarray
    _fitted: bool = field(default=False, init=False)
    _fit_signature: tuple[int, int] | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "TrainOnlyPipeline":
        """Fit every step on ``frame``, which must contain only training rows."""
        if self._fitted:
            raise LeakageError(
                "TrainOnlyPipeline has already been fitted. Refitting on a different "
                "partition is the classic leakage bug; construct a new pipeline instead."
            )
        offending = np.setdiff1d(np.asarray(frame.index), self.train_index, assume_unique=False)
        if offending.size:
            raise LeakageError(
                f"attempt to fit preprocessing on {offending.size} row(s) outside the declared "
                f"training partition (first offending index: {offending[0]!r})"
            )
        current = frame
        for _, step in self.steps:
            step.fit(current, y)
            current = step.transform(current)
        self._fitted = True
        self._fit_signature = (int(frame.shape[0]), int(frame.shape[1]))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted steps unchanged to any partition."""
        if not self._fitted:
            raise LeakageError("TrainOnlyPipeline.transform called before fit")
        current = frame
        for _, step in self.steps:
            current = step.transform(current)
        return current

    @property
    def fitted(self) -> bool:
        return self._fitted

    @property
    def fit_signature(self) -> tuple[int, int] | None:
        """(rows, columns) seen at fit time; used by leakage tests."""
        return self._fit_signature


class SmoothedTargetEncoder:
    """Smoothed target encoding (manuscript Table 2, D1 categorical encoding).

    ``encoding(c) = (n_c * mean_c + m * prior) / (n_c + m)``

    The smoothing weight ``m`` is not stated by the manuscript; the default of
    20 is an ASSUMPTION recorded as L-06. Unseen categories at transform time
    fall back to the training prior, which is the only choice that cannot leak.
    """

    def __init__(self, columns: Sequence[str], *, smoothing: float = 20.0) -> None:
        self.columns = list(columns)
        self.smoothing = float(smoothing)
        self.prior_: float | None = None
        self.maps_: dict[str, dict[Any, float]] = {}

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "SmoothedTargetEncoder":
        if y is None:
            raise ValueError("SmoothedTargetEncoder requires the training labels")
        y = np.asarray(y, dtype=float)
        if y.shape[0] != frame.shape[0]:
            raise ValueError("label vector length does not match the training frame")
        self.prior_ = float(y.mean())
        for column in self.columns:
            if column not in frame.columns:
                continue
            grouped = pd.DataFrame({"c": frame[column].to_numpy(), "y": y}).groupby("c")["y"]
            counts = grouped.count()
            means = grouped.mean()
            smoothed = (counts * means + self.smoothing * self.prior_) / (counts + self.smoothing)
            self.maps_[column] = smoothed.to_dict()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.prior_ is None:
            raise LeakageError("SmoothedTargetEncoder.transform called before fit")
        out = frame.copy()
        for column, mapping in self.maps_.items():
            if column in out.columns:
                out[column] = out[column].map(mapping).astype(float).fillna(self.prior_)
        return out


class HashEncoder:
    """64-bit feature hashing for high-cardinality categoricals (Table 2, D2/D3).

    Uses a stable BLAKE2b digest rather than Python's salted ``hash`` so the
    encoding is reproducible across processes and platforms.
    """

    def __init__(self, columns: Sequence[str], *, n_buckets: int = 1024) -> None:
        self.columns = list(columns)
        self.n_buckets = int(n_buckets)

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "HashEncoder":
        # Hashing is stateless by construction, which is precisely why the
        # manuscript can use it without a leakage risk.
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        import hashlib

        out = frame.copy()
        for column in self.columns:
            if column not in out.columns:
                continue
            values = out[column].astype(str).to_numpy()
            buckets = np.fromiter(
                (
                    int.from_bytes(
                        hashlib.blake2b(v.encode("utf-8"), digest_size=8).digest(), "big"
                    )
                    % self.n_buckets
                    for v in values
                ),
                dtype=np.int64,
                count=values.size,
            )
            out[column] = buckets
        return out


class ColumnDropper:
    """Drop named columns, recording why (leakage control vs dimensionality)."""

    def __init__(self, columns: Iterable[str], *, reason: str) -> None:
        self.columns = list(columns)
        self.reason = reason

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "ColumnDropper":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in self.columns if c in frame.columns]
        return frame.drop(columns=present)


class MedianImputer:
    """Column-median imputation fitted on training rows only (Table 2, D1/D3)."""

    def __init__(self, columns: Sequence[str] | None = None) -> None:
        self.columns = list(columns) if columns is not None else None
        self.medians_: dict[str, float] = {}

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "MedianImputer":
        cols = self.columns or list(frame.select_dtypes(include=[np.number]).columns)
        self.medians_ = {c: float(np.nanmedian(frame[c].to_numpy(dtype=float))) for c in cols}
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.medians_:
            raise LeakageError("MedianImputer.transform called before fit")
        out = frame.copy()
        for column, value in self.medians_.items():
            if column in out.columns:
                out[column] = out[column].fillna(value)
        return out


class SklearnScalerAdapter:
    """Adapt a scikit-learn scaler to the frame-in/frame-out transformer protocol."""

    def __init__(self, scaler: Any, columns: Sequence[str] | None = None) -> None:
        self.scaler = scaler
        self.columns = list(columns) if columns is not None else None
        self.columns_: list[str] = []

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "SklearnScalerAdapter":
        self.columns_ = self.columns or list(frame.select_dtypes(include=[np.number]).columns)
        self.scaler.fit(frame[self.columns_].to_numpy(dtype=float))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[self.columns_] = self.scaler.transform(out[self.columns_].to_numpy(dtype=float))
        return out


class TrainFittedPCA:
    """PCA fitted on training rows only (manuscript Table 2, D3: 339 V-cols -> 50)."""

    def __init__(self, columns: Sequence[str], n_components: int, *, prefix: str = "pca") -> None:
        self.columns = list(columns)
        self.n_components = int(n_components)
        self.prefix = prefix
        self.pca_: Any | None = None
        self.used_columns_: list[str] = []

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "TrainFittedPCA":
        from sklearn.decomposition import PCA

        self.used_columns_ = [c for c in self.columns if c in frame.columns]
        if not self.used_columns_:
            return self
        n_components = min(self.n_components, len(self.used_columns_), frame.shape[0])
        self.pca_ = PCA(n_components=n_components, random_state=0)
        self.pca_.fit(np.nan_to_num(frame[self.used_columns_].to_numpy(dtype=float)))
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.used_columns_:
            return frame
        if self.pca_ is None:
            raise LeakageError("TrainFittedPCA.transform called before fit")
        block = np.nan_to_num(frame[self.used_columns_].to_numpy(dtype=float))
        projected = self.pca_.transform(block)
        out = frame.drop(columns=self.used_columns_)
        for i in range(projected.shape[1]):
            out[f"{self.prefix}_{i:03d}"] = projected[:, i]
        return out
