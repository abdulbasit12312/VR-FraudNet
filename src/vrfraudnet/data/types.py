"""Common containers returned by every dataset preparation module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from vrfraudnet.data.splits import Split


@dataclass
class TemporalGraph:
    """A timestamped graph used by Stage 0.

    ``edge_index`` has shape ``(2, E)``; ``edge_time`` has shape ``(E,)`` and
    holds the time at which each edge becomes observable. Stage 0 filters edges
    by ``edge_time <= t`` so that no future edge can influence a prediction
    (manuscript S4.1: "only nodes, edges, features, and timestamps observable by
    time t are used").
    """

    edge_index: np.ndarray
    edge_time: np.ndarray
    num_nodes: int
    node_time: np.ndarray | None = None
    edge_type: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {self.edge_index.shape}")
        if self.edge_time.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_time length must equal the number of edges")
        if self.edge_index.size and int(self.edge_index.max()) >= self.num_nodes:
            raise ValueError("edge_index references a node id outside [0, num_nodes)")

    def snapshot(self, t: float) -> "TemporalGraph":
        """Return the sub-graph observable at time ``t`` (edges with time <= t)."""
        keep = self.edge_time <= t
        return TemporalGraph(
            edge_index=self.edge_index[:, keep],
            edge_time=self.edge_time[keep],
            num_nodes=self.num_nodes,
            node_time=self.node_time,
            edge_type=None if self.edge_type is None else self.edge_type[keep],
        )


@dataclass
class LeakageControls:
    """The leakage-mitigation rows of manuscript Table 2, as executed.

    Recorded per dataset and written into every run manifest so a reviewer can
    confirm which controls actually ran rather than which ones were described.
    """

    dropped_columns: list[str] = field(default_factory=list)
    controls_applied: list[str] = field(default_factory=list)
    standardisation_fit_policy: str = ""
    temporal_handling: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dropped_columns": sorted(self.dropped_columns),
            "controls_applied": list(self.controls_applied),
            "standardisation_fit_policy": self.standardisation_fit_policy,
            "temporal_handling": self.temporal_handling,
        }


@dataclass
class PreparedDataset:
    """Everything downstream stages need for one dataset.

    Attributes
    ----------
    features:
        Model-ready feature frame. Row order matches ``labels`` and ``times``.
    labels:
        Binary fraud labels; fraud is the positive class encoded as 1
        (manuscript S4.8.3).
    times:
        Per-row ordering value (month index, timestamp, or timestep).
    split:
        The partition indices produced by :mod:`vrfraudnet.data.splits`.
    graph:
        Temporal graph for D2/D4/D5; ``None`` for the purely tabular D1/D3.
    """

    dataset_id: str
    features: pd.DataFrame
    labels: np.ndarray
    times: np.ndarray
    split: Split
    graph: TemporalGraph | None
    leakage: LeakageControls
    node_index: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.features.shape[0]
        if self.labels.shape[0] != n or self.times.shape[0] != n:
            raise ValueError("features, labels and times must have equal length")
        unique = set(np.unique(self.labels).tolist())
        if not unique <= {0, 1}:
            raise ValueError(f"labels must be binary 0/1, found {sorted(unique)}")

    @property
    def prevalence(self) -> float:
        return float(self.labels.mean())

    def subset(self, idx: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
        """Return ``(X, y)`` for a partition index array."""
        return self.features.iloc[idx], self.labels[idx]
