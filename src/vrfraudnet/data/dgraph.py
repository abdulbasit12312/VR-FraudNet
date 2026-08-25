"""D5 DGraph-Fin preparation.

Manuscript Table 2, column "D5 DGraph-Fin":

===========================  ==================================================
Numeric scaling              StandardScaler
Categorical encoding         one-hot + min-max timestamp
Temporal handling            node-arrival-timestamp split
Graph construction           users -> typed temporal edges
Leakage mitigation 1         mask loss to labeled nodes
Leakage mitigation 2         no ID leakage across split
Leakage mitigation 3         bg-nodes excluded from loss
Standardisation-fit policy   train fold
Output to model              17 features + heterogeneous edges
===========================  ==================================================

DGraph-Fin ships as a single ``dgraphfin.npz``. Background nodes (classes 2 and
3 in the official release) are retained for message passing and excluded from
every supervised index array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vrfraudnet.config import Config
from vrfraudnet.data.splits import node_arrival_split
from vrfraudnet.data.types import LeakageControls, PreparedDataset, TemporalGraph
from vrfraudnet.errors import MissingArtefactError

#: Official DGraph-Fin node classes: 0 = normal, 1 = fraud, 2/3 = background.
FRAUD_CLASS = 1
FOREGROUND_CLASSES = (0, 1)


def load_raw(root: str | Path) -> dict[str, Any]:
    """Load ``dgraphfin.npz`` from a local dataset root."""
    root = Path(root)
    for candidate in (root / "dgraphfin.npz", root / "D5" / "dgraphfin.npz"):
        if candidate.exists():
            with np.load(candidate) as handle:
                return {key: handle[key] for key in handle.files}
    raise MissingArtefactError(
        f"dgraphfin.npz not found under {root}. Request access at "
        "https://dgraph.xinye.com/dataset. See docs/DATA_AVAILABILITY.md."
    )


def prepare(raw: dict[str, Any], config: Config) -> PreparedDataset:
    """Build the node-arrival split and the typed temporal edge set."""
    for key in ("x", "y", "edge_index"):
        if key not in raw:
            raise MissingArtefactError(
                f"dgraphfin archive is missing array {key!r}; found {sorted(raw)}"
            )

    x = np.asarray(raw["x"], dtype=np.float64)
    y_raw = np.asarray(raw["y"]).astype(np.int64)
    edge_index = np.asarray(raw["edge_index"], dtype=np.int64)
    if edge_index.shape[0] != 2:
        edge_index = edge_index.T

    edge_time = np.asarray(
        raw.get("edge_timestamp", np.zeros(edge_index.shape[1])), dtype=np.float64
    )
    edge_type = raw.get("edge_type")
    edge_type = None if edge_type is None else np.asarray(edge_type, dtype=np.int64)

    labelled_mask = np.isin(y_raw, FOREGROUND_CLASSES)
    labels = (y_raw == FRAUD_CLASS).astype(np.int64)

    # Node arrival time. When the release does not ship a node timestamp, the
    # earliest incident edge time is used; that is a derivation, not an
    # invention, and is recorded as an ASSUMPTION (L-05).
    if "node_timestamp" in raw:
        arrival = np.asarray(raw["node_timestamp"], dtype=np.float64)
    else:
        arrival = np.full(x.shape[0], np.inf, dtype=np.float64)
        np.minimum.at(arrival, edge_index[0], edge_time)
        np.minimum.at(arrival, edge_index[1], edge_time)
        arrival[~np.isfinite(arrival)] = np.nanmax(edge_time) if edge_time.size else 0.0

    split = node_arrival_split(
        arrival,
        labelled_mask,
        train_fraction=float(config.require("split.train_fraction")),
        validation_fraction=float(config.require("split.validation_fraction")),
    )

    mean = x[split.train].mean(axis=0)
    std = x[split.train].std(axis=0)
    std = np.where(std == 0, 1.0, std)
    features = pd.DataFrame(
        (x - mean) / std, columns=[f"f{i:02d}" for i in range(x.shape[1])]
    )

    graph = TemporalGraph(
        edge_index=edge_index,
        edge_time=edge_time,
        num_nodes=int(x.shape[0]),
        node_time=arrival,
        edge_type=edge_type,
    )

    leakage = LeakageControls(
        dropped_columns=[],
        controls_applied=[
            "supervised loss masked to labelled foreground nodes",
            "background nodes retained for message passing but excluded from every index array",
            "node ids never shared across partitions",
            "standardisation fitted on the training fold only",
        ],
        standardisation_fit_policy="train fold",
        temporal_handling="node-arrival-timestamp split",
    )

    return PreparedDataset(
        dataset_id="D5",
        features=features,
        labels=labels,
        times=arrival,
        split=split,
        graph=graph,
        leakage=leakage,
        metadata={
            "n_nodes": int(x.shape[0]),
            "n_edges": int(edge_index.shape[1]),
            "n_labelled": int(labelled_mask.sum()),
            "n_edge_types": 0 if edge_type is None else int(np.unique(edge_type).size),
        },
    )
