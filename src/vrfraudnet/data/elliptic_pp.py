"""D4 Elliptic++ transaction-graph preparation.

Manuscript Table 2, column "D4 Elliptic++ tx graph":

===========================  ==================================================
Numeric scaling              StandardScaler per timestep
Dimensionality reduction     drop 89 pre-aggregated features
Temporal handling            strict-inductive train t=1-34
Graph construction           tx-nodes -> money-flow edges
Leakage mitigation 1         drop 89 pre-aggregated features
Leakage mitigation 2         strict-inductive at training
Leakage mitigation 3         unlabeled nodes never pseudo-labelled
Standardisation-fit policy   train timesteps
Output to model              94 features + graph G_t
===========================  ==================================================

183 - 89 = 94: this is the one dataset whose declared feature arithmetic is
internally consistent.

Elliptic(++) convention: the first 94 transaction features are local features
and the remaining 72 (Elliptic) / 89 (Elliptic++) are neighbourhood aggregates.
The manuscript drops the aggregates because they summarise a node's
neighbourhood and therefore encode information a strict-inductive protocol is
supposed to withhold.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrfraudnet.config import Config
from vrfraudnet.data.splits import strict_inductive_split
from vrfraudnet.data.types import LeakageControls, PreparedDataset, TemporalGraph
from vrfraudnet.errors import MissingArtefactError

#: Elliptic label encoding: 1 = illicit, 2 = licit, 3 / "unknown" = unlabelled.
ILLICIT_CODE = 1
LICIT_CODE = 2


def load_raw(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load features, classes and the edge list from a local Elliptic++ root."""
    root = Path(root)
    for base in (root, root / "D4", root / "elliptic_pp"):
        feats = base / "txs_features.csv"
        classes = base / "txs_classes.csv"
        edges = base / "txs_edgelist.csv"
        if feats.exists() and classes.exists() and edges.exists():
            return pd.read_csv(feats), pd.read_csv(classes), pd.read_csv(edges)
    raise MissingArtefactError(
        f"Elliptic++ transaction files not found under {root}. Expected "
        "txs_features.csv, txs_classes.csv and txs_edgelist.csv. Obtain them from "
        "https://github.com/git-disl/EllipticPlusPlus. See docs/DATA_AVAILABILITY.md."
    )


def prepare(
    features_frame: pd.DataFrame,
    classes_frame: pd.DataFrame,
    edges_frame: pd.DataFrame,
    config: Config,
) -> PreparedDataset:
    """Build the strict-inductive D4 partition, dropping aggregate features."""
    id_col = features_frame.columns[0]
    time_col = _detect_timestep_column(features_frame)

    merged = features_frame.merge(classes_frame, on=id_col, how="left")
    class_col = [c for c in merged.columns if c.lower() in {"class", "label"}][0]

    # Only labelled nodes become supervised targets. Unlabelled nodes are kept
    # in the graph for message passing but never pseudo-labelled
    # (manuscript Table 2, leakage mitigation 3).
    raw_class = merged[class_col].astype(str)
    labelled_mask = raw_class.isin({str(ILLICIT_CODE), str(LICIT_CODE)}).to_numpy()
    labels = (raw_class == str(ILLICIT_CODE)).astype(np.int64).to_numpy()

    timesteps = merged[time_col].to_numpy(dtype=np.int64)

    feature_columns = [c for c in merged.columns if c not in {id_col, time_col, class_col}]
    n_keep = int(config.require("preprocess.n_local_features"))
    if len(feature_columns) < n_keep:
        raise MissingArtefactError(
            f"D4 frame has {len(feature_columns)} feature columns; expected at least {n_keep}"
        )
    kept = feature_columns[:n_keep]
    dropped = feature_columns[n_keep:]

    split = strict_inductive_split(
        timesteps,
        train_timesteps=tuple(config.require("split.train_timesteps")),
        test_timesteps=tuple(config.require("split.test_timesteps")),
    )
    # Restrict every partition to labelled nodes.
    labelled_idx = set(np.flatnonzero(labelled_mask).tolist())
    split = _restrict_to_labelled(split, labelled_idx)

    features = merged[kept].astype(float)
    features = _standardise_per_timestep(features, timesteps, train_idx=np.asarray(split.train))

    node_id_to_row = {int(v): i for i, v in enumerate(merged[id_col].to_numpy())}
    src = edges_frame.iloc[:, 0].map(node_id_to_row)
    dst = edges_frame.iloc[:, 1].map(node_id_to_row)
    valid = src.notna() & dst.notna()
    src_arr = src[valid].to_numpy(dtype=np.int64)
    dst_arr = dst[valid].to_numpy(dtype=np.int64)
    # An edge becomes observable at the later of its two endpoints' timesteps.
    edge_time = np.maximum(timesteps[src_arr], timesteps[dst_arr]).astype(np.float64)

    graph = TemporalGraph(
        edge_index=np.vstack([src_arr, dst_arr]),
        edge_time=edge_time,
        num_nodes=int(merged.shape[0]),
        node_time=timesteps.astype(np.float64),
    )

    leakage = LeakageControls(
        dropped_columns=list(dropped),
        controls_applied=[
            f"dropped {len(dropped)} pre-aggregated neighbourhood features",
            "strict-inductive: test-period nodes are not supervised targets during training",
            "unlabelled nodes never pseudo-labelled",
            "per-timestep standardisation fitted on training timesteps only",
        ],
        standardisation_fit_policy="train timesteps",
        temporal_handling="strict-inductive train t=1-34, test t=35-49",
    )

    return PreparedDataset(
        dataset_id="D4",
        features=features,
        labels=labels,
        times=timesteps.astype(np.float64),
        split=split,
        graph=graph,
        leakage=leakage,
        metadata={
            "n_features_after_preprocessing": int(features.shape[1]),
            "declared_output_to_model": 94,
            "n_dropped_aggregated": len(dropped),
            "n_labelled_nodes": int(labelled_mask.sum()),
        },
    )


def _detect_timestep_column(frame: pd.DataFrame) -> str:
    for name in ("Time step", "time_step", "timestep", "Timestep"):
        if name in frame.columns:
            return name
    # Elliptic's raw feature file is headerless in some releases; the timestep is
    # the second column by construction.
    return frame.columns[1]


def _restrict_to_labelled(split, labelled: set[int]):
    from vrfraudnet.data.splits import Split

    def keep(idx: np.ndarray) -> np.ndarray:
        return np.asarray([i for i in idx.tolist() if i in labelled], dtype=np.int64)

    return Split(
        train=keep(split.train),
        validation=keep(split.validation),
        test=keep(split.test),
        calibration=None if split.calibration is None else keep(split.calibration),
        description=split.description + " (restricted to labelled nodes)",
    )


def _standardise_per_timestep(
    features: pd.DataFrame, timesteps: np.ndarray, *, train_idx: np.ndarray
) -> pd.DataFrame:
    """Standardise within each timestep using statistics from training rows only.

    For a test timestep, no training rows exist by construction, so the pooled
    training statistics are used. Using the timestep's own rows would fit a
    transformation on test data and is therefore refused.
    """
    out = features.copy()
    values = out.to_numpy(dtype=float)
    train_mask = np.zeros(values.shape[0], dtype=bool)
    train_mask[train_idx] = True

    pooled_mean = values[train_mask].mean(axis=0) if train_mask.any() else values.mean(axis=0)
    pooled_std = values[train_mask].std(axis=0) if train_mask.any() else values.std(axis=0)
    pooled_std = np.where(pooled_std == 0, 1.0, pooled_std)

    scaled = np.empty_like(values)
    for step in np.unique(timesteps):
        rows = timesteps == step
        train_rows = rows & train_mask
        if train_rows.sum() > 1:
            mean = values[train_rows].mean(axis=0)
            std = values[train_rows].std(axis=0)
            std = np.where(std == 0, 1.0, std)
        else:
            mean, std = pooled_mean, pooled_std
        scaled[rows] = (values[rows] - mean) / std

    out.iloc[:, :] = scaled
    return out
