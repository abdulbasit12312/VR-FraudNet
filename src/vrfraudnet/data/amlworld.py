"""D2 AMLworld HI-Small preparation.

Manuscript Table 2, column "D2 AMLworld HI-Small":

===========================  ==================================================
Numeric scaling              log1p + per-currency standardization
Categorical encoding         one-hot + 64-bit hash
Temporal handling            sort by timestamp; 24 h windows
Graph construction           accounts -> tx-edges; per-bank partition
Leakage mitigation 1         drop pattern-label oracle
Leakage mitigation 2         per-bank label segregation
Leakage mitigation 3         no future edges at training
Imbalance handling           focal + class-balanced
Standardisation-fit policy   train segment
Output to model              graph G_t + 9 edge attrs
===========================  ==================================================

The transaction-level prediction target is the edge (a transaction); accounts
are nodes. Stage 0 therefore embeds the two endpoint accounts of a transaction
using only edges observable at or before that transaction's timestamp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vrfraudnet.config import Config
from vrfraudnet.data.preprocess_base import HashEncoder, TrainOnlyPipeline
from vrfraudnet.data.splits import chronological_fraction_split
from vrfraudnet.data.types import LeakageControls, PreparedDataset, TemporalGraph
from vrfraudnet.errors import MissingArtefactError

#: Columns that leak the AMLworld generator's ground-truth typology directly.
#: Manuscript Table 2, leakage mitigation 1: "drop pattern-label oracle".
PATTERN_ORACLE_COLUMNS: tuple[str, ...] = (
    "Laundering Type",
    "laundering_type",
    "pattern",
    "Pattern",
    "typology",
)

LABEL_CANDIDATES: tuple[str, ...] = ("Is Laundering", "is_laundering", "isLaundering")
TIMESTAMP_CANDIDATES: tuple[str, ...] = ("Timestamp", "timestamp")


def load_raw(root: str | Path) -> pd.DataFrame:
    """Load the HI-Small transaction table from a local dataset root."""
    root = Path(root)
    candidates = [
        root / "HI-Small_Trans.csv",
        root / "D2" / "HI-Small_Trans.csv",
        root / "amlworld" / "HI-Small_Trans.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise MissingArtefactError(
        "AMLworld HI-Small transaction file not found. Searched:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nSee docs/DATA_AVAILABILITY.md for the official source."
    )


def _first_present(frame: pd.DataFrame, names: tuple[str, ...], what: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise MissingArtefactError(
        f"could not locate the {what} column in the D2 frame; tried {names}. "
        f"Columns present: {sorted(frame.columns)}"
    )


def _build_account_index(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, int]:
    """Map (bank, account) pairs to contiguous node ids for source and target."""
    src_bank = frame.get("From Bank", pd.Series([""] * len(frame))).astype(str)
    dst_bank = frame.get("To Bank", pd.Series([""] * len(frame))).astype(str)
    src_acc = frame.iloc[:, frame.columns.get_loc("Account")] if "Account" in frame else None
    if src_acc is None:
        raise MissingArtefactError("D2 frame does not contain an 'Account' column")
    # AMLworld ships two columns literally named 'Account'; pandas suffixes the
    # second one. Resolve both by position when necessary.
    account_cols = [c for c in frame.columns if str(c).startswith("Account")]
    if len(account_cols) < 2:
        raise MissingArtefactError(
            "D2 frame must contain both a sender and a receiver account column; "
            f"found {account_cols}"
        )
    src_key = src_bank + ":" + frame[account_cols[0]].astype(str)
    dst_key = dst_bank + ":" + frame[account_cols[1]].astype(str)

    codes, _ = pd.factorize(pd.concat([src_key, dst_key], ignore_index=True))
    n = len(frame)
    return codes[:n].astype(np.int64), codes[n:].astype(np.int64), int(codes.max()) + 1


def prepare(frame: pd.DataFrame, config: Config) -> PreparedDataset:
    """Apply the Table 2 D2 pipeline, build the account graph and the 60/20/20 split."""
    frame = frame.reset_index(drop=True)
    label_col = _first_present(frame, LABEL_CANDIDATES, "laundering label")
    time_col = _first_present(frame, TIMESTAMP_CANDIDATES, "timestamp")

    labels = frame[label_col].to_numpy(dtype=np.int64)
    times = pd.to_datetime(frame[time_col]).astype("int64").to_numpy() // 10**9

    split = chronological_fraction_split(
        times,
        train_fraction=float(config.require("split.train_fraction")),
        validation_fraction=float(config.require("split.validation_fraction")),
    )

    dropped = [c for c in PATTERN_ORACLE_COLUMNS if c in frame.columns]
    inputs = frame.drop(columns=[label_col, *dropped])

    # Numeric scaling: log1p then per-currency standardisation, with the
    # per-currency location/scale estimated on the training segment only.
    amount_cols = [c for c in inputs.columns if "Amount" in str(c)]
    currency_cols = [c for c in inputs.columns if "Currency" in str(c)]
    per_currency = _PerCurrencyLog1pStandardiser(amount_cols, currency_cols[0] if currency_cols else None)

    categorical = [c for c in inputs.columns if inputs[c].dtype == object and c not in currency_cols]
    pipeline = TrainOnlyPipeline(
        steps=[
            ("log1p_per_currency", per_currency),
            ("hash_categorical", HashEncoder(categorical, n_buckets=int(config.get("preprocess.hash_buckets", 1024)))),
        ],
        train_index=np.asarray(split.train),
    )
    pipeline.fit(inputs.iloc[split.train], labels[split.train])
    features = pipeline.transform(inputs)
    features = features.select_dtypes(include=[np.number]).copy()

    src, dst, n_nodes = _build_account_index(frame)
    graph = TemporalGraph(
        edge_index=np.vstack([src, dst]),
        edge_time=times.astype(np.float64),
        num_nodes=n_nodes,
    )

    leakage = LeakageControls(
        dropped_columns=[label_col, *dropped],
        controls_applied=[
            "drop pattern-label oracle columns",
            "no future edges at training (Stage 0 snapshots by transaction timestamp)",
            "per-currency scaling fitted on the training segment only",
        ],
        standardisation_fit_policy="train segment",
        temporal_handling="sort by timestamp; 24 h windows",
    )

    return PreparedDataset(
        dataset_id="D2",
        features=features,
        labels=labels,
        times=times.astype(np.float64),
        split=split,
        graph=graph,
        leakage=leakage,
        node_index=src,
        metadata={
            "n_accounts": n_nodes,
            "n_transactions": int(len(frame)),
            "edge_endpoints": "source account -> destination account",
        },
    )


class _PerCurrencyLog1pStandardiser:
    """``log1p`` followed by standardisation within each currency (Table 2, D2)."""

    def __init__(self, amount_columns: list[str], currency_column: str | None) -> None:
        self.amount_columns = amount_columns
        self.currency_column = currency_column
        self.stats_: dict[tuple[str, str], tuple[float, float]] = {}

    def fit(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> "_PerCurrencyLog1pStandardiser":
        for column in self.amount_columns:
            values = np.log1p(np.clip(frame[column].to_numpy(dtype=float), 0.0, None))
            if self.currency_column is None:
                self.stats_[(column, "__all__")] = (float(values.mean()), float(values.std() or 1.0))
                continue
            currencies = frame[self.currency_column].astype(str).to_numpy()
            for currency in np.unique(currencies):
                block = values[currencies == currency]
                self.stats_[(column, currency)] = (
                    float(block.mean()),
                    float(block.std() or 1.0),
                )
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        for column in self.amount_columns:
            values = np.log1p(np.clip(out[column].to_numpy(dtype=float), 0.0, None))
            if self.currency_column is None:
                mean, std = self.stats_.get((column, "__all__"), (0.0, 1.0))
                out[column] = (values - mean) / std
                continue
            currencies = out[self.currency_column].astype(str).to_numpy()
            scaled = np.empty_like(values)
            for currency in np.unique(currencies):
                mask = currencies == currency
                # Unseen currency at test time falls back to the global training
                # statistics rather than being standardised with test data.
                mean, std = self.stats_.get(
                    (column, currency), _global_stats(self.stats_, column)
                )
                scaled[mask] = (values[mask] - mean) / std
            out[column] = scaled
        return out


def _global_stats(
    stats: dict[tuple[str, str], tuple[float, float]], column: str
) -> tuple[float, float]:
    """Pooled fallback statistics for a column across all seen currencies."""
    entries = [v for (c, _), v in stats.items() if c == column]
    if not entries:
        return (0.0, 1.0)
    means = float(np.mean([m for m, _ in entries]))
    stds = float(np.mean([s for _, s in entries])) or 1.0
    return means, stds
