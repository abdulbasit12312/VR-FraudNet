"""D3 IEEE-CIS (Vesta) preparation.

Manuscript Table 2, column "D3 IEEE-CIS":

===========================  ==================================================
Numeric scaling              StandardScaler
Numeric imputation           column median
Categorical encoding         64-bit hash + one-hot
Dimensionality reduction     PCA 339 V-cols -> 50
Temporal handling            TransactionDT for ordering only
Leakage mitigation 1         UID-magic disabled
Leakage mitigation 2         identity table imputed in-row only
Leakage mitigation 3         drop D-cols D2-D9, D11-D14
Standardisation-fit policy   train fold
Output to model              109 features
===========================  ==================================================

AUDIT NOTE (A-09). The operations above do not reconstruct exactly 109 model
inputs from the 394 transaction plus 41 identity columns. This module executes
the stated operations and reports the resulting count honestly; it does not
add or drop columns to hit the declared number.

AUDIT NOTE (A-01). Manuscript Table 5(c) reports Recall@top-1% values between
0.4234 and 0.6234 for a dataset with 3.50% fraud prevalence. Reviewing 1% of
transactions can capture at most 1/3.5 = 28.57% of all fraud cases, so those
values are arithmetically unattainable. The metric layer refuses to emit that
column for D3; see :mod:`vrfraudnet.evaluation.metrics`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from vrfraudnet.config import Config
from vrfraudnet.data.preprocess_base import (
    ColumnDropper,
    HashEncoder,
    MedianImputer,
    SklearnScalerAdapter,
    TrainFittedPCA,
    TrainOnlyPipeline,
)
from vrfraudnet.data.splits import quantile_time_split
from vrfraudnet.data.types import LeakageControls, PreparedDataset
from vrfraudnet.errors import MissingArtefactError

LABEL_COLUMN = "isFraud"
TIME_COLUMN = "TransactionDT"
JOIN_KEY = "TransactionID"

#: Manuscript Table 2, leakage mitigation 3 for D3.
DROPPED_D_COLUMNS: tuple[str, ...] = (
    "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D11", "D12", "D13", "D14",
)

#: Columns commonly used to construct the community "magic UID" feature. The
#: manuscript disables that shortcut (leakage mitigation 1); the control is
#: implemented as "no UID feature is constructed", which is the literal reading.
UID_CONSTRUCTION_NOTE = (
    "no client UID feature is derived from card/addr/D1 combinations "
    "(manuscript Table 2, 'UID-magic disabled')"
)


def load_raw(root: str | Path) -> pd.DataFrame:
    """Join the transaction and identity tables on ``TransactionID``."""
    root = Path(root)
    for base in (root, root / "D3", root / "ieee_cis"):
        tx = base / "train_transaction.csv"
        ident = base / "train_identity.csv"
        if tx.exists():
            transactions = pd.read_csv(tx)
            if ident.exists():
                identity = pd.read_csv(ident)
                return transactions.merge(identity, on=JOIN_KEY, how="left")
            return transactions
    raise MissingArtefactError(
        f"IEEE-CIS train_transaction.csv not found under {root}. "
        "Download from https://www.kaggle.com/competitions/ieee-fraud-detection/data "
        "and place the CSVs under datasets/raw/D3/. See docs/DATA_AVAILABILITY.md."
    )


def prepare(frame: pd.DataFrame, config: Config) -> PreparedDataset:
    """Apply the Table 2 D3 pipeline and build the 0.8-quantile time split."""
    frame = frame.reset_index(drop=True)
    for required in (LABEL_COLUMN, TIME_COLUMN):
        if required not in frame.columns:
            raise MissingArtefactError(f"D3 frame is missing required column {required!r}")

    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    times = frame[TIME_COLUMN].to_numpy(dtype=np.float64)

    split = quantile_time_split(
        times,
        test_quantile=float(config.require("split.test_quantile")),
        validation_fraction=float(config.get("split.validation_fraction", 0.10)),
    )

    v_columns = [c for c in frame.columns if str(c).startswith("V") and str(c)[1:].isdigit()]
    # TransactionDT is used for ordering and fold construction only; it must not
    # become a predictor (manuscript S3.1: "rather than as an unrestricted
    # predictive shortcut").
    inputs = frame.drop(columns=[LABEL_COLUMN, TIME_COLUMN, JOIN_KEY], errors="ignore")
    categorical = [c for c in inputs.columns if inputs[c].dtype == object]

    pipeline = TrainOnlyPipeline(
        steps=[
            ("drop_d_columns", ColumnDropper(DROPPED_D_COLUMNS, reason="leakage mitigation 3")),
            ("hash_categorical", HashEncoder(categorical, n_buckets=int(config.get("preprocess.hash_buckets", 1024)))),
            ("impute_median", MedianImputer()),
            ("pca_v_columns", TrainFittedPCA(v_columns, int(config.require("preprocess.pca_components")), prefix="V_pc")),
            ("standard_scale", SklearnScalerAdapter(StandardScaler())),
        ],
        train_index=np.asarray(split.train),
    )
    pipeline.fit(inputs.iloc[split.train], labels[split.train])
    features = pipeline.transform(inputs).select_dtypes(include=[np.number]).copy()

    leakage = LeakageControls(
        dropped_columns=[LABEL_COLUMN, TIME_COLUMN, JOIN_KEY, *DROPPED_D_COLUMNS],
        controls_applied=[
            UID_CONSTRUCTION_NOTE,
            "identity table joined in-row on TransactionID; no cross-row identity imputation",
            "drop D-columns D2-D9 and D11-D14",
            "TransactionDT used for ordering and fold construction only, never as a predictor",
            "PCA and scaler fitted on the training fold only",
        ],
        standardisation_fit_policy="train fold",
        temporal_handling="TransactionDT for ordering only",
    )

    return PreparedDataset(
        dataset_id="D3",
        features=features,
        labels=labels,
        times=times,
        split=split,
        graph=None,
        leakage=leakage,
        metadata={
            "n_features_after_preprocessing": int(features.shape[1]),
            "declared_output_to_model": 109,
            "n_v_columns_reduced": len(v_columns),
            "audit_note": "A-09: declared 109 model inputs is not reconstructible from Table 2",
        },
    )
