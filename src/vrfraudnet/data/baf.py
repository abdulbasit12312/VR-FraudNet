"""D1 BAF (Bank Account Fraud, Base variant) preparation.

Manuscript Table 2, column "D1 BAF":

===========================  ==================================================
Numeric scaling              RobustScaler
Numeric imputation           column median
Categorical encoding         smoothed target-encode
Dimensionality reduction     -
Temporal handling            retain month; chronological batching
Graph construction           -
Leakage mitigation 1         drop ``fraud_bool`` from inputs
Leakage mitigation 2         exclude protected attrs
Leakage mitigation 3         preserve train/test month boundary
Imbalance handling           focal + class-balanced
Standardisation-fit policy   train fold
Output to model              30 features
===========================  ==================================================

AUDIT NOTE (A-08). The BAF Base variant ships 30 predictor columns plus the
``fraud_bool`` label plus ``month``. Table 2 asks for both "exclude protected
attrs" and "output to model: 30 features"; those cannot both hold, because
excluding any protected attribute reduces the count below 30. This module
implements the *exclusion* (the leakage control) and reports the resulting
feature count honestly rather than padding back to 30.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from vrfraudnet.config import Config
from vrfraudnet.data.preprocess_base import (
    ColumnDropper,
    MedianImputer,
    SklearnScalerAdapter,
    SmoothedTargetEncoder,
    TrainOnlyPipeline,
)
from vrfraudnet.data.splits import chronological_holdout_by_group
from vrfraudnet.data.types import LeakageControls, PreparedDataset
from vrfraudnet.errors import MissingArtefactError

#: Protected attributes excluded from D1 inputs (manuscript Table 2, leakage
#: mitigation 2). BAF documents ``customer_age`` and ``employment_status`` among
#: its sensitive attributes; the manuscript names the age group as the protected
#: subgroup used in the fairness analysis (Supplementary Table S2).
DEFAULT_PROTECTED_ATTRIBUTES: tuple[str, ...] = ("customer_age",)

LABEL_COLUMN = "fraud_bool"
MONTH_COLUMN = "month"


def load_raw(root: str | Path) -> pd.DataFrame:
    """Load the BAF Base CSV from a local dataset root.

    The dataset is not redistributed with this repository. See
    docs/DATA_AVAILABILITY.md for the official download location.
    """
    root = Path(root)
    candidates = [root / "Base.csv", root / "baf" / "Base.csv", root / "D1" / "Base.csv"]
    for candidate in candidates:
        if candidate.exists():
            return pd.read_csv(candidate)
    raise MissingArtefactError(
        "BAF Base.csv not found. Searched:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nDownload it from "
        "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022 "
        "and place Base.csv under datasets/raw/D1/. See docs/DATA_AVAILABILITY.md."
    )


def prepare(frame: pd.DataFrame, config: Config) -> PreparedDataset:
    """Apply the Table 2 D1 pipeline and build the month-based split.

    All fitting happens on the training months only, enforced by
    :class:`~vrfraudnet.data.preprocess_base.TrainOnlyPipeline`.
    """
    frame = frame.reset_index(drop=True)
    for required in (LABEL_COLUMN, MONTH_COLUMN):
        if required not in frame.columns:
            raise MissingArtefactError(
                f"D1 frame is missing the required column {required!r}; "
                f"columns present: {sorted(frame.columns)[:10]}..."
            )

    labels = frame[LABEL_COLUMN].to_numpy(dtype=np.int64)
    months = frame[MONTH_COLUMN].to_numpy(dtype=np.int64)

    split = chronological_holdout_by_group(
        months,
        train_groups=tuple(config.require("split.train_months")),
        test_groups=tuple(config.require("split.test_months")),
        validation_fraction=float(config.require("split.validation_fraction")),
    )

    protected = tuple(config.get("preprocess.protected_attributes", DEFAULT_PROTECTED_ATTRIBUTES))
    # Leakage mitigation 1 and 2: the label never enters the inputs and the
    # protected attributes are removed before any transformer sees them.
    inputs = frame.drop(columns=[LABEL_COLUMN])
    categorical = [c for c in inputs.columns if inputs[c].dtype == object]

    pipeline = TrainOnlyPipeline(
        steps=[
            ("drop_protected", ColumnDropper(protected, reason="leakage mitigation 2")),
            ("impute_median", MedianImputer()),
            ("target_encode", SmoothedTargetEncoder(categorical)),
            ("robust_scale", SklearnScalerAdapter(RobustScaler())),
        ],
        train_index=np.asarray(split.train),
    )
    pipeline.fit(inputs.iloc[split.train], labels[split.train])
    features = pipeline.transform(inputs)

    leakage = LeakageControls(
        dropped_columns=[LABEL_COLUMN, *[p for p in protected if p in inputs.columns]],
        controls_applied=[
            "drop fraud_bool from inputs",
            "exclude protected attributes",
            "preserve train/test month boundary",
            "all transformers fitted on training months only",
        ],
        standardisation_fit_policy="train fold (months 0-5 minus validation tail)",
        temporal_handling="retain month; chronological batching",
    )

    return PreparedDataset(
        dataset_id="D1",
        features=features,
        labels=labels,
        times=months,
        split=split,
        graph=None,
        leakage=leakage,
        metadata={
            "n_features_after_preprocessing": int(features.shape[1]),
            "declared_output_to_model": 30,
            "audit_note": "A-08: declared 30 features conflicts with protected-attribute exclusion",
            "protected_attributes_excluded": list(protected),
        },
    )
