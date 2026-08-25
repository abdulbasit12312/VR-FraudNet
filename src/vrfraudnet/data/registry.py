"""Dataset registry: the facts stated in manuscript Table 1.

This module holds *declared* dataset characteristics only. Nothing here is a
substitute for the data itself; the numbers exist so that
``scripts/prepare_data.py`` can verify that a locally prepared dataset actually
matches what the manuscript describes, and fail loudly when it does not.

Manuscript Table 1, "Public fraud-detection benchmark characteristics and
leakage-aware study splits".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

from vrfraudnet.errors import ConfigurationError

Modality = Literal["tabular", "graph", "tabular+time", "graph+time"]
Role = Literal["primary", "stress"]


@dataclass(frozen=True)
class DatasetSpec:
    """Declared characteristics of one benchmark.

    Attributes
    ----------
    n_rows:
        Row count as stated in Table 1. For graph datasets this is the count of
        the unit the manuscript names first (nodes for D2/D5, transaction nodes
        for D4).
    declared_feature_count:
        Input-feature count as stated in Table 1. Note that Table 2 reports a
        possibly different "output to model" count after preprocessing; both are
        recorded because they disagree for D1 and D3 (audit findings A-08, A-09).
    model_input_count:
        The "Output to model" row of Table 2, or ``None`` when not stated.
    fraud_rate:
        Prevalence as a fraction (Table 1 reports percentages).
    role:
        ``primary`` for D1-D3 (the predictive benchmark suite) and ``stress``
        for D4-D5, per manuscript Section 3.1.
    """

    dataset_id: str
    name: str
    modality: Modality
    n_rows: int
    declared_feature_count: int | None
    model_input_count: int | None
    fraud_rate: float
    time_span: str
    split_rule: str
    role: Role
    official_source: str
    graph: bool
    has_free_text_field: bool
    notes: str = ""
    extra: Mapping[str, object] = field(default_factory=dict)

    @property
    def recall_at_top_1pct_ceiling(self) -> float:
        """Maximum attainable Recall@top-1% given the declared prevalence.

        Reviewing the top 1% of transactions can capture at most ``0.01 / p`` of
        all fraud cases, where ``p`` is the fraud prevalence. Values above this
        ceiling are arithmetically impossible. See docs/MANUSCRIPT_AUDIT.md
        finding A-01, which uses this bound to reject the D3 column of Table 5(c).
        """
        return min(1.0, 0.01 / self.fraud_rate)


DATASETS: dict[str, DatasetSpec] = {
    "D1": DatasetSpec(
        dataset_id="D1",
        name="BAF - Bank Account Fraud (Base variant)",
        modality="tabular",
        n_rows=1_000_000,
        declared_feature_count=30,
        model_input_count=30,
        fraud_rate=0.0110,
        time_span="8 months (month 0-7)",
        split_rule="train month 0-5; test month 6-7; validation = last 10% of train, chronological",
        role="primary",
        official_source="https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022",
        graph=False,
        has_free_text_field=False,
        notes=(
            "Table 2 lists both 'exclude protected attrs' and 'Output to model: 30 features'. "
            "Those two statements cannot both hold if BAF has 30 predictors before exclusion. "
            "See audit finding A-08."
        ),
        extra={"month_column": "month", "label_column": "fraud_bool"},
    ),
    "D2": DatasetSpec(
        dataset_id="D2",
        name="AMLworld HI-Small",
        modality="graph",
        n_rows=515_088,
        declared_feature_count=9,
        model_input_count=None,
        fraud_rate=0.0010,
        time_span="10 days",
        split_rule="temporal 60/20/20 by transaction timestamp",
        role="primary",
        official_source="https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        graph=True,
        has_free_text_field=False,
        notes="9 graph attributes plus entity metadata; 5,078,345 transactions as edges.",
        extra={"n_edges": 5_078_345, "unit": "accounts as nodes, transactions as edges"},
    ),
    "D3": DatasetSpec(
        dataset_id="D3",
        name="IEEE-CIS Fraud Detection (Vesta)",
        modality="tabular+time",
        n_rows=590_540,
        declared_feature_count=435,
        model_input_count=109,
        fraud_rate=0.0350,
        time_span="~6 months",
        split_rule=(
            "TransactionDT < 0.8-quantile train; >= 0.8 test; "
            "5-fold time-series CV with expanding window"
        ),
        role="primary",
        official_source="https://www.kaggle.com/competitions/ieee-fraud-detection/data",
        graph=False,
        has_free_text_field=False,
        notes=(
            "394 transaction columns + 41 identity columns joined on TransactionID. "
            "The stated preprocessing operations do not reconstruct the declared "
            "'109 features'. See audit finding A-09."
        ),
        extra={"n_transaction_cols": 394, "n_identity_cols": 41, "join_key": "TransactionID"},
    ),
    "D4": DatasetSpec(
        dataset_id="D4",
        name="Elliptic++ transaction graph",
        modality="graph+time",
        n_rows=203_769,
        declared_feature_count=183,
        model_input_count=94,
        fraud_rate=0.0223,
        time_span="49 timesteps (~2 weeks each)",
        split_rule="strict-inductive train t=1-34; test t=35-49; pre/post-shock evaluation",
        role="stress",
        official_source="https://github.com/git-disl/EllipticPlusPlus",
        graph=True,
        has_free_text_field=False,
        notes="183 - 89 pre-aggregated features = 94 model inputs; internally consistent.",
        extra={"n_edges": 234_355, "n_dropped_aggregated": 89, "n_timesteps": 49},
    ),
    "D5": DatasetSpec(
        dataset_id="D5",
        name="DGraph-Fin (Finvolution)",
        modality="graph",
        n_rows=3_700_550,
        declared_feature_count=17,
        model_input_count=17,
        fraud_rate=0.0130,
        time_span="full Finvolution P2P platform history",
        split_rule="temporal node-arrival split; background nodes retained only for message passing",
        role="stress",
        official_source="https://dgraph.xinye.com/dataset",
        graph=True,
        has_free_text_field=False,
        notes="17 node features + 11 edge types + edge timestamps.",
        extra={"n_edges": 4_300_999, "n_edge_types": 11},
    ),
}

#: Datasets used as the primary predictive benchmark suite (manuscript S3.1).
PRIMARY_DATASETS: tuple[str, ...] = ("D1", "D2", "D3")

#: Datasets used only for stress tests (manuscript S3.1).
STRESS_DATASETS: tuple[str, ...] = ("D4", "D5")


def get_spec(dataset_id: str) -> DatasetSpec:
    """Look up a dataset spec by its manuscript identifier (``"D1"`` ... ``"D5"``)."""
    key = dataset_id.upper()
    if key not in DATASETS:
        raise ConfigurationError(
            f"unknown dataset {dataset_id!r}; expected one of {sorted(DATASETS)}"
        )
    return DATASETS[key]


def graph_datasets() -> tuple[str, ...]:
    """Dataset ids for which Stage 0 operates on an explicit graph.

    Manuscript Section 4.8.1: "For D1 and D3, which do not provide an explicit
    transaction graph suitable for Stage 0, the graph encoder was bypassed and
    replaced by the past-only temporal features defined during preprocessing."
    """
    return tuple(k for k, v in DATASETS.items() if v.graph)
