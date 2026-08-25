"""Manuscript table builders.

Every builder reads result files and nothing else. If a required run is absent,
the builder raises :class:`TableBuildError` carrying the exact command that
produces the missing artefact. There is no code path in this module that can
emit a number the repository did not compute.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from vrfraudnet.baselines.registry import BASELINE_ORDER
from vrfraudnet.data.registry import PRIMARY_DATASETS, get_spec
from vrfraudnet.evaluation.metrics import aggregate_seeds
from vrfraudnet.io_utils import read_result
from vrfraudnet.seeds import MANUSCRIPT_SEEDS


class TableBuildError(RuntimeError):
    """A table cannot be built from the available results."""


@dataclass
class RenderedTable:
    """A table ready to write out."""

    title: str
    columns: list[str]
    rows: list[list[str]]
    caveats: list[str] = field(default_factory=list)
    subtables: list["RenderedTable"] = field(default_factory=list)


def render_markdown(table: RenderedTable) -> str:
    """Render a table (and its subtables) as GitHub-flavoured Markdown."""
    lines = [f"## {table.title}", ""]
    if table.columns:
        lines.append("| " + " | ".join(table.columns) + " |")
        lines.append("|" + "|".join(["---"] * len(table.columns)) + "|")
        for row in table.rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    for caveat in table.caveats:
        lines.append(f"> {caveat}")
        lines.append("")
    for sub in table.subtables:
        lines.append(render_markdown(sub))
    return "\n".join(lines)


def _load_metrics(results_dir: Path) -> dict[str, dict[str, dict[int, dict[str, float]]]]:
    """``{dataset: {model: {seed: metrics}}}`` from ``results/metrics``."""
    root = results_dir / "metrics"
    if not root.exists():
        raise TableBuildError(
            f"no metrics directory at {root}.\nProduce metrics with, for each dataset "
            "and seed:\n"
            "    python scripts/train.py --dataset D1 --config configs/d1.yaml --seed 42\n"
            "    python scripts/evaluate.py --dataset D1 --config configs/d1.yaml --seed 42"
        )
    table: dict[str, dict[str, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for path in sorted(root.rglob("*.json")):
        document = read_result(path)
        manifest, results = document["manifest"], document["results"]
        table[manifest["dataset"]][manifest["model"]][int(manifest["seed"])] = results
    if not table:
        raise TableBuildError(f"{root} contains no result files")
    return {k: dict(v) for k, v in table.items()}


def _format_mean_std(stats: dict[str, float] | None) -> str:
    if stats is None or not np.isfinite(stats.get("mean", float("nan"))):
        return "n/a"
    return f"{stats['mean']:.4f} ± {stats['std']:.4f}"


def table_5_primary_performance(results_dir: Path) -> RenderedTable:
    """Manuscript Table 5(a-c): primary benchmark performance on D1-D3."""
    metrics = _load_metrics(results_dir)
    subtables: list[RenderedTable] = []
    global_caveats: list[str] = []

    for dataset in PRIMARY_DATASETS:
        if dataset not in metrics:
            raise TableBuildError(
                f"no results for {dataset}. Run:\n"
                f"    python scripts/train.py --dataset {dataset} "
                f"--config configs/{dataset.lower()}.yaml --seed <seed>\n"
                f"    python scripts/evaluate.py --dataset {dataset} "
                f"--config configs/{dataset.lower()}.yaml --seed <seed>"
            )
        spec = get_spec(dataset)
        ceiling = spec.recall_at_top_1pct_ceiling
        show_recall = ceiling >= 0.999

        columns = ["#", "Method", "AUPRC", "ROC-AUC", "F1-score"]
        if show_recall:
            columns.append("Recall@top-1%")
        columns.append("MCC")

        rows: list[list[str]] = []
        ordering = [*BASELINE_ORDER, "vr_fraudnet"]
        present = [m for m in ordering if m in metrics[dataset]]
        for i, model in enumerate(present, start=1):
            per_seed = metrics[dataset][model]
            if len(per_seed) < 2:
                raise TableBuildError(
                    f"{dataset}/{model} has {len(per_seed)} seed(s); a mean ± std needs at "
                    f"least 2 and the manuscript uses 10: {list(MANUSCRIPT_SEEDS)}"
                )
            aggregated = aggregate_seeds(per_seed)
            row = [
                str(i),
                model,
                _format_mean_std(aggregated.get("auprc")),
                _format_mean_std(aggregated.get("roc_auc")),
                _format_mean_std(aggregated.get("f1")),
            ]
            if show_recall:
                row.append(_format_mean_std(aggregated.get("recall_at_top_1pct")))
            row.append(_format_mean_std(aggregated.get("mcc")))
            rows.append(row)

        caveats = [
            f"Seeds used: {sorted({s for m in metrics[dataset].values() for s in m})}."
        ]
        if not show_recall:
            caveats.append(
                f"**Recall@top-1% is omitted for {dataset}.** At the declared fraud "
                f"prevalence of {spec.fraud_rate:.2%}, reviewing the top 1% of "
                f"transactions can recover at most {ceiling:.4f} of all fraud. "
                f"Manuscript Table 5(c) reports 0.4234-0.6234 for every model on this "
                f"dataset, which is arithmetically unattainable. See "
                f"docs/MANUSCRIPT_AUDIT.md finding A-01. The column is withheld rather "
                f"than reproduced."
            )
            global_caveats.append(f"{dataset}: Recall@top-1% withheld (finding A-01).")

        subtables.append(
            RenderedTable(
                title=f"Table 5 - {dataset} ({spec.name}), prevalence {spec.fraud_rate:.2%}",
                columns=columns,
                rows=rows,
                caveats=caveats,
            )
        )

    return RenderedTable(
        title="Table 5 (a-c). Primary benchmark performance on D1-D3",
        columns=[],
        rows=[],
        caveats=[
            "All values computed from result files in results/metrics/. Nothing in this "
            "table is copied from the manuscript.",
            *global_caveats,
        ],
        subtables=subtables,
    )


def table_7_ablation(results_dir: Path) -> RenderedTable:
    """Manuscript Table 7(a-b): component ablation on D1 and D2."""
    root = results_dir / "ablation"
    if not root.exists() or not any(root.rglob("*.json")):
        raise TableBuildError(
            "no ablation results found under results/ablation/.\n"
            "Ablations A1, A3 and A6 all require a trained Stage 2 rationale model, "
            "which this repository does not ship (docs/KNOWN_LIMITATIONS.md L-25). "
            "Ablations A2, A4, A5, A7, A8 and A9 can be run once checkpoint export is "
            "enabled in scripts/train.py."
        )
    documents = [read_result(p) for p in sorted(root.rglob("*.json"))]
    rows = [
        [
            str(d["results"].get("ablation_id", "?")),
            str(d["results"].get("label", "?")),
            f"{d['results'].get('auprc', float('nan')):.4f}",
            f"{d['results'].get('delta_auprc_pp', float('nan')):+.2f}",
        ]
        for d in documents
    ]
    return RenderedTable(
        title="Table 7. Component ablation",
        columns=["ID", "Variant", "AUPRC", "Δ AUPRC vs Full (pp)"],
        rows=rows,
    )


def table_8_transfer(results_dir: Path) -> RenderedTable:
    """Manuscript Table 8(a-c): cross-dataset transfer."""
    root = results_dir / "transfer"
    if not root.exists() or not any(root.glob("*.json")):
        raise TableBuildError(
            "no transfer results found under results/transfer/.\n"
            "Produce them with, for each ordered pair:\n"
            "    python scripts/transfer.py --source D1 --target D2 "
            "--source-config configs/d1.yaml --target-config configs/d2.yaml "
            "--alignment rank_projection --seed 42\n\n"
            "Note: the manuscript specifies no feature-space alignment for transfer "
            "across these five heterogeneous schemas (audit finding A-14), so any "
            "matrix produced here documents this repository's alignment choice and is "
            "not a reproduction of Table 8."
        )
    cells = [read_result(p)["results"] for p in sorted(root.glob("*.json"))]
    datasets = sorted({c["source"] for c in cells} | {c["target"] for c in cells})
    lookup = {(c["source"], c["target"]): c for c in cells}

    rows = []
    for source in datasets:
        row = [source]
        for target in datasets:
            cell = lookup.get((source, target))
            row.append(f"{cell['auprc']:.4f}" if cell else "-")
        rows.append(row)

    strategies = sorted({c["alignment_strategy"] for c in cells})
    return RenderedTable(
        title="Table 8. Cross-dataset transfer (AUPRC)",
        columns=["Train ↓ / Test →", *datasets],
        rows=rows,
        caveats=[
            f"Alignment strategy/strategies used: {strategies}. Chosen by this "
            "repository, not by the manuscript (audit finding A-14).",
            "D5 DGraph-Fin's label is loan default, not fraud in the sense of D1-D4; "
            "transferring between them changes the task, not just the domain "
            "(audit note A-15).",
        ],
    )


def table_9_temporal(results_dir: Path) -> RenderedTable:
    """Manuscript Table 9(a-c): temporal drift and shock robustness."""
    root = results_dir / "robustness"
    documents = [
        read_result(p)
        for p in sorted(root.rglob("*.json"))
        if p.name.startswith(("month_drift", "shock"))
    ] if root.exists() else []
    if not documents:
        raise TableBuildError(
            "no temporal-robustness results under results/robustness/.\n"
            "Produce them with:\n"
            "    python scripts/robustness.py --dataset D1 --config configs/d1.yaml "
            "--seed 42 --protocol month_drift\n"
            "    python scripts/robustness.py --dataset D4 --config configs/d4.yaml "
            "--seed 42 --protocol shock --shock-timestep 43"
        )
    rows = []
    for document in documents:
        r = document["results"]
        rows.append(
            [
                str(r.get("model", "?")),
                str(r.get("dataset", "?")),
                str(r.get("reference_partition", "?")),
                f"{r.get('reference_auprc', float('nan')):.4f}",
                str(r.get("deltas_pp", r.get("delta_post_pp", "-"))),
            ]
        )
    return RenderedTable(
        title="Table 9. Temporal drift and shock robustness",
        columns=["Model", "Dataset", "Reference partition", "Reference AUPRC", "Δ (pp)"],
        rows=rows,
        caveats=[
            "The reference partition is stated explicitly for every row. Manuscript "
            "Table 9(a) labels its reference column 'Train AUPRC' while reporting the "
            "value Table 5(a) gives as the held-out test result; the two cannot both be "
            "right (audit finding A-16).",
            "The D4 shock timestep is not stated in the manuscript (ASSUMPTION L-19).",
        ],
    )


def table_10_evasion(results_dir: Path) -> RenderedTable:
    """Manuscript Table 10(a-c): budgeted evasion robustness."""
    root = results_dir / "robustness"
    documents = [
        read_result(p) for p in sorted(root.rglob("evasion_*.json"))
    ] if root.exists() else []
    if not documents:
        raise TableBuildError(
            "no evasion results under results/robustness/.\n"
            "Produce them with:\n"
            "    python scripts/robustness.py --dataset D2 --config configs/d2.yaml "
            "--seed 42 --protocol evasion --budgets 1 2 4"
        )
    rows = []
    for document in documents:
        r = document["results"]
        for budget, families in r.get("results", {}).items():
            for family, record in families.items():
                rows.append(
                    [
                        str(r.get("model", "?")),
                        str(r.get("dataset", "?")),
                        budget,
                        family,
                        str(record.get("status")),
                        "-" if record.get("evasion_rate") is None
                        else f"{record['evasion_rate']:.4f}",
                    ]
                )
    return RenderedTable(
        title="Table 10. Budgeted evasion robustness",
        columns=["Model", "Dataset", "Budget", "Edit family", "Status", "Evasion rate"],
        rows=rows,
        caveats=[
            "Edit families marked `not_applicable` are undefined for that "
            "(dataset, model) pair and produce no number. In particular the "
            "prompt-injection family requires an untrusted free-text field and a "
            "text-consuming model; neither exists for D1-D5 or for any tabular "
            "baseline (audit finding A-06).",
            "Transaction splitting is undefined for D1, whose rows are account "
            "applications rather than payments (audit finding A-07).",
        ],
    )
