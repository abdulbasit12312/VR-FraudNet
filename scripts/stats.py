"""Statistical comparison of VR-FraudNet against the baselines (manuscript S5.2).

Usage
-----
    python scripts/stats.py --test wilcoxon --metrics-dir results/metrics
    python scripts/stats.py --test delong --dataset D1 --config configs/d1.yaml \
        --seed-policy single_seed --seed 42

Wilcoxon
--------
Reads every ``results/metrics/<dataset>/<model>_seed<seed>.json``, forms paired
(dataset, seed) observations, and runs the one-sided paired Wilcoxon signed-rank
test against each baseline with Holm-Bonferroni correction. Both the pooled test
(as in Table 6a) and the per-dataset tests are reported, because pooling across
datasets assumes an exchangeability the data does not have (audit A-04).

DeLong
------
Requires per-transaction prediction files and an explicit ``--seed-policy``,
because the manuscript does not say which of its ten seeded prediction vectors
produced Tables 6(b)-6(d) (audit A-03).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.io_utils import read_result  # noqa: E402
from vrfraudnet.statistics.delong import delong_test  # noqa: E402
from vrfraudnet.statistics.holm import holm_bonferroni, paired_bootstrap_ci  # noqa: E402
from vrfraudnet.statistics.wilcoxon import pooled_and_per_dataset  # noqa: E402

log = logging.getLogger("stats")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", required=True, choices=("wilcoxon", "delong"))
    parser.add_argument("--metrics-dir", type=Path, default=REPO_ROOT / "results" / "metrics")
    parser.add_argument("--predictions-dir", type=Path,
                        default=REPO_ROOT / "results" / "predictions")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "statistics")
    parser.add_argument("--reference-model", default="vr_fraudnet")
    parser.add_argument("--metric", default="auprc")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed-policy", choices=("single_seed", "seed_mean_scores"),
                        default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.test == "wilcoxon":
        return _wilcoxon(args)
    return _delong(args)


def _load_metric_table(metrics_dir: Path, metric: str) -> dict[str, dict[str, dict[int, float]]]:
    """Return ``{dataset: {model: {seed: value}}}`` from stored metric files."""
    table: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(metrics_dir.rglob("*.json")):
        document = read_result(path)
        manifest, results = document["manifest"], document["results"]
        value = results.get(metric)
        if value is None:
            continue
        table[manifest["dataset"]][manifest["model"]][int(manifest["seed"])] = float(value)
    return {k: dict(v) for k, v in table.items()}


def _wilcoxon(args) -> int:
    table = _load_metric_table(args.metrics_dir, args.metric)
    if not table:
        log.error(
            "no metric files found under %s. Produce them with scripts/train.py then "
            "scripts/evaluate.py; this script never invents results.",
            args.metrics_dir,
        )
        return 1

    models = sorted({m for per_model in table.values() for m in per_model})
    if args.reference_model not in models:
        log.error("reference model %r not present in %s", args.reference_model, sorted(models))
        return 1
    baselines = [m for m in models if m != args.reference_model]

    per_baseline: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for baseline in baselines:
        model_by_dataset: dict[str, list[float]] = {}
        baseline_by_dataset: dict[str, list[float]] = {}
        for dataset, per_model in table.items():
            if args.reference_model not in per_model or baseline not in per_model:
                continue
            shared = sorted(set(per_model[args.reference_model]) & set(per_model[baseline]))
            if len(shared) < 2:
                continue
            model_by_dataset[dataset] = [per_model[args.reference_model][s] for s in shared]
            baseline_by_dataset[dataset] = [per_model[baseline][s] for s in shared]
        if not model_by_dataset:
            log.warning("no paired observations for baseline %s", baseline)
            continue
        try:
            results = pooled_and_per_dataset(
                model_by_dataset, baseline_by_dataset, baseline_name=baseline
            )
        except ValueError as exc:
            log.warning("skipping %s: %s", baseline, exc)
            continue
        flat_model = [v for values in model_by_dataset.values() for v in values]
        flat_base = [v for values in baseline_by_dataset.values() for v in values]
        point, low, high = paired_bootstrap_ci(flat_model, flat_base)
        per_baseline[baseline] = {
            "scopes": {k: v.to_dict() for k, v in results.items()},
            "paired_mean_delta_pp": point,
            "paired_bootstrap_ci_pp": [low, high],
            "n_datasets": len(model_by_dataset),
        }
        raw_p[baseline] = results["pooled"].p_value_one_sided

    adjusted = holm_bonferroni(raw_p)
    for baseline, value in adjusted.items():
        per_baseline[baseline]["holm_adjusted_p"] = value

    out = {
        "test": "one-sided paired Wilcoxon signed-rank",
        "metric": args.metric,
        "reference_model": args.reference_model,
        "multiple_comparison_correction": "Holm-Bonferroni",
        "datasets": sorted(table),
        "seeds_present": sorted(
            {s for per_model in table.values() for per_seed in per_model.values() for s in per_seed}
        ),
        "audit_note_A04": (
            "The 'pooled' scope reproduces manuscript Table 6(a), which pools paired "
            "differences across D1-D3. Those differences are not exchangeable across "
            "datasets; the per-dataset scopes are the stricter reading."
        ),
        "results": per_baseline,
    }
    path = args.out / f"wilcoxon_{args.metric}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    for baseline, record in sorted(per_baseline.items()):
        pooled = record["scopes"]["pooled"]
        log.info(
            "%-20s mean delta %+7.2f pp | V=%-5.0f/%d | p_raw=%.3g | p_holm=%.3g",
            baseline, pooled["mean_delta_pp"], pooled["wilcoxon_V"],
            pooled["wilcoxon_V_max"], pooled["p_raw_one_sided"],
            record.get("holm_adjusted_p", float("nan")),
        )
    return 0


def _delong(args) -> int:
    if args.dataset is None or args.seed_policy is None:
        log.error(
            "DeLong requires --dataset and --seed-policy. The manuscript does not state "
            "which of its ten seeded prediction vectors produced Tables 6(b)-6(d) "
            "(audit finding A-03), so the choice must be made explicitly."
        )
        return 1

    directory = args.predictions_dir / args.dataset
    if not directory.exists():
        log.error("no predictions under %s", directory)
        return 1

    by_model: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        document = read_result(path)
        manifest, results = document["manifest"], document["results"]
        by_model[manifest["model"]][int(manifest["seed"])] = (
            np.asarray(results["test"]["scores"], dtype=float),
            np.asarray(results["test"]["labels"], dtype=int),
        )

    if args.reference_model not in by_model:
        log.error("reference model %r has no predictions for %s", args.reference_model, args.dataset)
        return 1

    def resolve(model: str) -> tuple[np.ndarray, np.ndarray]:
        per_seed = by_model[model]
        if args.seed_policy == "single_seed":
            if args.seed not in per_seed:
                raise KeyError(f"{model} has no predictions for seed {args.seed}")
            return per_seed[args.seed]
        scores = np.mean([v[0] for v in per_seed.values()], axis=0)
        labels = next(iter(per_seed.values()))[1]
        return scores, labels

    model_scores, labels = resolve(args.reference_model)
    records: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for model in sorted(by_model):
        if model == args.reference_model:
            continue
        try:
            baseline_scores, baseline_labels = resolve(model)
        except KeyError as exc:
            log.warning("%s", exc)
            continue
        if not np.array_equal(labels, baseline_labels):
            log.error(
                "%s and %s were evaluated on different label vectors; DeLong requires "
                "the same sample", args.reference_model, model,
            )
            return 1
        result = delong_test(
            model_scores, baseline_scores, labels,
            dataset=args.dataset, baseline=model,
            seed_policy=args.seed_policy, seed=args.seed,
        )
        records[model] = result.to_dict()
        raw_p[model] = result.p_value_two_sided

    for model, value in holm_bonferroni(raw_p).items():
        records[model]["p_holm_adjusted"] = value

    out = {
        "test": "DeLong two-sided z-test on correlated ROC AUCs",
        "dataset": args.dataset,
        "reference_model": args.reference_model,
        "seed_policy": args.seed_policy,
        "seed": args.seed,
        "audit_note_A03": (
            "The manuscript does not state which prediction vectors produced its DeLong "
            "statistics. The policy used here is recorded above and must be quoted "
            "alongside any comparison with Tables 6(b)-6(d)."
        ),
        "results": records,
    }
    path = args.out / f"delong_{args.dataset}_{args.seed_policy}.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
