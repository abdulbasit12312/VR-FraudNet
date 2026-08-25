"""Compute the manuscript metric set from stored prediction files.

Usage
-----
    python scripts/evaluate.py --dataset D1 --config configs/d1.yaml --seed 42
    python scripts/evaluate.py --dataset D1 --config configs/d1.yaml \
        --model tabtransformer --seed 42

Reads ``results/predictions/<dataset>/<model>_seed<seed>.json`` and writes
``results/metrics/<dataset>/<model>_seed<seed>.json``. It never trains anything,
so a metric can only exist if a run produced the predictions it summarises.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from _common import add_common_arguments, bootstrap, make_manifest, result_path

from vrfraudnet.data.registry import get_spec
from vrfraudnet.evaluation.metrics import compute_all, recall_at_top_k_ceiling
from vrfraudnet.io_utils import read_result, require_file, write_result

log = logging.getLogger("evaluate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", default="vr_fraudnet")
    args = parser.parse_args()

    bootstrap(args)
    spec = get_spec(args.dataset)

    predictions_path = require_file(
        result_path(args, "predictions", args.dataset, f"{args.model}_seed{args.seed}.json"),
        produced_by=(
            f"python scripts/train.py --dataset {args.dataset} "
            f"--config {args.config} --seed {args.seed} --model {args.model}"
        ),
    )
    document = read_result(predictions_path)
    payload = document["results"]

    scores = np.asarray(payload["test"]["scores"], dtype=float)
    labels = np.asarray(payload["test"]["labels"], dtype=int)
    threshold = float(payload["operating_threshold"])

    metrics = compute_all(
        labels,
        scores,
        threshold=threshold,
        dataset_id=args.dataset,
        declared_prevalence=spec.fraud_rate,
    )
    for warning in metrics.warnings:
        log.warning("%s", warning)

    ceiling = recall_at_top_k_ceiling(spec.fraud_rate)
    out = metrics.to_dict()
    out["recall_at_top_1pct_ceiling_declared_prevalence"] = ceiling
    out["model"] = args.model
    out["dataset"] = args.dataset

    if ceiling < 1.0:
        out["audit_note_A01"] = (
            f"Recall@top-1% cannot exceed {ceiling:.4f} at the declared prevalence "
            f"{spec.fraud_rate:.4%}. Manuscript values above that bound are "
            f"unattainable; see docs/MANUSCRIPT_AUDIT.md finding A-01."
        )

    path = result_path(args, "metrics", args.dataset, f"{args.model}_seed{args.seed}.json")
    write_result(
        path,
        make_manifest(args, experiment="evaluate", model=args.model, partition="test"),
        out,
        overwrite=args.overwrite,
    )
    log.info(
        "%s %s seed=%s | AUPRC=%.4f ROC-AUC=%.4f F1=%.4f MCC=%.4f",
        args.dataset, args.model, args.seed,
        metrics.auprc, metrics.roc_auc, metrics.f1, metrics.mcc,
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
