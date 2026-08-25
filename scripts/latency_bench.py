"""Measure per-transaction latency on the common and escalation paths (S4.8.2).

Usage
-----
    python scripts/latency_bench.py --dataset D1 --config configs/d1.yaml \
        --pathway common --n-timed 10000

The escalation pathway requires a loaded Stage 2 adapter. Without one the script
exits non-zero and explains what is missing rather than timing a stub and
reporting the result as an escalation-path latency.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _common import add_common_arguments, bootstrap, make_manifest, result_path  # noqa: E402

from vrfraudnet.evaluation.latency import (  # noqa: E402
    N_TIMED,
    N_WARMUP,
    check_escalation_plausibility,
    collect_hardware_facts,
    measure_pathway,
)
from vrfraudnet.io_utils import write_result  # noqa: E402
from vrfraudnet.models.stage1_triage import TriageClassifier, TriageParams  # noqa: E402

log = logging.getLogger("latency")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--pathway", default="common", choices=("common", "escalation"))
    parser.add_argument("--n-timed", type=int, default=N_TIMED)
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--adapter", type=Path, default=None,
                        help="path to a trained Stage 2 LoRA adapter")
    args = parser.parse_args()

    config, data_root = bootstrap(args)

    if args.pathway == "escalation" and args.adapter is None:
        log.error(
            "the escalation path includes retrieval, Stage 2 generation and Stage 3 "
            "verification. Without a trained LoRA adapter (--adapter) there is nothing "
            "to time. This repository ships no weights and will not report an "
            "escalation-path latency it did not measure.\n\n"
            "Note also audit finding A-17: the manuscript's reported escalation p99 of "
            "175.23 ms is below the memory-bandwidth floor for single-stream decoding "
            "from an 8B model. Run scripts/audit_manuscript.py for the arithmetic."
        )
        return 2

    from vrfraudnet.data import load_prepared

    prepared = load_prepared(args.dataset, config, root=data_root)
    features = prepared.features.to_numpy(dtype=np.float64)
    split = prepared.split

    triage = TriageClassifier(TriageParams(), seed=args.seed)
    triage.fit(
        features[split.train], prepared.labels[split.train],
        features[split.validation], prepared.labels[split.validation],
    )

    test_features = features[split.test]
    n_rows = test_features.shape[0]

    def common_path(i: int):
        """Stage 1 triage plus routing at batch size 1 (Stage 0 precomputed)."""
        row = test_features[i % n_rows : i % n_rows + 1]
        probability = triage.predict_proba(row)
        return {"probability": float(probability[0])}

    measurement = measure_pathway(
        "common",
        common_path,
        n_timed=min(args.n_timed, 10 * n_rows),
        n_warmup=args.n_warmup,
        hardware=collect_hardware_facts(),
    )

    payload = measurement.to_dict()
    payload["protocol"] = (
        "batch size 1; data loading and offline preprocessing excluded; percentiles "
        "computed from individual transaction times (manuscript S4.8.2)"
    )
    payload["manuscript_reference"] = {
        "reported_common_p99_ms": 10.23,
        "reported_escalation_p99_ms": 175.23,
        "hardware_in_manuscript": "AMD EPYC 7543 32-core, 256 GB RAM, 1x NVIDIA A100 80GB",
        "note": "reported values apply only to the manuscript's hardware and stack",
    }
    payload["escalation_plausibility_check"] = check_escalation_plausibility(175.23, 384)

    path = result_path(args, "latency", args.dataset, f"{args.pathway}_seed{args.seed}.json")
    write_result(
        path,
        make_manifest(args, experiment="latency", model="vr_fraudnet", partition="test"),
        payload,
        overwrite=args.overwrite,
    )
    log.info(
        "%s path | p50 %.3f ms | p95 %.3f ms | p99 %.3f ms | budget %s",
        args.pathway, measurement.p50_ms, measurement.p95_ms, measurement.p99_ms,
        "OK" if measurement.within_budget else "EXCEEDED",
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
