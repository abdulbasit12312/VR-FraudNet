"""Prepare one benchmark: load raw files, apply Table 2 preprocessing, build splits.

Usage
-----
    python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml

Writes a preparation report to ``results/prepared/<dataset>/report.json``
containing the split sizes, the executed leakage controls, and a comparison of
the realised feature count against the count manuscript Table 1/Table 2
declares. It does **not** write the preprocessed data itself: derived data
belongs outside version control (see ``.gitignore``).
"""

from __future__ import annotations

import argparse
import logging

from _common import add_common_arguments, bootstrap, make_manifest, result_path

from vrfraudnet.data import load_prepared
from vrfraudnet.data.registry import get_spec
from vrfraudnet.io_utils import write_result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--check-only", action="store_true",
                        help="verify the raw files are present and readable, then stop")
    args = parser.parse_args()

    config, data_root = bootstrap(args)
    log = logging.getLogger("prepare_data")
    spec = get_spec(args.dataset)

    log.info("preparing %s (%s) from %s", spec.dataset_id, spec.name, data_root)
    prepared = load_prepared(args.dataset, config, root=data_root)

    realised_features = int(prepared.features.shape[1])
    declared = spec.model_input_count
    if declared is not None and realised_features != declared:
        log.warning(
            "feature-count mismatch: realised %d, manuscript Table 2 declares %d. "
            "This repository reports the realised count rather than padding to the "
            "declared one; see docs/MANUSCRIPT_AUDIT.md.",
            realised_features,
            declared,
        )

    prevalence = prepared.prevalence
    if abs(prevalence - spec.fraud_rate) > 0.002:
        log.warning(
            "prevalence mismatch: realised %.4f%%, Table 1 declares %.4f%%",
            prevalence * 100,
            spec.fraud_rate * 100,
        )

    payload = {
        "dataset": spec.dataset_id,
        "name": spec.name,
        "n_rows": int(prepared.features.shape[0]),
        "n_features_realised": realised_features,
        "n_features_declared_table2": declared,
        "prevalence_realised": prevalence,
        "prevalence_declared_table1": spec.fraud_rate,
        "recall_at_top_1pct_ceiling": spec.recall_at_top_1pct_ceiling,
        "split_sizes": prepared.split.sizes,
        "split_description": prepared.split.description,
        "leakage_controls": prepared.leakage.to_dict(),
        "graph": None
        if prepared.graph is None
        else {
            "num_nodes": prepared.graph.num_nodes,
            "num_edges": int(prepared.graph.edge_index.shape[1]),
        },
        "metadata": prepared.metadata,
    }

    if args.check_only:
        log.info("check-only: %s", payload["split_sizes"])
        return 0

    path = result_path(args, "prepared", args.dataset, "report.json")
    write_result(
        path,
        make_manifest(args, experiment="prepare_data", model="none", partition="all"),
        payload,
        overwrite=args.overwrite,
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
