"""Cross-dataset transfer evaluation (manuscript Section 5.4, Table 8).

Usage
-----
    python scripts/transfer.py --source D1 --target D3 \
        --source-config configs/d1.yaml --target-config configs/d3.yaml \
        --alignment rank_projection --seed 42
    python scripts/transfer.py ... --alignment head_refit --few-shot-fraction 0.01

The ``--alignment`` flag is mandatory. The five benchmarks share no feature
space, and the manuscript describes no mechanism for applying a model trained on
one to another, so "zero-shot transfer" has no defined meaning until an
alignment is named (audit finding A-14). Whichever alignment is chosen is
recorded in the result file next to every number it produced.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from vrfraudnet.config import load_config  # noqa: E402
from vrfraudnet.data import load_prepared  # noqa: E402
from vrfraudnet.determinism import rng_for, set_global_seed  # noqa: E402
from vrfraudnet.evaluation.metrics import auprc  # noqa: E402
from vrfraudnet.evaluation.transfer import (  # noqa: E402
    AlignmentStrategy,
    TransferCell,
    few_shot_subset,
    mutual_information_order,
    rank_project,
    require_alignment_strategy,
)
from vrfraudnet.io_utils import RunManifest, sha256_of_file, write_result  # noqa: E402
from vrfraudnet.models.stage1_triage import TriageClassifier, TriageParams  # noqa: E402

log = logging.getLogger("transfer")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--target-config", required=True, type=Path)
    parser.add_argument("--alignment", required=True,
                        choices=[s.value for s in AlignmentStrategy])
    parser.add_argument("--projection-width", type=int, default=32)
    parser.add_argument("--few-shot-fraction", type=float, default=0.0,
                        help="fraction of TARGET labels used for adaptation; "
                             "manuscript Table 8(b) uses 0.01")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")
    set_global_seed(args.seed)
    strategy = require_alignment_strategy(AlignmentStrategy(args.alignment))

    source_config = load_config(args.source_config)
    target_config = load_config(args.target_config)
    source_root = Path(args.data_root or source_config.require("dataset.raw_root"))
    target_root = Path(args.data_root or target_config.require("dataset.raw_root"))

    source = load_prepared(args.source, source_config, root=source_root)
    target = load_prepared(args.target, target_config, root=target_root)

    if strategy is AlignmentStrategy.COLUMN_NAME:
        shared = [c for c in source.features.columns if c in target.features.columns]
        if not shared:
            log.error(
                "column-name alignment is impossible: %s and %s share no column names. "
                "Use rank_projection or head_refit, and note in any write-up that the "
                "alignment was invented by this repository, not by the manuscript.",
                args.source, args.target,
            )
            return 1
        x_src = source.features[shared].to_numpy(dtype=float)
        x_tgt = target.features[shared].to_numpy(dtype=float)
    else:
        order = mutual_information_order(
            source.features.iloc[source.split.train].to_numpy(dtype=float),
            source.labels[source.split.train],
            seed=args.seed,
        )
        x_src = rank_project(source.features.to_numpy(dtype=float),
                             width=args.projection_width, order=order)
        x_tgt = rank_project(target.features.to_numpy(dtype=float),
                             width=args.projection_width)

    model = TriageClassifier(TriageParams(), seed=args.seed)
    model.fit(
        x_src[source.split.train], source.labels[source.split.train],
        x_src[source.split.validation], source.labels[source.split.validation],
    )

    n_target_labels = 0
    if strategy is AlignmentStrategy.HEAD_REFIT or args.few_shot_fraction > 0:
        if args.few_shot_fraction <= 0:
            log.error("head_refit requires --few-shot-fraction > 0")
            return 1
        rng = rng_for(args.seed, "few_shot", args.source, args.target)
        subset = few_shot_subset(
            target.labels[target.split.train], fraction=args.few_shot_fraction, rng=rng
        )
        absolute = target.split.train[subset]
        n_target_labels = int(absolute.size)
        log.info("adapting on %d target labels (%.2f%%)", n_target_labels,
                 100 * args.few_shot_fraction)
        model.fit(
            x_tgt[absolute], target.labels[absolute],
            x_tgt[target.split.validation], target.labels[target.split.validation],
        )

    scores = model.predict_proba(x_tgt[target.split.test])
    value = auprc(target.labels[target.split.test], scores)

    cell = TransferCell(
        source=args.source,
        target=args.target,
        auprc=value,
        strategy=strategy.value,
        n_target_labels_used=n_target_labels,
        is_diagonal=args.source == args.target,
        note=(
            "The manuscript specifies no feature-space alignment for cross-dataset "
            "transfer (audit finding A-14). The strategy recorded here was chosen by "
            "this repository and must be quoted alongside the number."
        ),
    )

    path = Path(args.results_dir) / "transfer" / f"{args.source}_to_{args.target}_{strategy.value}.json"
    write_result(
        path,
        RunManifest(
            experiment="transfer",
            dataset=f"{args.source}->{args.target}",
            model="vr_fraudnet_stage1",
            seed=args.seed,
            config_path=str(args.source_config.as_posix()),
            config_sha256=sha256_of_file(args.source_config),
            partition="target test",
        ),
        cell.to_dict(),
        overwrite=args.overwrite,
    )
    log.info("%s -> %s | AUPRC %.4f | alignment %s", args.source, args.target, value, strategy.value)
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
