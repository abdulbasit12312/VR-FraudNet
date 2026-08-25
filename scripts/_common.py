"""Shared CLI plumbing for every script in this repository.

Keeps three things uniform across scripts: argument names, the way a run is
seeded, and the way a result file is stamped with its provenance.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.config import Config, load_config, summarise_assumptions  # noqa: E402
from vrfraudnet.determinism import set_global_seed  # noqa: E402
from vrfraudnet.io_utils import RunManifest, sha256_of_file  # noqa: E402
from vrfraudnet.seeds import MANUSCRIPT_SEEDS, is_manuscript_seed_set  # noqa: E402

DATASET_CHOICES = ("D1", "D2", "D3", "D4", "D5")


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Arguments every script shares."""
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES,
                        help="manuscript dataset id")
    parser.add_argument("--config", required=True, type=Path,
                        help="path to the dataset config, e.g. configs/d1.yaml")
    parser.add_argument("--seed", type=int, default=MANUSCRIPT_SEEDS[0],
                        help=f"run seed; the manuscript uses {list(MANUSCRIPT_SEEDS)}")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override the dataset root declared in the config")
    parser.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                        help="where result files are written")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing result file instead of refusing")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def bootstrap(args: argparse.Namespace) -> tuple[Config, Path]:
    """Configure logging, load the config, seed the run, print assumptions."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("vrfraudnet")

    config = load_config(args.config)
    report = set_global_seed(args.seed)
    log.info("seed=%s torch_deterministic=%s", report.seed, report.torch_deterministic)

    if args.seed not in MANUSCRIPT_SEEDS:
        log.warning(
            "seed %s is not one of the manuscript seeds %s; results from this run "
            "are not comparable to the paper's mean +/- std",
            args.seed,
            list(MANUSCRIPT_SEEDS),
        )
    for line in summarise_assumptions(config).splitlines():
        log.info("%s", line)

    data_root = Path(args.data_root or config.require("dataset.raw_root"))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    return config, data_root


def make_manifest(
    args: argparse.Namespace,
    *,
    experiment: str,
    model: str,
    partition: str,
    notes: str = "",
) -> RunManifest:
    """Stamp a result file with everything needed to interpret it later."""
    return RunManifest(
        experiment=experiment,
        dataset=args.dataset,
        model=model,
        seed=args.seed,
        config_path=str(Path(args.config).as_posix()),
        config_sha256=sha256_of_file(args.config),
        partition=partition,
        notes=notes,
    )


def result_path(args: argparse.Namespace, *parts: str) -> Path:
    """Deterministic result-file location: results/<experiment>/<dataset>/<file>."""
    path = Path(args.results_dir).joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def warn_if_incomplete_seed_sweep(seeds: list[int]) -> None:
    """Emit a warning when a table is built from fewer than the manuscript's seeds."""
    log = logging.getLogger("vrfraudnet")
    if not is_manuscript_seed_set(seeds):
        log.warning(
            "results cover seeds %s, not the manuscript's ten seeds %s. Any "
            "mean +/- std computed from them is NOT comparable to the paper.",
            sorted(seeds),
            list(MANUSCRIPT_SEEDS),
        )
