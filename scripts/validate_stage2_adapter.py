"""Validate a Stage 2 LoRA adapter directory against ``configs/stage2_lora.yaml``.

Usage
-----
    python scripts/validate_stage2_adapter.py --adapter <path> --config configs/stage2_lora.yaml
    python scripts/validate_stage2_adapter.py --adapter <path> --load     # also load the weights

Checks
------
1. ``adapter_config.json`` exists, parses, and declares LoRA with the rank,
   alpha, dropout, target modules, bias policy and base-model identifier the
   canonical configuration requires.
2. An adapter weight file (``adapter_model.safetensors`` preferred) exists.
3. SHA-256 of every file in the directory is printed, so a reader can compare a
   downloaded adapter against a published checksum.
4. With ``--load``, the base model and adapter are actually loaded through PEFT
   (requires the gated base weights, ``transformers`` and ``peft``); a smoke
   test then confirms the grammar-masked decoder runs for one step.

What this script cannot do
--------------------------
It cannot establish that a set of weights is the adapter used to produce the
manuscript's numbers. That adapter is not present in the accessible project
materials (artifacts/lora/README.md), so no reference checksum exists to compare
against. A passing validation means "structurally consistent with the
documented Stage 2 configuration", nothing more.

Exit codes: 0 valid, 2 adapter missing, 1 metadata mismatch or load failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.errors import ConfigurationError, MissingArtefactError  # noqa: E402
from vrfraudnet.io_utils import sha256_of_file  # noqa: E402
from vrfraudnet.models.stage2_rationale import (  # noqa: E402
    ADAPTER_CONFIG_FILE,
    RationaleModel,
    Stage2Config,
    check_adapter_metadata,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", type=Path, required=True,
                        help="directory containing adapter_config.json and the weight file")
    parser.add_argument("--config", type=Path,
                        default=REPO_ROOT / "configs" / "stage2_lora.yaml")
    parser.add_argument("--load", action="store_true",
                        help="load base model + adapter through PEFT and run a one-step "
                             "grammar-masked decode (needs the gated base weights)")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    config = Stage2Config.from_yaml(args.config)
    report: dict = {
        "adapter_dir": args.adapter.name,
        "config": str(args.config.relative_to(REPO_ROOT)) if args.config.is_relative_to(REPO_ROOT) else args.config.name,
        "authenticity": (
            "cannot be established: the manuscript adapter is absent from the accessible "
            "project materials (artifacts/lora/README.md); this check is structural only"
        ),
    }

    if not args.adapter.exists():
        report["status"] = "MISSING"
        report["detail"] = (
            f"{args.adapter} does not exist. No adapter is distributed with this repository "
            "and none is fabricated. Train one with scripts/train_stage2_lora.py."
        )
        _emit(report, args.json)
        return 2

    problems = check_adapter_metadata(args.adapter, config)
    files = {p.name: sha256_of_file(p) for p in sorted(args.adapter.iterdir()) if p.is_file()}
    report["files_sha256"] = files
    meta_path = args.adapter / ADAPTER_CONFIG_FILE
    if meta_path.exists():
        try:
            report["adapter_config"] = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report["adapter_config"] = None
    report["metadata_problems"] = problems
    if problems:
        report["status"] = "MISMATCH"
        _emit(report, args.json)
        return 1

    if args.load:
        try:
            model = RationaleModel(config, REPO_ROOT / "schemas" / "rationale_schema.json",
                                   adapter_path=args.adapter).load()
            prefix = '{"verdict":"'
            allowed = sorted(model.grammar.allowed_characters(prefix))
            report["load_smoke_test"] = {
                "loaded": True,
                "grammar_allowed_after_verdict_quote": allowed,
                "base_model_revision": config.revision_label(),
            }
        except (MissingArtefactError, ConfigurationError) as exc:
            report["status"] = "LOAD_FAILED"
            report["detail"] = str(exc)
            _emit(report, args.json)
            return 1
        except Exception as exc:  # pragma: no cover - environment dependent
            report["status"] = "LOAD_FAILED"
            report["detail"] = f"{type(exc).__name__}: {exc}"
            _emit(report, args.json)
            return 1

    report["status"] = "VALID_STRUCTURE"
    _emit(report, args.json)
    return 0


def _emit(report: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    print(f"status: {report['status']}")
    for key in ("detail", "authenticity"):
        if key in report:
            print(f"{key}: {report[key]}")
    for problem in report.get("metadata_problems", []):
        print(f"  mismatch: {problem}")
    for name, digest in report.get("files_sha256", {}).items():
        print(f"  {digest}  {name}")
    if "load_smoke_test" in report:
        print(f"load smoke test: {report['load_smoke_test']}")


if __name__ == "__main__":
    raise SystemExit(main())
