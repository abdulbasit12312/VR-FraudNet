"""Verify a reproducibility archive produced by build_reproducibility_bundle.py.

Usage
-----
    python scripts/verify_reproducibility_bundle.py dist/VR-FraudNet-reproducibility-1.0.0.tar.gz
    python scripts/verify_reproducibility_bundle.py <archive> --json

Verifies, without extracting anything to disk:

* every required file is present (README, LICENSE, CITATION, manifest,
  SHA256SUMS, seed file, Stage 2 config, schema, verifier rules, retrieval and
  edit configuration, grammar and Stage 2 source, reviewer documentation, ...);
* every hash in REPRODUCIBILITY_MANIFEST.json matches the archived bytes, every
  archived file is listed in the manifest, and every ``components`` hash matches;
* SHA256SUMS matches;
* configs/manuscript_seeds.yaml equals the manifest seeds and, when run from a
  checkout, the code constant ``vrfraudnet.seeds.MANUSCRIPT_SEEDS``;
* the JSON schema and every YAML / JSON file parse;
* no dataset, weight, cache or credential file is present;
* no secret pattern appears in any text file.

Exit code 0 when the archive verifies, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.reproducibility import verify_archive  # noqa: E402

try:
    from vrfraudnet.seeds import MANUSCRIPT_SEEDS
except ImportError:  # pragma: no cover - standalone use outside a checkout
    MANUSCRIPT_SEEDS = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    report = verify_archive(
        args.archive, expected_seeds=list(MANUSCRIPT_SEEDS) if MANUSCRIPT_SEEDS else None
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"archive: {report.get('archive')}")
        print(f"sha256:  {report.get('archive_sha256')}")
        print(f"files:   {report.get('file_count')}  version: {report.get('release_version')}")
        git = report.get("git") or {}
        print(f"commit:  {git.get('commit')}  tag: {git.get('tag')}")
        if report["ok"]:
            print("RESULT:  OK - every check passed")
        else:
            print("RESULT:  FAILED")
            for problem in report["problems"]:
                print(f"  - {problem}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
