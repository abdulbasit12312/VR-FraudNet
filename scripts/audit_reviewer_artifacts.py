"""Audit the reviewer-facing artefact map against the repository.

Usage
-----
    python scripts/audit_reviewer_artifacts.py
    python scripts/audit_reviewer_artifacts.py --json

Checks that everything the reviewer was told exists, exists:

1. every backticked repository path in docs/REVIEWER_REPRODUCIBILITY.md,
   docs/ARTIFACT_AVAILABILITY.md, docs/STAGE2_LORA.md, docs/RETRIEVAL_SETUP.md,
   docs/EDIT_GENERATORS.md, docs/DECODING_GRAMMAR.md and artifacts/lora/README.md
   resolves to a file or directory in the checkout;
2. every relative Markdown link in README.md and those documents resolves;
3. README.md links to each reviewer-requested material;
4. configs/manuscript_seeds.yaml equals vrfraudnet.seeds.MANUSCRIPT_SEEDS;
5. every ``COMPONENT_PATHS`` entry of the manifest module exists;
6. the committed REPRODUCIBILITY_MANIFEST.json, when present, is current.

Exit 0 when every check passes, 1 otherwise. No data, no model, no network.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.reproducibility import (  # noqa: E402
    COMPONENT_PATHS,
    MANIFEST_FILE,
    build_manifest,
    compare_manifest_files,
)
from vrfraudnet.seeds import MANUSCRIPT_SEEDS  # noqa: E402

REVIEWER_DOCS = (
    "docs/REVIEWER_REPRODUCIBILITY.md",
    "docs/ARTIFACT_AVAILABILITY.md",
    "docs/STAGE2_LORA.md",
    "docs/RETRIEVAL_SETUP.md",
    "docs/EDIT_GENERATORS.md",
    "docs/DECODING_GRAMMAR.md",
    "docs/RELEASE.md",
    "artifacts/lora/README.md",
)

#: Materials the reviewer named, each with the README link target that answers it.
README_REQUIRED_LINKS = {
    "verifier rules": "verifier/rules/claim_primitives.yaml",
    "JSON schema": "schemas/rationale_schema.json",
    "decoding grammar": "docs/DECODING_GRAMMAR.md",
    "LoRA adapters": "docs/STAGE2_LORA.md",
    "LoRA adapter status": "artifacts/lora/README.md",
    "LoRA configuration": "configs/stage2_lora.yaml",
    "retrieval setup": "docs/RETRIEVAL_SETUP.md",
    "edit generators": "docs/EDIT_GENERATORS.md",
    "configuration files": "configs/",
    "seeds": "configs/manuscript_seeds.yaml",
    "versioned archive": "docs/RELEASE.md",
    "artifact availability": "docs/ARTIFACT_AVAILABILITY.md",
    "reviewer map": "docs/REVIEWER_REPRODUCIBILITY.md",
    "known limitations": "docs/KNOWN_LIMITATIONS.md",
}

# A backticked token that looks like a repository path: contains a directory
# separator and a known suffix (or a trailing slash), no spaces, no wildcard.
# Bare file names such as `edits.py` are prose references, not path claims.
PATH_TOKEN = re.compile(
    r"`([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-/]*(?:\.(?:py|md|yaml|yml|json|jsonl|txt|toml|cff|sh)|/))`"
)
SKIP_TOKENS = {"dist/", "retrieval/index/", "checkpoints/", "datasets/raw/", "results/",
               "artifacts/lora/manuscript", "artifacts/lora/does-not-exist", "/tmp/idx"}
LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)]*)?\)")


def _path_tokens(text: str) -> set[str]:
    out = set()
    for match in PATH_TOKEN.finditer(text):
        token = match.group(1)
        if "<" in token or ">" in token or token.startswith("http"):
            continue
        # Strip a "::symbol" suffix and a trailing slash.
        token = token.split("::")[0]
        out.add(token)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    problems: list[str] = []
    checked: dict[str, int] = {}

    # 1. backticked paths in the reviewer documents
    for doc in REVIEWER_DOCS:
        path = REPO_ROOT / doc
        if not path.exists():
            problems.append(f"missing reviewer document: {doc}")
            continue
        text = path.read_text(encoding="utf-8")
        tokens = _path_tokens(text)
        checked[doc] = len(tokens)
        for token in sorted(tokens):
            if token in SKIP_TOKENS or any(token.startswith(s) for s in SKIP_TOKENS):
                continue
            candidate = REPO_ROOT / token.rstrip("/")
            if not candidate.exists():
                problems.append(f"{doc}: path `{token}` does not exist")

    # 2. relative markdown links
    for doc in ("README.md", *REVIEWER_DOCS):
        path = REPO_ROOT / doc
        if not path.exists():
            continue
        for match in LINK.finditer(path.read_text(encoding="utf-8")):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                problems.append(f"{doc}: link target {target} does not exist")

    # 3. README links to every reviewer-requested material
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for label, target in README_REQUIRED_LINKS.items():
        if f"]({target}" not in readme:
            problems.append(f"README.md does not link to {target} ({label})")

    # 4. seeds
    seed_file = yaml.safe_load((REPO_ROOT / "configs" / "manuscript_seeds.yaml").read_text(encoding="utf-8"))
    if [int(s) for s in seed_file["seeds"]] != list(MANUSCRIPT_SEEDS):
        problems.append("configs/manuscript_seeds.yaml differs from vrfraudnet.seeds.MANUSCRIPT_SEEDS")

    # 5. manifest components
    for key, paths in COMPONENT_PATHS.items():
        for rel in paths:
            if not (REPO_ROOT / rel).exists():
                problems.append(f"manifest component {key}: {rel} does not exist")

    # 6. committed manifest currency
    committed_path = REPO_ROOT / MANIFEST_FILE
    if committed_path.exists():
        committed = json.loads(committed_path.read_text(encoding="utf-8"))
        fresh = build_manifest(REPO_ROOT, version=committed.get("release_version", "0"),
                               created_utc=committed.get("created_utc"))
        for diff in compare_manifest_files(committed, fresh):
            problems.append(f"{MANIFEST_FILE} stale: {diff}")
    else:
        problems.append(f"{MANIFEST_FILE} is not present at the repository root")

    report = {"ok": not problems, "problems": problems, "paths_checked": checked,
              "readme_links_required": len(README_REQUIRED_LINKS)}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for doc, n in checked.items():
            print(f"{doc}: {n} repository paths referenced")
        if problems:
            print("\nFAILED:")
            for problem in problems:
                print(f"  - {problem}")
        else:
            print("\nOK: every referenced path exists, every link resolves, README covers every "
                  "reviewer-requested material, seeds agree, manifest is current.")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
