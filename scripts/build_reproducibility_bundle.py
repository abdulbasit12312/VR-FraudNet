"""Build the versioned reproducibility archive, manifest and checksum listing.

Usage
-----
    python scripts/build_reproducibility_bundle.py --version 1.0.0 --tag v1.0.0-manuscript-revision
    python scripts/build_reproducibility_bundle.py --write-manifest     # refresh the committed manifest
    python scripts/build_reproducibility_bundle.py --check-manifest     # CI: committed manifest is current
    python scripts/build_reproducibility_bundle.py --dry-run            # list what would be bundled

Outputs (default ``dist/``):
    VR-FraudNet-reproducibility-<version>.tar.gz
    REPRODUCIBILITY_MANIFEST.json      (also inside the archive)
    SHA256SUMS                         (also inside the archive)

Every hash is computed from the bytes on disk. The builder refuses to run when
a selected file is a dataset, a weight file, a cache, or contains a secret or
local-path pattern. The archive is deterministic (see docs/RELEASE.md).

Exit codes: 0 ok, 1 forbidden content or stale manifest, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.reproducibility import (  # noqa: E402
    ARCHIVE_PREFIX,
    CHECKSUMS_FILE,
    MANIFEST_FILE,
    build_manifest,
    checksums_text,
    collect_bundle_files,
    compare_manifest_files,
    forbidden_content_problems,
    manifest_json,
    sha256_bytes,
    write_archive,
)


def _project_version() -> str:
    import re

    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if not match:
        raise SystemExit("could not read version from pyproject.toml")
    return match.group(1)


def _commit_time() -> int:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%ct"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=15, check=False)
        return int(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0
    except (OSError, subprocess.SubprocessError, ValueError):  # pragma: no cover
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default=None, help="release version (default: pyproject.toml)")
    parser.add_argument("--tag", default=None, help="git tag to record in the manifest")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--write-manifest", action="store_true",
                        help=f"write {MANIFEST_FILE} at the repository root and stop")
    parser.add_argument("--check-manifest", action="store_true",
                        help=f"fail if the committed {MANIFEST_FILE} is stale")
    parser.add_argument("--dry-run", action="store_true", help="list the files and stop")
    args = parser.parse_args(argv)

    version = args.version or _project_version()
    files = collect_bundle_files(REPO_ROOT)
    problems = forbidden_content_problems(files)
    if problems:
        print("refusing to bundle; forbidden content found:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    if args.dry_run:
        for f in files:
            print(f.relative)
        print(f"\n{len(files)} files would be bundled as {ARCHIVE_PREFIX}-{version}.tar.gz")
        return 0

    committed_path = REPO_ROOT / MANIFEST_FILE
    committed = json.loads(committed_path.read_text(encoding="utf-8")) if committed_path.exists() else None
    fresh = build_manifest(
        REPO_ROOT, version=version, tag=args.tag, files=files,
        created_utc=(committed or {}).get("created_utc") if args.check_manifest else None,
    )

    if args.check_manifest:
        if committed is None:
            print(f"{MANIFEST_FILE} is not committed; run --write-manifest", file=sys.stderr)
            return 1
        diffs = compare_manifest_files(committed, fresh)
        if diffs:
            print(f"{MANIFEST_FILE} is stale; run --write-manifest and commit it:", file=sys.stderr)
            for d in diffs:
                print(f"  {d}", file=sys.stderr)
            return 1
        print(f"{MANIFEST_FILE} is current ({fresh['file_count']} files, version {version}).")
        return 0

    if args.write_manifest:
        committed_path.write_text(manifest_json(fresh), encoding="utf-8")
        print(f"wrote {committed_path.relative_to(REPO_ROOT)} "
              f"({fresh['file_count']} files, commit {fresh['git']['commit']})")
        return 0

    # The manifest is bundled too; recompute the file set after (re)writing it so
    # the archive's copy hashes the manifest that describes the archive.
    mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0") or 0) or _commit_time()
    output = args.output_dir / f"{ARCHIVE_PREFIX}-{version}.tar.gz"
    result = write_archive(files, manifest=fresh, output=output, mtime=mtime)
    (args.output_dir / MANIFEST_FILE).write_text(manifest_json(fresh), encoding="utf-8")
    listing = checksums_text(
        {**fresh["files"], MANIFEST_FILE: sha256_bytes(manifest_json(fresh).encode("utf-8"))},
        result["prefix"],
    )
    listing += f"{result['sha256']}  {output.name}\n"
    (args.output_dir / CHECKSUMS_FILE).write_text(listing, encoding="utf-8")
    print(json.dumps({
        "archive": output.name,
        "archive_sha256": result["sha256"],
        "file_count": fresh["file_count"],
        "release_version": version,
        "git_commit": fresh["git"]["commit"],
        "git_tag": fresh["git"]["tag"],
        "working_tree_dirty": fresh["git"]["working_tree_dirty"],
        "mtime_used": mtime,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
