"""Reproducibility bundle: file selection, manifest, checksums, deterministic archive.

This module is the single definition of

* which repository files belong in the reviewer-facing reproducibility
  archive (:data:`BUNDLE_INCLUDE`), and which must never be in it
  (:data:`FORBIDDEN_SUFFIXES`, :data:`EXCLUDED_DIRS`, :data:`SECRET_PATTERNS`);
* the structure of ``REPRODUCIBILITY_MANIFEST.json`` (:func:`build_manifest`);
* the deterministic ``.tar.gz`` layout (:func:`write_archive`);
* the verification performed on an archive (:func:`verify_archive`).

``scripts/build_reproducibility_bundle.py`` and
``scripts/verify_reproducibility_bundle.py`` are thin wrappers so that every
rule here is unit-testable without building an archive.

Every hash written by this module is computed from the bytes on disk at build
time. There are no placeholders.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import platform
import re
import subprocess
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

MANIFEST_SCHEMA_VERSION = "vrfraudnet-reproducibility-manifest/1"
MANIFEST_FILE = "REPRODUCIBILITY_MANIFEST.json"
CHECKSUMS_FILE = "SHA256SUMS"
PROJECT = "VR-FraudNet"
ARCHIVE_PREFIX = "VR-FraudNet-reproducibility"

#: Directories (relative to the repository root) whose text files are bundled
#: in full, and individual files that are bundled.
BUNDLE_INCLUDE: tuple[str, ...] = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "Makefile",
    "pyproject.toml",
    "requirements.txt",
    "requirements-optional.txt",
    "environment.yml",
    "reproduce_all.sh",
    ".gitignore",
    "configs",
    "schemas",
    "verifier",
    "retrieval/config",
    "adversarial",
    "evaluation",
    "statistics",
    "src",
    "scripts",
    "tests",
    "examples",
    "docs",
    "artifacts",
    "datasets/README.md",
    "datasets/checksums/README.md",
    "results/README.md",
    ".github/workflows",
)

#: Directory names that are never descended into, wherever they appear.
EXCLUDED_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv",
    "dist", "build", "checkpoints", "runs", "logs", "wandb", "mlruns", "node_modules",
    "datasets/raw", "datasets/interim", "datasets/processed", "retrieval/index",
})

#: File suffixes that must never be bundled: raw data and model weights.
FORBIDDEN_SUFFIXES: frozenset[str] = frozenset({
    ".csv", ".parquet", ".npz", ".npy", ".zip", ".tar", ".gz", ".xlsx", ".pkl", ".joblib",
    ".ckpt", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".faiss", ".pyc", ".pyo",
    ".key", ".pem",
})

#: File names that must never be bundled.
FORBIDDEN_NAMES: frozenset[str] = frozenset({
    ".env", ".hf_token", ".netrc", "credentials.json", "secrets.yaml", "adapter_config.json",
})

#: Text suffixes scanned for secrets and local paths.
TEXT_SUFFIXES: frozenset[str] = frozenset({
    ".py", ".md", ".yaml", ".yml", ".json", ".toml", ".cff", ".txt", ".sh", ".cfg", ".jsonl",
    ".gitignore",
})

#: Secret patterns. Mirrors tests/test_repo_hygiene.py; kept here so the
#: archive verifier works standalone on an unpacked bundle.
SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "GitHub personal access token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "OpenAI-style API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile(r"(?i)(password|passwd|api[_-]?key|secret)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
     "credential assignment"),
)

LOCAL_PATH_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z]:\\\\?Users\\\\?"), "Windows user path"),
    (re.compile(r"/home/[a-z][a-z0-9_-]+/"), "Linux home path"),
    (re.compile(r"/Users/[A-Za-z][A-Za-z0-9_-]+/"), "macOS home path"),
    (re.compile("C:" + "/Users/"), "Windows user path"),
)

#: The artefacts the reviewer asked for, keyed for the manifest's ``components``
#: block. Every path must exist in the repository.
COMPONENT_PATHS: dict[str, tuple[str, ...]] = {
    "verifier_rules": ("verifier/rules/claim_primitives.yaml",),
    "verifier_engine": (
        "src/vrfraudnet/verifier/engine.py",
        "src/vrfraudnet/verifier/primitives.py",
        "src/vrfraudnet/verifier/evidence.py",
    ),
    "rationale_schema": ("schemas/rationale_schema.json",),
    "decoding_grammar": ("src/vrfraudnet/models/grammar.py", "docs/DECODING_GRAMMAR.md"),
    "stage2_lora": (
        "configs/stage2_lora.yaml",
        "src/vrfraudnet/models/stage2_rationale.py",
        "scripts/train_stage2_lora.py",
        "scripts/validate_stage2_adapter.py",
        "docs/STAGE2_LORA.md",
        "artifacts/lora/README.md",
        "examples/stage2_corpus_example.jsonl",
    ),
    "retrieval": (
        "retrieval/config/retrieval.yaml",
        "src/vrfraudnet/retrieval/index.py",
        "scripts/build_retrieval_index.py",
        "docs/RETRIEVAL_SETUP.md",
    ),
    "edit_generators": (
        "adversarial/families/edit_families.yaml",
        "src/vrfraudnet/adversarial/edits.py",
        "src/vrfraudnet/adversarial/applicability.py",
        "docs/EDIT_GENERATORS.md",
    ),
    "experiment_configurations": (
        "configs/base.yaml", "configs/d1.yaml", "configs/d2.yaml", "configs/d3.yaml",
        "configs/d4.yaml", "configs/d5.yaml", "configs/baselines.yaml",
        "configs/ablations.yaml", "configs/costs.yaml",
    ),
    "seeds": ("configs/manuscript_seeds.yaml", "src/vrfraudnet/seeds.py"),
    "environment": (
        "requirements.txt", "requirements-optional.txt", "environment.yml", "pyproject.toml",
    ),
    "reproduction_scripts": (
        "reproduce_all.sh", "scripts/prepare_data.py", "scripts/train.py", "scripts/evaluate.py",
        "scripts/calibrate.py", "scripts/robustness.py", "scripts/stats.py",
        "scripts/make_tables.py", "scripts/audit_manuscript.py",
    ),
    "reviewer_documentation": (
        "docs/REVIEWER_REPRODUCIBILITY.md", "docs/ARTIFACT_AVAILABILITY.md",
        "docs/REPRODUCIBILITY.md", "docs/KNOWN_LIMITATIONS.md", "docs/MANUSCRIPT_AUDIT.md",
        "docs/MANUSCRIPT_CODE_MAP.md", "docs/DATA_AVAILABILITY.md", "docs/RELEASE.md",
    ),
}

BASE_MODEL = {
    "identifier": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "revision": None,
    "revision_status": "not recorded by the manuscript or any project artefact (L-29)",
    "licence": "Llama 3.1 Community License",
    "redistributed": False,
}
RETRIEVAL_ENCODER = {
    "identifier": "sentence-transformers/all-MiniLM-L6-v2",
    "revision": None,
    "revision_status": "not recorded by the manuscript or any project artefact (L-29)",
    "embedding_dimension": 384,
    "redistributed": False,
}
ADAPTER_STATUS = {
    "status": "NOT_AVAILABLE",
    "statement": "artifacts/lora/README.md",
    "location": None,
    "sha256": None,
    "detail": (
        "The exact trained LoRA adapter checkpoint used in the original experiments is not "
        "included because a standalone archival copy of the fitted adapter was not retained "
        "as a release-ready artifact during the original experimental workflow. Searched: "
        "working tree, git history, tags, branches, Git LFS, project directories. "
        "No adapter has been fabricated."
    ),
}
DATASET_AVAILABILITY = {
    "D1": {"name": "BAF Base", "redistributed": False, "terms": "CC BY-NC-SA 4.0 (Kaggle)"},
    "D2": {"name": "AMLworld HI-Small", "redistributed": False, "terms": "CDLA-Sharing-1.0 (Kaggle)"},
    "D3": {"name": "IEEE-CIS Fraud Detection", "redistributed": False,
           "terms": "Kaggle competition rules"},
    "D4": {"name": "Elliptic++", "redistributed": False, "terms": "see repository; CC BY-NC-SA 4.0 origin"},
    "D5": {"name": "DGraph-Fin", "redistributed": False, "terms": "research agreement required"},
    "statement": "docs/DATA_AVAILABILITY.md",
}
KNOWN_LIMITATION_REFERENCES = {
    "stage2_corpus": "L-25",
    "stage2_adapter": "L-31",
    "model_revisions": "L-29",
    "stage2_optimisation_details": "L-30",
    "counterfactual_construction": "L-32",
    "retrieval_index_type": "L-33",
    "adversarial_training_loop": "L-34",
    "baseline_hyperparameters": "L-18",
    "cost_values": "L-26",
    "manuscript_audit": "docs/MANUSCRIPT_AUDIT.md",
}


@dataclass(frozen=True)
class BundleFile:
    """One file selected for the archive."""

    relative: str      # POSIX path inside the archive (after the top-level prefix)
    absolute: Path


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_normalised(path: Path) -> bytes:
    """File bytes with CRLF normalised to LF for text files.

    Git stores every file in this repository with LF endings (``core.autocrlf``
    may check them out as CRLF on Windows). Hashing and archiving the LF form
    makes the manifest and the archive identical on every platform and equal to
    what a fresh clone contains. Binary content (any NUL byte) is left untouched.
    """
    data = Path(path).read_bytes()
    if b"\x00" in data:
        return data
    return data.replace(b"\r\n", b"\n")


def sha256_path(path: Path) -> str:
    return hashlib.sha256(read_normalised(path)).hexdigest()


def _is_excluded(relative: str) -> bool:
    parts = relative.split("/")
    for i in range(1, len(parts) + 1):
        if "/".join(parts[:i]) in EXCLUDED_DIRS or parts[i - 1] in EXCLUDED_DIRS:
            return True
    return False


def collect_bundle_files(repo_root: Path) -> list[BundleFile]:
    """Every file the archive contains, sorted by archive path."""
    repo_root = Path(repo_root)
    selected: dict[str, Path] = {}
    for entry in BUNDLE_INCLUDE:
        path = repo_root / entry
        if path.is_file():
            selected[entry] = path
            continue
        if not path.is_dir():
            continue
        for child in sorted(path.rglob("*")):
            if not child.is_file():
                continue
            relative = child.relative_to(repo_root).as_posix()
            if _is_excluded(relative):
                continue
            selected[relative] = child
    return [BundleFile(rel, selected[rel]) for rel in sorted(selected)]


def forbidden_content_problems(files: Iterable[BundleFile]) -> list[str]:
    """Data files, weights, secrets or local paths that must not be shipped."""
    problems: list[str] = []
    for f in files:
        name = Path(f.relative).name
        suffix = Path(f.relative).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"{f.relative}: forbidden suffix {suffix} (data or weights)")
        if name in FORBIDDEN_NAMES:
            problems.append(f"{f.relative}: forbidden file name")
        if suffix in TEXT_SUFFIXES or name in {".gitignore"}:
            try:
                text = f.absolute.read_text(encoding="utf-8", errors="ignore")
            except OSError as exc:  # pragma: no cover
                problems.append(f"{f.relative}: unreadable ({exc})")
                continue
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"{f.relative}: possible {label}")
            if name not in {"test_repo_hygiene.py", "reproducibility.py"}:
                for pattern, label in LOCAL_PATH_PATTERNS:
                    if pattern.search(text):
                        problems.append(f"{f.relative}: {label}")
    return problems


# ---------------------------------------------------------------------------
# Git and environment
# ---------------------------------------------------------------------------

def git_info(repo_root: Path, tag: str | None = None) -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True,
                                 text=True, timeout=15, check=False)
            return out.stdout.strip() if out.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):  # pragma: no cover
            return ""

    commit = run("rev-parse", "HEAD") or None
    dirty = bool(run("status", "--porcelain"))
    exact_tag = tag or (run("describe", "--tags", "--exact-match") or None)
    return {
        "commit": commit,
        "working_tree_dirty": dirty,
        "tag": exact_tag,
        "remote": "https://github.com/abdulbasit12312/VR-FraudNet",
        "note": (
            "File hashes are computed from the working tree at 'commit'. When this manifest is "
            "itself committed, the commit that adds it is the child of 'commit'; the release "
            "archive regenerates the manifest at the tagged commit."
        ),
    }


def environment_info(repo_root: Path) -> dict[str, Any]:
    locks = {}
    for name in ("requirements.txt", "requirements-optional.txt", "environment.yml", "pyproject.toml"):
        path = repo_root / name
        if path.exists():
            locks[name] = sha256_path(path)
    return {
        "python_at_packaging": platform.python_version(),
        "platform_at_packaging": platform.platform(),
        "manuscript_python": "3.11.7",
        "dependency_files_sha256": locks,
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def build_manifest(
    repo_root: Path,
    *,
    version: str,
    tag: str | None = None,
    files: list[BundleFile] | None = None,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble the manifest with real hashes for every bundled file."""
    repo_root = Path(repo_root)
    files = files if files is not None else collect_bundle_files(repo_root)
    hashes = {f.relative: sha256_path(f.absolute) for f in files}

    components: dict[str, Any] = {}
    missing: list[str] = []
    for key, paths in COMPONENT_PATHS.items():
        entries = {}
        for rel in paths:
            if rel in hashes:
                entries[rel] = hashes[rel]
            elif (repo_root / rel).exists():
                entries[rel] = sha256_path(repo_root / rel)
            else:
                missing.append(rel)
        components[key] = entries
    if missing:
        raise FileNotFoundError(f"manifest components reference missing files: {missing}")

    seeds = _load_seed_file(repo_root / "configs" / "manuscript_seeds.yaml")
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "project": PROJECT,
        "manuscript": "Grounding language models with deterministic verifiers for fraud detection",
        "release_version": version,
        "archive_name": f"{ARCHIVE_PREFIX}-{version}.tar.gz",
        "created_utc": created_utc or datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git": git_info(repo_root, tag),
        "environment": environment_info(repo_root),
        "seeds": {
            "path": "configs/manuscript_seeds.yaml",
            "values": seeds,
            "sha256": hashes.get("configs/manuscript_seeds.yaml")
            or sha256_path(repo_root / "configs" / "manuscript_seeds.yaml"),
        },
        "components": components,
        "base_model": BASE_MODEL,
        "retrieval_encoder": RETRIEVAL_ENCODER,
        "stage2_adapter": ADAPTER_STATUS,
        "datasets": DATASET_AVAILABILITY,
        "known_limitation_references": KNOWN_LIMITATION_REFERENCES,
        "excluded_by_construction": [
            "raw and processed benchmark datasets", "model weights and checkpoints",
            "retrieval indexes", "result and prediction files", "caches", "secrets",
        ],
        "file_count": len(hashes),
        "files": hashes,
    }
    return manifest


def _load_seed_file(path: Path) -> list[int]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [int(s) for s in data["seeds"]]


def manifest_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def compare_manifest_files(committed: dict[str, Any], fresh: dict[str, Any]) -> list[str]:
    """Differences in the hash tables between a committed and a freshly built manifest."""
    problems: list[str] = []
    old, new = committed.get("files", {}), fresh.get("files", {})
    for rel in sorted(set(old) | set(new)):
        if rel not in old:
            problems.append(f"{rel}: bundled but absent from the committed manifest")
        elif rel not in new:
            problems.append(f"{rel}: in the committed manifest but no longer bundled")
        elif old[rel] != new[rel]:
            problems.append(f"{rel}: hash changed since the manifest was written")
    if committed.get("seeds", {}).get("values") != fresh.get("seeds", {}).get("values"):
        problems.append("seed values differ")
    if committed.get("release_version") != fresh.get("release_version"):
        problems.append("release_version differs")
    return problems


def checksums_text(hashes: dict[str, str], prefix: str) -> str:
    """``sha256sum -c``-compatible listing, paths relative to the unpacked directory."""
    lines = [f"{digest}  {prefix}/{rel}" for rel, digest in sorted(hashes.items())]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def write_archive(
    files: list[BundleFile],
    *,
    manifest: dict[str, Any],
    output: Path,
    mtime: int,
) -> dict[str, str]:
    """Write a deterministic ``.tar.gz``.

    Determinism: entries in sorted order, fixed ``mtime``, uid/gid 0, empty
    user/group names, mode 0644 (0755 for ``*.sh`` and ``scripts/*.py``), gzip
    header mtime 0 and no filename. Building twice from the same tree and the
    same ``mtime`` gives byte-identical archives.
    """
    prefix = f"{ARCHIVE_PREFIX}-{manifest['release_version']}"
    hashes = dict(manifest["files"])
    manifest_bytes = manifest_json(manifest).encode("utf-8")
    checksums = checksums_text({**hashes, MANIFEST_FILE: sha256_bytes(manifest_bytes)}, prefix)
    checksum_bytes = checksums.encode("utf-8")

    def info(name: str, size: int, executable: bool = False) -> tarfile.TarInfo:
        ti = tarfile.TarInfo(name=f"{prefix}/{name}")
        ti.size = size
        ti.mtime = mtime
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        ti.mode = 0o755 if executable else 0o644
        return ti

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for f in files:
            data = read_normalised(f.absolute)
            executable = f.relative.endswith(".sh") or f.relative.startswith("scripts/")
            tar.addfile(info(f.relative, len(data), executable), io.BytesIO(data))
        tar.addfile(info(MANIFEST_FILE, len(manifest_bytes)), io.BytesIO(manifest_bytes))
        tar.addfile(info(CHECKSUMS_FILE, len(checksum_bytes)), io.BytesIO(checksum_bytes))
    raw = buffer.getvalue()
    with output.open("wb") as handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0, compresslevel=9) as gz:
            gz.write(raw)
    return {"archive": str(output), "sha256": sha256_path(output), "prefix": prefix,
            "manifest_sha256": sha256_bytes(manifest_bytes)}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

REQUIRED_IN_ARCHIVE: tuple[str, ...] = (
    "README.md", "LICENSE", "CITATION.cff", MANIFEST_FILE, CHECKSUMS_FILE,
    "configs/manuscript_seeds.yaml", "configs/stage2_lora.yaml", "configs/base.yaml",
    "schemas/rationale_schema.json", "verifier/rules/claim_primitives.yaml",
    "retrieval/config/retrieval.yaml", "adversarial/families/edit_families.yaml",
    "src/vrfraudnet/models/grammar.py", "src/vrfraudnet/models/stage2_rationale.py",
    "src/vrfraudnet/retrieval/index.py", "src/vrfraudnet/adversarial/edits.py",
    "src/vrfraudnet/adversarial/applicability.py", "src/vrfraudnet/verifier/engine.py",
    "scripts/train_stage2_lora.py", "scripts/validate_stage2_adapter.py",
    "scripts/build_retrieval_index.py", "scripts/verify_reproducibility_bundle.py",
    "docs/REVIEWER_REPRODUCIBILITY.md", "docs/REPRODUCIBILITY.md",
    "docs/ARTIFACT_AVAILABILITY.md", "docs/KNOWN_LIMITATIONS.md", "docs/MANUSCRIPT_AUDIT.md",
    "artifacts/lora/README.md", "requirements.txt", "environment.yml", "reproduce_all.sh",
)


def verify_archive(archive: Path, *, expected_seeds: list[int] | None = None) -> dict[str, Any]:
    """Verify an archive produced by :func:`write_archive`.

    Returns a report with ``ok`` and a list of problems. Nothing is extracted to
    disk; every member is read from the tar stream.
    """
    import yaml

    problems: list[str] = []
    archive = Path(archive)
    if not archive.exists():
        return {"ok": False, "problems": [f"{archive} does not exist"]}
    members: dict[str, bytes] = {}
    with tarfile.open(archive, mode="r:gz") as tar:
        prefix = None
        for member in tar.getmembers():
            if not member.isfile():
                continue
            top, _, rest = member.name.partition("/")
            prefix = prefix or top
            if top != prefix:
                problems.append(f"member outside the archive prefix: {member.name}")
                continue
            handle = tar.extractfile(member)
            members[rest] = handle.read() if handle else b""
    if not members:
        return {"ok": False, "problems": ["archive contains no files"]}

    for required in REQUIRED_IN_ARCHIVE:
        if required not in members:
            problems.append(f"required file missing: {required}")

    manifest: dict[str, Any] = {}
    if MANIFEST_FILE in members:
        try:
            manifest = json.loads(members[MANIFEST_FILE].decode("utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"{MANIFEST_FILE} is not valid JSON: {exc}")
    if manifest:
        if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            problems.append(f"unexpected manifest schema_version {manifest.get('schema_version')!r}")
        for rel, expected in manifest.get("files", {}).items():
            if rel not in members:
                problems.append(f"manifest lists {rel} but it is not in the archive")
            elif sha256_bytes(members[rel]) != expected:
                problems.append(f"SHA-256 mismatch: {rel}")
        listed = set(manifest.get("files", {}))
        for rel in members:
            if rel not in listed and rel not in {MANIFEST_FILE, CHECKSUMS_FILE}:
                problems.append(f"{rel} is in the archive but not in the manifest")
        for key, entries in manifest.get("components", {}).items():
            for rel, expected in entries.items():
                if rel not in members:
                    problems.append(f"component {key}: {rel} missing from archive")
                elif sha256_bytes(members[rel]) != expected:
                    problems.append(f"component {key}: SHA-256 mismatch for {rel}")
        if not manifest.get("files"):
            problems.append("manifest has no file hashes")
        adapter = manifest.get("stage2_adapter", {})
        if adapter.get("status") == "NOT_AVAILABLE" and adapter.get("sha256"):
            problems.append("adapter marked NOT_AVAILABLE but carries a hash")

    if CHECKSUMS_FILE in members:
        for line in members[CHECKSUMS_FILE].decode("utf-8").splitlines():
            if not line.strip():
                continue
            digest, _, path = line.partition("  ")
            rel = path.split("/", 1)[1] if "/" in path else path
            if rel not in members:
                problems.append(f"{CHECKSUMS_FILE} lists {rel} but it is not in the archive")
            elif sha256_bytes(members[rel]) != digest:
                problems.append(f"{CHECKSUMS_FILE}: mismatch for {rel}")

    for rel, data in members.items():
        suffix = Path(rel).suffix.lower()
        if suffix in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden data/weight file in archive: {rel}")
        if Path(rel).name in FORBIDDEN_NAMES:
            problems.append(f"forbidden file in archive: {rel}")
        if suffix in TEXT_SUFFIXES:
            text = data.decode("utf-8", errors="ignore")
            for pattern, label in SECRET_PATTERNS:
                if pattern.search(text):
                    problems.append(f"possible {label} in {rel}")

    if "configs/manuscript_seeds.yaml" in members:
        try:
            seeds = [int(s) for s in yaml.safe_load(members["configs/manuscript_seeds.yaml"])["seeds"]]
        except Exception as exc:  # pragma: no cover
            problems.append(f"seed file unreadable: {exc}")
            seeds = []
        if manifest and seeds != manifest.get("seeds", {}).get("values"):
            problems.append("seed file and manifest seed values disagree")
        if expected_seeds is not None and seeds != list(expected_seeds):
            problems.append("seed file does not match the code constant MANUSCRIPT_SEEDS")
    if "schemas/rationale_schema.json" in members:
        try:
            json.loads(members["schemas/rationale_schema.json"].decode("utf-8"))
        except json.JSONDecodeError as exc:
            problems.append(f"schema does not parse: {exc}")
    for rel, data in members.items():
        if rel.endswith((".yaml", ".yml")):
            try:
                yaml.safe_load(data.decode("utf-8"))
            except yaml.YAMLError as exc:
                problems.append(f"{rel} does not parse as YAML: {exc}")
        elif rel.endswith(".json"):
            try:
                json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"{rel} does not parse as JSON: {exc}")

    return {
        "ok": not problems,
        "problems": problems,
        "archive": str(archive),
        "archive_sha256": sha256_path(archive),
        "file_count": len(members),
        "release_version": manifest.get("release_version"),
        "git": manifest.get("git"),
    }
