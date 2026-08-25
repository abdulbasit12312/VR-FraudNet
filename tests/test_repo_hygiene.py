"""Repository hygiene: no secrets, no local paths, no data, no fabricated results.

These tests are part of the release gate. They are cheap, they run on every
commit, and they are the reason a reviewer can trust that this package does not
quietly ship a dataset, a token or a hard-coded manuscript number.
"""

from __future__ import annotations

import re
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".toml", ".cff", ".txt", ".sh", ".cfg"}
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", ".venv", "venv", "node_modules",
             ".mypy_cache", ".ruff_cache", "results", "datasets"}

#: Files permitted to quote manuscript numbers, and why.
AUDIT_INPUT_FILES = {
    "scripts/audit_manuscript.py",   # quotes published values in order to test them
    "docs/MANUSCRIPT_AUDIT.md",
    "docs/MANUSCRIPT_CODE_MAP.md",
    "docs/KNOWN_LIMITATIONS.md",
    "docs/REPRODUCIBILITY.md",
    "README.md",
    "src/vrfraudnet/evaluation/latency.py",   # quotes the p99 for the plausibility check
    "src/vrfraudnet/data/registry.py",        # Table 1 dataset characteristics
    "src/vrfraudnet/data/ieee_cis.py",
    "src/vrfraudnet/evaluation/metrics.py",
    "src/vrfraudnet/evaluation/temporal.py",
    "src/vrfraudnet/statistics/wilcoxon.py",
    "src/vrfraudnet/adversarial/applicability.py",
    "src/vrfraudnet/models/stage1_triage.py",  # per-dataset routing thresholds
    "src/vrfraudnet/tables/build_tables.py",
    "tests/test_metrics.py",
    "tests/test_repo_hygiene.py",
    # Declarative protocol files. They are documentation of the manuscript's own
    # protocol and are not read by the experimental pipeline; the audit notes
    # they carry quote published values in order to explain a finding.
    "evaluation/protocols/temporal.yaml",
    "evaluation/protocols/metrics.yaml",
    "evaluation/protocols/latency.yaml",
    "evaluation/protocols/transfer.yaml",
    "statistics/tests/tests.yaml",
    "adversarial/families/edit_families.yaml",
}

SECRET_PATTERNS = [
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), "GitHub personal access token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "GitHub fine-grained token"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}"), "OpenAI-style API key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"hf_[A-Za-z0-9]{30,}"), "Hugging Face token"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
]

LOCAL_PATH_PATTERNS = [
    (re.compile(r"[A-Za-z]:\\\\?Users\\\\?"), "Windows user path"),
    (re.compile(r"/home/[a-z][a-z0-9_-]+/"), "Linux home path"),
    (re.compile(r"/Users/[A-Za-z][A-Za-z0-9_-]+/"), "macOS home path"),
    (re.compile(r"C:/Users/"), "Windows user path"),
]

DATA_SUFFIXES = {".csv", ".parquet", ".npz", ".npy", ".zip", ".tar", ".gz", ".xlsx"}
WEIGHT_SUFFIXES = {".ckpt", ".safetensors", ".bin", ".pt", ".pth", ".onnx", ".pkl", ".joblib"}
MAX_FILE_BYTES = 1_000_000


def _tracked_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out.append(path)
    return out


def _text_files(repo_root: Path) -> list[Path]:
    return [p for p in _tracked_files(repo_root) if p.suffix in TEXT_SUFFIXES]


def test_no_secrets_anywhere(repo_root):
    offenders = []
    for path in _text_files(repo_root):
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in SECRET_PATTERNS:
            if pattern.search(content):
                offenders.append(f"{path.relative_to(repo_root)}: {label}")
    assert not offenders, "possible secrets found:\n" + "\n".join(offenders)


def test_no_local_filesystem_paths(repo_root):
    offenders = []
    for path in _text_files(repo_root):
        if path.name == "test_repo_hygiene.py":
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for pattern, label in LOCAL_PATH_PATTERNS:
            match = pattern.search(content)
            if match:
                offenders.append(f"{path.relative_to(repo_root)}: {label} ({match.group(0)!r})")
    assert not offenders, "local paths leaked into the repository:\n" + "\n".join(offenders)


def test_no_datasets_committed(repo_root):
    offenders = [
        str(p.relative_to(repo_root))
        for p in _tracked_files(repo_root)
        if p.suffix.lower() in DATA_SUFFIXES
    ]
    assert not offenders, "dataset files must never be committed:\n" + "\n".join(offenders)


def test_no_model_weights_committed(repo_root):
    offenders = [
        str(p.relative_to(repo_root))
        for p in _tracked_files(repo_root)
        if p.suffix.lower() in WEIGHT_SUFFIXES
    ]
    assert not offenders, "model weights must never be committed:\n" + "\n".join(offenders)


def test_no_oversized_files(repo_root):
    offenders = [
        f"{p.relative_to(repo_root)} ({p.stat().st_size} bytes)"
        for p in _tracked_files(repo_root)
        if p.stat().st_size > MAX_FILE_BYTES
    ]
    assert not offenders, "files above 1 MB:\n" + "\n".join(offenders)


def test_gitignore_blocks_data_and_weights(repo_root):
    content = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for required in ("datasets/raw/", "checkpoints/", "*.safetensors", ".env", "results/**"):
        assert required in content, f".gitignore is missing {required!r}"


def test_no_manuscript_numbers_in_the_runtime_path(repo_root):
    """The decisive anti-fabrication check.

    Manuscript AUPRC values must not appear anywhere that could be read by the
    experimental pipeline. They are permitted only in documentation and in
    scripts/audit_manuscript.py, whose entire purpose is to test them.
    """
    forbidden = ["0.5247", "0.5824", "0.7456", "0.5012", "0.5137", "0.7234",
                 "0.6823", "0.6087", "0.9156", "0.9812", "0.9312"]
    offenders = []
    for path in _text_files(repo_root):
        relative = path.relative_to(repo_root).as_posix()
        if relative in AUDIT_INPUT_FILES:
            continue
        content = path.read_text(encoding="utf-8", errors="ignore")
        for value in forbidden:
            if value in content:
                offenders.append(f"{relative}: contains manuscript value {value}")
    assert not offenders, (
        "manuscript result values found outside the audit and documentation "
        "allow-list:\n" + "\n".join(offenders)
    )


def test_required_documentation_exists(repo_root):
    for name in (
        "README.md",
        "LICENSE",
        "CITATION.cff",
        "docs/MANUSCRIPT_CODE_MAP.md",
        "docs/REPRODUCIBILITY.md",
        "docs/DATA_AVAILABILITY.md",
        "docs/KNOWN_LIMITATIONS.md",
        "docs/MANUSCRIPT_AUDIT.md",
    ):
        assert (repo_root / name).exists(), f"missing required file: {name}"


def test_every_python_file_compiles(repo_root):
    import py_compile

    failures = []
    for path in _tracked_files(repo_root):
        if path.suffix != ".py":
            continue
        try:
            py_compile.compile(str(path), doraise=True, cfile=str(path) + "c")
        except py_compile.PyCompileError as exc:
            failures.append(f"{path.relative_to(repo_root)}: {exc.msg}")
        finally:
            Path(str(path) + "c").unlink(missing_ok=True)
    assert not failures, "syntax errors:\n" + "\n".join(failures)


def test_all_yaml_files_parse(repo_root):
    import yaml

    failures = []
    for path in _tracked_files(repo_root):
        if path.suffix not in {".yaml", ".yml"}:
            continue
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            failures.append(f"{path.relative_to(repo_root)}: {exc}")
    assert not failures, "invalid YAML:\n" + "\n".join(failures)


def test_all_json_files_parse(repo_root):
    import json

    failures = []
    for path in _tracked_files(repo_root):
        if path.suffix != ".json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{path.relative_to(repo_root)}: {exc}")
    assert not failures, "invalid JSON:\n" + "\n".join(failures)


def test_no_todo_or_fixme_markers_in_source(repo_root):
    """A reproducibility package must not ship placeholders."""
    offenders = []
    marker = re.compile(r"\b(TODO|FIXME|XXX|HACK)\b")
    for path in _tracked_files(repo_root):
        if path.suffix != ".py" or path.name == "test_repo_hygiene.py":
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if marker.search(line):
                offenders.append(f"{path.relative_to(repo_root)}:{i}: {line.strip()}")
    assert not offenders, "placeholder markers in source:\n" + "\n".join(offenders)
