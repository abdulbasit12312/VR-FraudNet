"""Run manifests, result serialisation, and artefact-presence guards.

Every artefact this repository writes carries a manifest describing exactly how
it was produced. Table builders refuse to consume a result file whose manifest
does not match the table they are asked to build. That is what keeps
"reproduced" honest: a table can only be rendered from a run that actually
happened.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from vrfraudnet.errors import MissingArtefactError

SCHEMA_VERSION = "vrfraudnet-result/1"


def _git_revision() -> str:
    """Best-effort git revision of the working tree, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        rev = out.stdout.strip()
        if not rev:
            return "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return rev + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # pragma: no cover - git may be absent
        return "unknown"


def _environment() -> dict[str, Any]:
    """Environment facts needed to interpret a result file.

    Deliberately excludes the username and hostname: result files are meant to
    be committed or shared, and neither field is scientifically relevant.
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }
    for name in ("numpy", "scipy", "sklearn", "lightgbm", "xgboost", "torch"):
        try:
            module = __import__(name)
            env[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            env[name] = "not-installed"
    return env


@dataclass
class RunManifest:
    """Provenance record attached to every result file."""

    experiment: str
    dataset: str
    model: str
    seed: int
    config_path: str
    config_sha256: str
    partition: str
    schema_version: str = SCHEMA_VERSION
    git_revision: str = field(default_factory=_git_revision)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    environment: dict[str, Any] = field(default_factory=_environment)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_of_file(path: str | Path) -> str:
    """Stream a SHA-256 of a file (used for config and dataset fingerprints)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_result(
    path: str | Path,
    manifest: RunManifest,
    payload: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a JSON result file consisting of a manifest plus a payload."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite to replace it, or write to a new "
            f"results directory. Result files are never silently clobbered."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {"manifest": manifest.to_dict(), "results": payload}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return path


def _json_default(obj: Any) -> Any:
    """Serialise NumPy scalars and arrays that survive into a payload."""
    try:
        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serialisable")


def read_result(path: str | Path) -> dict[str, Any]:
    """Read a result file written by :func:`write_result`, validating its schema."""
    path = Path(path)
    if not path.exists():
        raise MissingArtefactError(
            f"required result file not found: {path}\n"
            f"Produce it with the command named for this artefact in "
            f"docs/MANUSCRIPT_CODE_MAP.md. This repository never fabricates results."
        )
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    manifest = document.get("manifest", {})
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported result schema {manifest.get('schema_version')!r}; "
            f"this build expects {SCHEMA_VERSION!r}"
        )
    return document


def require_file(path: str | Path, *, produced_by: str) -> Path:
    """Assert a required input exists, naming the command that produces it."""
    path = Path(path)
    if not path.exists():
        raise MissingArtefactError(
            f"required artefact not found: {path}\nProduce it with:\n    {produced_by}"
        )
    return path


def collect_results(root: str | Path, pattern: str = "*.json") -> list[dict[str, Any]]:
    """Load every result document under ``root`` matching ``pattern``.

    Returns an empty list when the directory does not exist, so callers can give
    a precise "no runs found, here is how to produce them" message rather than a
    traceback.
    """
    root = Path(root)
    if not root.exists():
        return []
    return [read_result(p) for p in sorted(root.rglob(pattern))]
