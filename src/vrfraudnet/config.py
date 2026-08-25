"""Configuration loading with explicit manuscript provenance.

Every leaf value in a config file carries a provenance tag so a reviewer can
tell manuscript-specified settings apart from choices this repository had to
make in order for the code to run.

Tag vocabulary
--------------
``MANUSCRIPT``
    Stated verbatim in the manuscript. The ``source`` field names the section,
    table, or figure.
``ASSUMPTION``
    Not stated in the manuscript. A default was chosen so the software is
    executable. Every ASSUMPTION must have a matching entry in
    docs/KNOWN_LIMITATIONS.md, referenced by its ``limitation`` field.
``DERIVED``
    Computed from other manuscript values (for example the review-band width
    implied by a pair of routing thresholds).

Provenance is expressed in YAML as a sibling ``_provenance`` mapping, so the
config itself stays readable:

.. code-block:: yaml

    stage1:
      num_leaves: 64
    _provenance:
      stage1.num_leaves: {tag: MANUSCRIPT, source: "S4.8.1"}
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from vrfraudnet.errors import ConfigurationError

VALID_TAGS = frozenset({"MANUSCRIPT", "ASSUMPTION", "DERIVED"})


@dataclass(frozen=True)
class Provenance:
    """Where a single configuration value came from."""

    tag: str
    source: str = ""
    limitation: str = ""

    def __post_init__(self) -> None:
        if self.tag not in VALID_TAGS:
            raise ConfigurationError(
                f"unknown provenance tag {self.tag!r}; expected one of {sorted(VALID_TAGS)}"
            )
        if self.tag == "ASSUMPTION" and not self.limitation:
            raise ConfigurationError(
                "ASSUMPTION entries must name a docs/KNOWN_LIMITATIONS.md id "
                "in their 'limitation' field"
            )
        if self.tag == "MANUSCRIPT" and not self.source:
            raise ConfigurationError(
                "MANUSCRIPT entries must name the manuscript section or table "
                "in their 'source' field"
            )


class Config(Mapping[str, Any]):
    """An immutable, dotted-path view over a loaded configuration tree."""

    def __init__(self, data: Mapping[str, Any], provenance: Mapping[str, Provenance]) -> None:
        self._data = copy.deepcopy(dict(data))
        self._provenance = dict(provenance)

    # -- Mapping protocol ---------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.get(key, _raise=True)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    # -- Access -------------------------------------------------------------
    def get(self, dotted: str, default: Any = None, *, _raise: bool = False) -> Any:
        """Look up a value by dotted path, e.g. ``"stage1.num_leaves"``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                if _raise:
                    raise ConfigurationError(f"missing configuration key: {dotted!r}")
                return default
            node = node[part]
        return node

    def require(self, dotted: str) -> Any:
        """Look up a value, raising :class:`ConfigurationError` when absent."""
        return self.get(dotted, _raise=True)

    def provenance_of(self, dotted: str) -> Provenance | None:
        """Return the provenance record for a dotted path, if one was declared."""
        return self._provenance.get(dotted)

    def assumptions(self) -> dict[str, Provenance]:
        """Every ASSUMPTION-tagged value in this config, keyed by dotted path."""
        return {k: v for k, v in self._provenance.items() if v.tag == "ASSUMPTION"}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Config(keys={sorted(self._data)}, provenance={len(self._provenance)} entries)"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` and return a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, *, base: str | Path | None = None) -> Config:
    """Load a YAML config, optionally merged onto a base config.

    When the config declares ``extends: <relative path>`` the referenced file is
    loaded first and this file is merged on top of it. Provenance mappings are
    merged the same way, so a derived config can re-tag a value it overrides.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigurationError(f"config root must be a mapping: {path}")

    parent = raw.pop("extends", base)
    data: dict[str, Any] = {}
    prov: dict[str, Provenance] = {}
    if parent:
        parent_path = (path.parent / parent).resolve()
        parent_cfg = load_config(parent_path)
        data = parent_cfg.as_dict()
        prov = dict(parent_cfg._provenance)  # noqa: SLF001 - deliberate merge

    raw_prov = raw.pop("_provenance", {}) or {}
    data = _deep_merge(data, raw)
    for dotted, record in raw_prov.items():
        if not isinstance(record, Mapping):
            raise ConfigurationError(f"_provenance[{dotted!r}] must be a mapping")
        prov[dotted] = Provenance(**record)

    return Config(data, prov)


def summarise_assumptions(config: Config) -> str:
    """Render a human-readable list of every ASSUMPTION in a config.

    Printed at the top of every training run so an operator can never claim
    they did not see which settings were not taken from the manuscript.
    """
    items = config.assumptions()
    if not items:
        return "No ASSUMPTION-tagged settings in this configuration."
    lines = ["ASSUMPTION-tagged settings (NOT specified by the manuscript):"]
    for dotted in sorted(items):
        rec = items[dotted]
        lines.append(f"  - {dotted}: value={config.get(dotted)!r} [{rec.limitation}]")
    return "\n".join(lines)
