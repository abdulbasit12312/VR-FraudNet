"""Dataset acquisition, splitting and leakage-controlled preprocessing.

Public entry point: :func:`load_prepared`, which dispatches on the manuscript
dataset id (``"D1"`` ... ``"D5"``) and returns a
:class:`~vrfraudnet.data.types.PreparedDataset`.
"""

from __future__ import annotations

from pathlib import Path

from vrfraudnet.config import Config
from vrfraudnet.data.registry import DATASETS, DatasetSpec, get_spec  # noqa: F401
from vrfraudnet.data.splits import Split  # noqa: F401
from vrfraudnet.data.types import (  # noqa: F401
    LeakageControls,
    PreparedDataset,
    TemporalGraph,
)
from vrfraudnet.errors import ConfigurationError


def load_prepared(dataset_id: str, config: Config, *, root: str | Path) -> PreparedDataset:
    """Load and preprocess one benchmark from a local dataset root.

    Raises
    ------
    vrfraudnet.errors.MissingArtefactError
        When the raw files are not present. No synthetic substitute is ever
        generated: an absent dataset is an absent dataset.
    """
    key = dataset_id.upper()
    root = Path(root)

    if key == "D1":
        from vrfraudnet.data import baf

        return baf.prepare(baf.load_raw(root), config)
    if key == "D2":
        from vrfraudnet.data import amlworld

        return amlworld.prepare(amlworld.load_raw(root), config)
    if key == "D3":
        from vrfraudnet.data import ieee_cis

        return ieee_cis.prepare(ieee_cis.load_raw(root), config)
    if key == "D4":
        from vrfraudnet.data import elliptic_pp

        feats, classes, edges = elliptic_pp.load_raw(root)
        return elliptic_pp.prepare(feats, classes, edges, config)
    if key == "D5":
        from vrfraudnet.data import dgraph

        return dgraph.prepare(dgraph.load_raw(root), config)

    raise ConfigurationError(f"unknown dataset id {dataset_id!r}; expected D1-D5")

__all__ = [
    "load_prepared",
    "DATASETS",
    "DatasetSpec",
    "get_spec",
    "Split",
    "PreparedDataset",
    "TemporalGraph",
    "LeakageControls",
]
