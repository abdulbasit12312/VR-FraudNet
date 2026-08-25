"""Deterministic execution helpers.

The manuscript reports mean +/- standard deviation across ten seeded runs. For
that to be meaningful, a single (seed, dataset, config) triple must be
reproducible on fixed hardware. This module centralises every global
random-state switch so that no other module has to remember them.

Honest limits (also recorded in docs/REPRODUCIBILITY.md):

* GPU floating-point reductions are not bitwise reproducible across different
  GPU architectures, CUDA versions, or thread counts.
* LightGBM is deterministic for a fixed ``num_threads``; changing the thread
  count changes histogram construction order and therefore the model.
* ``torch.use_deterministic_algorithms(True)`` makes some kernels raise rather
  than fall back to a non-deterministic implementation. That is intentional,
  and this module uses ``warn_only=True`` so a missing deterministic kernel
  degrades to a loud warning rather than an unrecoverable crash.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeterminismReport:
    """What was actually pinned, for inclusion in run manifests."""

    seed: int
    python_hash_seed: str
    numpy_seeded: bool
    torch_seeded: bool
    torch_deterministic: bool
    cublas_workspace: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_global_seed(seed: int, *, strict_torch: bool = True) -> DeterminismReport:
    """Seed every RNG this project can reach.

    Parameters
    ----------
    seed:
        The run seed. Use a value from :data:`vrfraudnet.seeds.MANUSCRIPT_SEEDS`
        when reproducing a manuscript table.
    strict_torch:
        Enable ``torch.use_deterministic_algorithms``. Disable only when a
        required kernel has no deterministic implementation, and record that
        fact in the run manifest.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch_seeded = False
    torch_deterministic = False
    cublas: str | None = None
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is optional on some paths
        _LOG.info("torch not installed; skipping torch seeding")
    else:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch_seeded = True
        if strict_torch:
            # Must be set before the first CUDA context is created.
            cublas = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            torch_deterministic = True

    return DeterminismReport(
        seed=seed,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        numpy_seeded=True,
        torch_seeded=torch_seeded,
        torch_deterministic=torch_deterministic,
        cublas_workspace=cublas,
    )


def derive_subseed(seed: int, *tags: str) -> int:
    """Derive a stable child seed from a run seed and a set of string tags.

    Used so that, for example, the adversarial edit sampler and the model
    initialiser consume independent streams while both remain a deterministic
    function of the single run seed. Unlike :func:`hash`, this is stable across
    processes and platforms.
    """
    payload = "|".join((str(seed), *tags)).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], "big") % (2**31 - 1)


def rng_for(seed: int, *tags: str) -> np.random.Generator:
    """Return a NumPy :class:`~numpy.random.Generator` for a named substream."""
    return np.random.default_rng(derive_subseed(seed, *tags))
