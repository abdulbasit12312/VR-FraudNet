"""Pytest configuration: put ``src`` on the path without requiring installation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260825)


@pytest.fixture
def toy_frame():
    """A tiny chronologically ordered frame used by the leakage tests."""
    import pandas as pd

    n = 200
    months = np.repeat(np.arange(8), n // 8)
    values = np.linspace(0.0, 1.0, n)
    labels = (np.arange(n) % 17 == 0).astype(int)
    return pd.DataFrame(
        {"month": months, "value": values, "category": ["a", "b"] * (n // 2), "y": labels}
    )
