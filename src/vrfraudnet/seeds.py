"""The ten random seeds reported in manuscript Section 4.8.1.

    "Ten independent runs were conducted using random seeds 42, 123, 456, 789,
     2024, 3141, 2718, 1001, 1729, and 4096. The same seeds and data partitions
     were used for VR-FraudNet and all stochastic baselines."

The list is frozen: any code path that needs "the manuscript seeds" imports
:data:`MANUSCRIPT_SEEDS` rather than restating the numbers.
"""

from __future__ import annotations

from typing import Final, Sequence

#: Seeds exactly as listed in manuscript Section 4.8.1, in manuscript order.
MANUSCRIPT_SEEDS: Final[tuple[int, ...]] = (
    42,
    123,
    456,
    789,
    2024,
    3141,
    2718,
    1001,
    1729,
    4096,
)

#: Number of paired observations obtained by pooling 10 seeds over D1-D3.
#: Manuscript Section 4.8.1 ties this to the maximum Wilcoxon statistic 465.
#: See docs/MANUSCRIPT_AUDIT.md finding A-04 on the independence assumption.
N_POOLED_PAIRS: Final[int] = 30


def wilcoxon_max_statistic(n_pairs: int) -> int:
    """Maximum attainable Wilcoxon signed-rank statistic V for ``n_pairs`` pairs.

    ``V_max = n(n + 1) / 2``. For ``n = 30`` this is 465, the value reported in
    manuscript Table 6(a).
    """
    if n_pairs < 1:
        raise ValueError("n_pairs must be >= 1")
    return n_pairs * (n_pairs + 1) // 2


def validate_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    """Validate a user-supplied seed list.

    Duplicates are rejected because they would silently inflate the apparent
    number of independent runs.
    """
    out = tuple(int(s) for s in seeds)
    if len(set(out)) != len(out):
        raise ValueError(f"duplicate seeds are not permitted: {out}")
    if not out:
        raise ValueError("at least one seed is required")
    return out


def is_manuscript_seed_set(seeds: Sequence[int]) -> bool:
    """True when ``seeds`` is exactly the manuscript seed set (order-insensitive)."""
    return sorted(int(s) for s in seeds) == sorted(MANUSCRIPT_SEEDS)
