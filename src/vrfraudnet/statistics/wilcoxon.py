"""Paired Wilcoxon signed-rank testing on AUPRC (manuscript Section 5.2, Table 6a).

Manuscript specification
------------------------
* One-sided paired Wilcoxon signed-rank test of VR-FraudNet against each
  baseline, on paired observations pooled across D1, D2 and D3.
* The one-sided alternative (improvement in AUPRC) was predeclared.
* 10 seeds x 3 datasets = 30 paired observations, consistent with the reported
  maximum statistic ``V = 465``.
* Holm-Bonferroni correction across the nine baseline comparisons.

AUDIT FINDING A-04 (surfaced, not silently accepted)
----------------------------------------------------
Pooling paired differences across three datasets whose AUPRC scales differ by a
factor of roughly 1.5 (0.52 on D1, 0.58 on D2, 0.75 on D3) treats
(dataset, seed) cells as exchangeable units. The ten seeds *within* a dataset are
exchangeable; observations *across* datasets are not, because the paired
difference distribution is dataset-dependent. The test therefore answers "is the
median paired improvement over this particular mixture of three datasets
positive", not "is the improvement consistent across fraud datasets". This
implementation runs the pooled test the manuscript describes **and** the
per-dataset tests, and reports both, so a reviewer can see whether the
conclusion survives the stricter reading.

AUDIT FINDING A-02
------------------
The mean AUPRC differences in Table 6(a) do not equal the mean of the
per-dataset differences implied by Table 5. Example, against TabTransformer:
D1 +2.35 pp, D2 +6.87 pp, D3 +2.22 pp gives a mean of +3.81 pp, but Table 6(a)
reports +3.65 pp. Against LightGBM the implied mean is +6.05 pp against a
reported +4.94 pp; against Logistic Regression +24.68 pp against +20.18 pp. See
docs/MANUSCRIPT_AUDIT.md. This module computes the differences from the actual
per-seed results, so any such gap will reappear as a discrepancy rather than be
reproduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from vrfraudnet.seeds import wilcoxon_max_statistic


@dataclass
class WilcoxonResult:
    """One paired comparison."""

    baseline: str
    n_pairs: int
    mean_delta_pp: float
    median_delta_pp: float
    statistic_v: float
    max_statistic_v: int
    p_value_one_sided: float
    effect_size_cliffs_delta: float
    scope: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "scope": self.scope,
            "n_pairs": self.n_pairs,
            "mean_delta_pp": self.mean_delta_pp,
            "median_delta_pp": self.median_delta_pp,
            "wilcoxon_V": self.statistic_v,
            "wilcoxon_V_max": self.max_statistic_v,
            "p_raw_one_sided": self.p_value_one_sided,
            "effect_size_cliffs_delta": self.effect_size_cliffs_delta,
            "note": self.note,
        }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta, the non-parametric effect size for paired/unpaired samples.

    The manuscript reports "Effect size ... as delta" without naming the
    estimator. Cliff's delta is the effect size conventionally paired with a
    Wilcoxon test, and is what is computed here; the ambiguity is recorded as
    audit note A-12.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    greater = int(np.sum(a[:, None] > b[None, :]))
    less = int(np.sum(a[:, None] < b[None, :]))
    return float((greater - less) / (a.size * b.size))


def paired_wilcoxon(
    model_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    baseline_name: str,
    scope: str,
    alternative: str = "greater",
) -> WilcoxonResult:
    """One-sided paired Wilcoxon signed-rank test on AUPRC.

    ``model_scores[i]`` and ``baseline_scores[i]`` must be the same
    (dataset, seed) cell evaluated under identical splits, as manuscript S4.8.1
    guarantees ("The same seeds and data partitions were used for VR-FraudNet
    and all stochastic baselines").
    """
    model = np.asarray(model_scores, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    if model.shape != base.shape:
        raise ValueError("paired arrays must have identical shape")
    if model.size < 6:
        raise ValueError(
            f"a Wilcoxon signed-rank test with n={model.size} cannot reach conventional "
            "significance; refusing to report it"
        )

    differences = model - base
    result = stats.wilcoxon(model, base, alternative=alternative, zero_method="wilcox")

    return WilcoxonResult(
        baseline=baseline_name,
        n_pairs=int(model.size),
        mean_delta_pp=float(np.mean(differences) * 100.0),
        median_delta_pp=float(np.median(differences) * 100.0),
        statistic_v=float(result.statistic),
        max_statistic_v=wilcoxon_max_statistic(int(model.size)),
        p_value_one_sided=float(result.pvalue),
        effect_size_cliffs_delta=cliffs_delta(model, base),
        scope=scope,
    )


def pooled_and_per_dataset(
    model_by_dataset: Mapping[str, Sequence[float]],
    baseline_by_dataset: Mapping[str, Sequence[float]],
    *,
    baseline_name: str,
) -> dict[str, WilcoxonResult]:
    """Run the manuscript's pooled test and the per-dataset tests side by side.

    Returns a mapping keyed by ``"pooled"`` and by each dataset id. Reporting
    both is the honest response to audit finding A-04: the pooled test is what
    the manuscript describes, and the per-dataset tests are what a reviewer will
    ask for.
    """
    datasets = sorted(set(model_by_dataset) & set(baseline_by_dataset))
    if not datasets:
        raise ValueError("no overlapping datasets between model and baseline results")

    out: dict[str, WilcoxonResult] = {}
    pooled_model: list[float] = []
    pooled_base: list[float] = []
    for dataset in datasets:
        model = list(model_by_dataset[dataset])
        base = list(baseline_by_dataset[dataset])
        pooled_model.extend(model)
        pooled_base.extend(base)
        if len(model) >= 6:
            out[dataset] = paired_wilcoxon(
                model, base, baseline_name=baseline_name, scope=dataset
            )

    pooled = paired_wilcoxon(
        pooled_model, pooled_base, baseline_name=baseline_name, scope="pooled-D1-D3"
    )
    pooled.note = (
        "Pooled across datasets as in manuscript Table 6(a). Paired differences from "
        "different datasets are not exchangeable; see audit finding A-04. The "
        "per-dataset results in this same object are the stricter reading."
    )
    out["pooled"] = pooled
    return out
