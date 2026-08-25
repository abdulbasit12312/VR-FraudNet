"""Recompute every arithmetic consistency check in docs/MANUSCRIPT_AUDIT.md.

Usage
-----
    python scripts/audit_manuscript.py
    python scripts/audit_manuscript.py --json

This script contains manuscript-reported values **as audit inputs only**. They
are never used to produce a result, a metric, or a table; they exist so that a
reader can rerun the arithmetic behind each finding and see for themselves. The
distinction matters: quoting a published number in order to test it is the
opposite of hard-coding it in order to fake reproducing it.

Every check returns PASS (the manuscript is internally consistent on that point)
or FAIL (it is not), with the arithmetic shown.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vrfraudnet.evaluation.latency import check_escalation_plausibility  # noqa: E402
from vrfraudnet.evaluation.metrics import recall_at_top_k_ceiling  # noqa: E402
from vrfraudnet.seeds import wilcoxon_max_statistic  # noqa: E402


@dataclass
class Check:
    finding: str
    title: str
    passed: bool
    detail: str


# --- Manuscript values used ONLY as audit inputs ----------------------------
TABLE_5_AUPRC = {
    "D1": {"logistic_regression": 0.3245, "isolation_forest": 0.2987, "mlp": 0.4123,
           "cnn1d": 0.4287, "lstm": 0.4456, "random_forest": 0.4534, "xgboost": 0.4823,
           "lightgbm": 0.4912, "tabtransformer": 0.5012, "vr_fraudnet": 0.5247},
    "D2": {"logistic_regression": 0.2645, "isolation_forest": 0.2398, "mlp": 0.3287,
           "cnn1d": 0.3478, "lstm": 0.3823, "random_forest": 0.4012, "xgboost": 0.4523,
           "lightgbm": 0.4712, "tabtransformer": 0.5137, "vr_fraudnet": 0.5824},
    "D3": {"logistic_regression": 0.5234, "isolation_forest": 0.4823, "mlp": 0.6234,
           "cnn1d": 0.6398, "lstm": 0.6534, "random_forest": 0.6712, "xgboost": 0.6987,
           "lightgbm": 0.7087, "tabtransformer": 0.7234, "vr_fraudnet": 0.7456},
}
TABLE_6A_MEAN_DELTA_PP = {
    "logistic_regression": 20.18, "isolation_forest": 22.45, "mlp": 12.18,
    "cnn1d": 10.78, "lstm": 9.04, "random_forest": 8.32, "xgboost": 5.45,
    "lightgbm": 4.94, "tabtransformer": 3.65,
}
TABLE_5C_RECALL_TOP1 = {"logistic_regression": 0.4567, "isolation_forest": 0.4234,
                        "vr_fraudnet": 0.6234, "tabtransformer": 0.6034}
TABLE_8C_TABTRANSFORMER_DIAGONAL = {"D1": 0.5012, "D2": 0.5187, "D3": 0.7234}
TABLE_9A_D1 = {"reference_labelled_train": 0.5247, "test_m6": 0.5089, "test_m7": 0.4945}
DECLARED_PREVALENCE = {"D1": 0.0110, "D2": 0.0010, "D3": 0.0350, "D4": 0.0223, "D5": 0.0130}
REPORTED_P99_ESCALATION_MS = 175.23
MAX_OUTPUT_TOKENS = 384


def check_a01() -> Check:
    """Recall@top-1% on D3 exceeds its arithmetic ceiling."""
    ceiling = recall_at_top_k_ceiling(DECLARED_PREVALENCE["D3"])
    offenders = {k: v for k, v in TABLE_5C_RECALL_TOP1.items() if v > ceiling}
    return Check(
        "A-01",
        "Recall@top-1% ceiling on D3",
        passed=not offenders,
        detail=(
            f"D3 prevalence {DECLARED_PREVALENCE['D3']:.2%} implies a ceiling of "
            f"0.01/{DECLARED_PREVALENCE['D3']} = {ceiling:.4f}. Table 5(c) values above "
            f"that ceiling: {offenders}. Every model in Table 5(c) exceeds it, so the "
            f"whole column is unattainable, not just the proposed model's entry."
        ),
    )


def check_a02() -> Check:
    """Table 6(a) mean AUPRC deltas vs the means implied by Table 5."""
    mismatches = {}
    for baseline, reported in TABLE_6A_MEAN_DELTA_PP.items():
        implied = sum(
            (TABLE_5_AUPRC[d]["vr_fraudnet"] - TABLE_5_AUPRC[d][baseline]) * 100.0
            for d in ("D1", "D2", "D3")
        ) / 3.0
        if abs(implied - reported) > 0.05:
            mismatches[baseline] = {"reported_pp": reported, "implied_pp": round(implied, 2)}
    return Check(
        "A-02",
        "Table 6(a) mean deltas vs Table 5",
        passed=not mismatches,
        detail=(
            "Mean paired AUPRC difference implied by Table 5 does not match the value "
            f"printed in Table 6(a) for: {json.dumps(mismatches, indent=2)}"
        ),
    )


def check_a04() -> Check:
    """The reported Wilcoxon V_max is consistent with 30 pairs."""
    v_max = wilcoxon_max_statistic(30)
    return Check(
        "A-04",
        "Wilcoxon V_max consistency (arithmetic only)",
        passed=v_max == 465,
        detail=(
            f"n(n+1)/2 for n=30 is {v_max}, matching the reported max of 465. The "
            "arithmetic is consistent; the exchangeability objection to pooling three "
            "datasets is separate and is not settled by this check."
        ),
    )


def check_a08_a09() -> Check:
    """Declared feature counts against the stated preprocessing operations."""
    issues = []
    issues.append(
        "D1: Table 2 requires both 'exclude protected attrs' and 'Output to model: "
        "30 features'; BAF Base has 30 predictors before exclusion, so the two cannot "
        "both hold."
    )
    issues.append(
        "D3: 394 transaction + 41 identity columns, minus TransactionID/isFraud/"
        "TransactionDT, minus 12 dropped D-columns, with 339 V-columns replaced by 50 "
        "principal components, gives roughly 130 inputs, not the declared 109."
    )
    issues.append("D4: 183 - 89 = 94 is internally consistent.")
    return Check(
        "A-08/A-09", "Declared model-input counts", passed=False, detail=" | ".join(issues)
    )


def check_a16() -> Check:
    """Table 9(a) reference value against the month-level values it reports."""
    mean_of_months = (TABLE_9A_D1["test_m6"] + TABLE_9A_D1["test_m7"]) / 2.0
    reference = TABLE_9A_D1["reference_labelled_train"]
    return Check(
        "A-16",
        "D1 drift reference vs Table 5(a)",
        passed=abs(mean_of_months - reference) < 0.005,
        detail=(
            f"Table 1 assigns months 6-7 to the D1 test partition, so the Table 5(a) "
            f"test AUPRC ({reference}) should equal the combined month-6/7 performance. "
            f"The mean of the Table 9(a) month values is {mean_of_months:.4f}. The gap "
            f"is {abs(reference - mean_of_months) * 100:.2f} pp. Either Table 9(a)'s "
            f"'Train AUPRC' column really is a training-period figure - in which case "
            f"Table 5(a) coincidentally reports the same number - or the two tables "
            f"disagree."
        ),
    )


def check_a17() -> Check:
    """Escalation-path p99 against the memory-bandwidth floor for an 8B model."""
    verdict = check_escalation_plausibility(
        REPORTED_P99_ESCALATION_MS, tokens_generated=MAX_OUTPUT_TOKENS
    )
    short = check_escalation_plausibility(REPORTED_P99_ESCALATION_MS, tokens_generated=60)
    return Check(
        "A-17",
        "Escalation-path latency vs decoding physics",
        passed=bool(verdict["plausible"]),
        detail=(
            f"At the 384-token cap the reported p99 of {REPORTED_P99_ESCALATION_MS} ms "
            f"implies {verdict['implied_ms_per_token']:.3f} ms/token against a "
            f"bandwidth floor of {verdict['bandwidth_floor_ms_per_token']:.2f} ms/token "
            f"for an 8B bfloat16 model on an A100-80GB. Even a short 60-token rationale "
            f"implies {short['implied_ms_per_token']:.2f} ms/token, still below the "
            f"floor. Reproducing the figure requires a smaller generator, speculative "
            f"decoding, or an optimised serving stack - none of which S4.8.2 describes."
        ),
    )


def check_a19() -> Check:
    """TabTransformer D2 diagonal in Table 8(c) vs its Table 5(b) value."""
    table5 = TABLE_5_AUPRC["D2"]["tabtransformer"]
    table8 = TABLE_8C_TABTRANSFORMER_DIAGONAL["D2"]
    return Check(
        "A-19",
        "TabTransformer D2 in-domain AUPRC across tables",
        passed=abs(table5 - table8) < 1e-6,
        detail=(
            f"Table 5(b) reports {table5} for TabTransformer on D2; the Table 8(c) "
            f"diagonal reports {table8} for the same in-domain setting. D1 and D3 "
            f"diagonals agree with Table 5, so D2 is an isolated discrepancy of "
            f"{abs(table8 - table5) * 100:.2f} pp."
        ),
    )


CHECKS = [check_a01, check_a02, check_a04, check_a08_a09, check_a16, check_a17, check_a19]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    args = parser.parse_args()

    results = [check() for check in CHECKS]
    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
    else:
        for result in results:
            status = "PASS" if result.passed else "FAIL"
            print(f"[{status}] {result.finding}  {result.title}")
            print(f"        {result.detail}\n")
        failed = sum(1 for r in results if not r.passed)
        print(f"{failed} of {len(results)} consistency checks FAILED.")
        print("Full analysis: docs/MANUSCRIPT_AUDIT.md")
    # Exit 0 regardless: reporting inconsistencies is this script's job, not a
    # failure of the script.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
