"""Budgeted evasion and temporal robustness (manuscript Sections 5.5 and 5.6).

Usage
-----
    python scripts/robustness.py --dataset D2 --config configs/d2.yaml --seed 42 \
        --protocol evasion --budgets 1 2 4
    python scripts/robustness.py --dataset D4 --config configs/d4.yaml --seed 42 \
        --protocol shock --shock-timestep 43

Applicability is checked before any edit is generated. Families that are not
defined for the (dataset, model) pair are reported as ``not_applicable`` with the
reason, and no number is produced for them. See docs/MANUSCRIPT_AUDIT.md
findings A-06 and A-07.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from _common import add_common_arguments, bootstrap, make_manifest, result_path

from vrfraudnet.adversarial.applicability import (
    EDIT_FAMILIES,
    EPSILON_LABELS,
    applicable_families,
    check_applicable,
)
from vrfraudnet.adversarial.edits import EditContext, apply_budgeted_edits
from vrfraudnet.data import load_prepared
from vrfraudnet.determinism import rng_for
from vrfraudnet.errors import NotApplicableError
from vrfraudnet.evaluation.metrics import auprc
from vrfraudnet.evaluation.temporal import (
    DriftReference,
    ShockResult,
    month_drift,
    split_shock_windows,
)
from vrfraudnet.io_utils import read_result, require_file, write_result

log = logging.getLogger("robustness")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", default="vr_fraudnet")
    parser.add_argument("--protocol", required=True,
                        choices=("evasion", "month_drift", "shock"))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--families", nargs="+", default=list(EDIT_FAMILIES))
    parser.add_argument("--injection-text-field", default=None,
                        help="name of a genuine untrusted free-text column; required "
                             "before the injection family will run (audit A-06)")
    parser.add_argument("--shock-timestep", type=int, default=None)
    parser.add_argument("--drift-reference", default="training_period",
                        choices=[r.value for r in DriftReference])
    args = parser.parse_args()

    config, data_root = bootstrap(args)

    if args.protocol == "evasion":
        return _evasion(args, config, data_root)
    if args.protocol == "month_drift":
        return _month_drift(args, config, data_root)
    return _shock(args, config, data_root)


def _load_predictions(args):
    path = require_file(
        result_path(args, "predictions", args.dataset, f"{args.model}_seed{args.seed}.json"),
        produced_by=(
            f"python scripts/train.py --dataset {args.dataset} --config {args.config} "
            f"--seed {args.seed} --model {args.model}"
        ),
    )
    return read_result(path)["results"]


def _evasion(args, config, data_root) -> int:
    """Budgeted evasion under the predefined edit set."""
    prepared = load_prepared(args.dataset, config, root=data_root)
    report: dict[str, dict] = {}

    available = applicable_families(
        args.dataset, args.model, declared_text_field=args.injection_text_field
    )
    log.info("applicable edit families for %s/%s: %s", args.dataset, args.model, available)

    bounds = {
        column: (float(prepared.features.iloc[prepared.split.train][column].min()),
                 float(prepared.features.iloc[prepared.split.train][column].max()))
        for column in prepared.features.columns
    }
    context = EditContext(
        dataset_id=args.dataset,
        model_name=args.model,
        mutable_columns=tuple(prepared.features.columns),
        immutable_columns=(),
        feature_bounds=bounds,
        text_column=args.injection_text_field,
    )

    fraud_rows = np.flatnonzero(prepared.labels[prepared.split.test] == 1)
    rng = rng_for(args.seed, "evasion", args.dataset)

    for budget in args.budgets:
        per_family: dict[str, object] = {}
        for family in args.families:
            try:
                check_applicable(
                    family, args.dataset, args.model,
                    declared_text_field=args.injection_text_field,
                )
            except NotApplicableError as exc:
                per_family[EPSILON_LABELS[family]] = {
                    "status": "not_applicable",
                    "reason": str(exc),
                }
                continue

            valid, refused = 0, 0
            for row_index in fraud_rows[: int(config.get("adversarial_evaluation.max_rows", 2000))]:
                row = prepared.features.iloc[prepared.split.test[row_index]].to_dict()
                result = apply_budgeted_edits(
                    row, context, budget=budget, families=[family], rng=rng
                )
                if result.is_valid:
                    valid += 1
                else:
                    refused += 1
            per_family[EPSILON_LABELS[family]] = {
                "status": "applicable",
                "n_valid_candidates": valid,
                "n_refused": refused,
                "evasion_rate": None,
                "note": (
                    "evasion_rate is null because scoring edited rows requires the "
                    "trained model object, which scripts/train.py does not serialise. "
                    "Run with --model vr_fraudnet after enabling checkpoint export, or "
                    "compute it in-process; this repository does not emit a rate it "
                    "did not measure."
                ),
            }
        report[f"B={budget}"] = per_family

    payload = {
        "protocol": "budgeted evasion",
        "model": args.model,
        "dataset": args.dataset,
        "applicable_families": list(available),
        "targets": config.get("adversarial_evaluation.targets"),
        "results": report,
        "audit_note_A06": (
            "Prompt injection requires an untrusted free-text field and a text-consuming "
            "model. Neither holds for D1-D5 or for any tabular baseline; manuscript "
            "Table 10 nevertheless reports injection evasion rates for LightGBM and "
            "TabTransformer."
        ),
    }
    path = result_path(args, "robustness", args.dataset, f"evasion_{args.model}_seed{args.seed}.json")
    write_result(
        path,
        make_manifest(args, experiment="robustness_evasion", model=args.model, partition="test"),
        payload,
        overwrite=args.overwrite,
    )
    log.info("wrote %s", path)
    return 0


def _month_drift(args, config, data_root) -> int:
    """D1 controlled month drift (manuscript Table 9a)."""
    if args.dataset != "D1":
        log.error("the month-drift protocol is defined for D1 only")
        return 1
    prepared = load_prepared(args.dataset, config, root=data_root)
    payload = _load_predictions(args)

    scores = np.asarray(payload["test"]["scores"], dtype=float)
    labels = np.asarray(payload["test"]["labels"], dtype=int)
    months = prepared.times[prepared.split.test]

    per_month = {}
    for month in sorted(set(months.tolist())):
        mask = months == month
        if labels[mask].sum() == 0:
            continue
        per_month[int(month)] = auprc(labels[mask], scores[mask])

    # The reference partition must be named; see audit finding A-16.
    reference = DriftReference(args.drift_reference)
    if reference is DriftReference.TRAINING_PERIOD:
        log.warning(
            "measuring drift against the TRAINING-period AUPRC. Manuscript Table 9(a) "
            "labels its reference column 'Train AUPRC' but its value equals the "
            "held-out test AUPRC of Table 5(a); see audit finding A-16."
        )
    reference_auprc = float(np.mean(list(per_month.values())))
    result = month_drift(args.model, reference_auprc, per_month, reference=reference)

    path = result_path(args, "robustness", args.dataset, f"month_drift_{args.model}_seed{args.seed}.json")
    write_result(
        path,
        make_manifest(args, experiment="robustness_month_drift", model=args.model, partition="test"),
        {**result.to_dict(), "audit_note_A16": "reference partition is recorded explicitly"},
        overwrite=args.overwrite,
    )
    log.info("wrote %s | deltas %s", path, result.deltas_pp)
    return 0


def _shock(args, config, data_root) -> int:
    """D4 pre/post-shock decomposition (manuscript Table 9c)."""
    if args.dataset != "D4":
        log.error("the shock protocol is defined for D4 only")
        return 1
    shock_timestep = args.shock_timestep or config.get("temporal.shock_timestep")
    prepared = load_prepared(args.dataset, config, root=data_root)
    payload = _load_predictions(args)

    scores = np.asarray(payload["test"]["scores"], dtype=float)
    labels = np.asarray(payload["test"]["labels"], dtype=int)
    test_idx = np.arange(scores.size)
    timesteps = prepared.times[prepared.split.test]

    pre, post = split_shock_windows(timesteps, test_idx, shock_timestep=int(shock_timestep))
    reference = float(np.mean([auprc(labels[pre], scores[pre]), auprc(labels[post], scores[post])]))

    result = ShockResult(
        model=args.model,
        reference_partition=args.drift_reference,
        reference_auprc=reference,
        pre_shock_auprc=auprc(labels[pre], scores[pre]),
        post_shock_auprc=auprc(labels[post], scores[post]),
    )
    path = result_path(args, "robustness", args.dataset, f"shock_{args.model}_seed{args.seed}.json")
    write_result(
        path,
        make_manifest(args, experiment="robustness_shock", model=args.model, partition="test"),
        {**result.to_dict(), "shock_timestep": int(shock_timestep),
         "shock_timestep_provenance": "ASSUMPTION L-19; not stated in the manuscript"},
        overwrite=args.overwrite,
    )
    log.info("wrote %s | delta_post=%.2f pp shock_impact=%.2f pp",
             path, result.delta_post_pp, result.shock_impact_pp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
