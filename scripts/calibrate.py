"""Split-conformal calibration and coverage evaluation (manuscript S4.5, S4.8.1).

Usage
-----
    python scripts/calibrate.py --dataset D1 --config configs/d1.yaml --seed 42
    python scripts/calibrate.py --dataset D1 --config configs/d1.yaml --seed 42 \
        --alpha 0.10

Calibrates on the held-out calibration half of the validation segment and
evaluates empirical coverage on the test partition. Reports both the primary
alpha = 0.05 and, when requested, the secondary alpha = 0.10 sensitivity setting.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from _common import add_common_arguments, bootstrap, make_manifest, result_path

from vrfraudnet.io_utils import read_result, require_file, write_result
from vrfraudnet.models.stage4_calibration import SplitConformalCalibrator

log = logging.getLogger("calibrate")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", default="vr_fraudnet")
    parser.add_argument("--alpha", type=float, default=None,
                        help="significance level; defaults to the config's primary value")
    args = parser.parse_args()

    config, _ = bootstrap(args)
    alpha = float(args.alpha if args.alpha is not None else config.require("stage4.conformal_alpha"))

    predictions_path = require_file(
        result_path(args, "predictions", args.dataset, f"{args.model}_seed{args.seed}.json"),
        produced_by=(
            f"python scripts/train.py --dataset {args.dataset} --config {args.config} "
            f"--seed {args.seed} --model {args.model}"
        ),
    )
    payload = read_result(predictions_path)["results"]

    # The calibration partition is the second chronological half of validation
    # (manuscript S4.8.1). scripts/train.py stores validation scores for the
    # model-selection half; the calibration half is re-derived here from the
    # stored validation block when train.py recorded it, and otherwise the
    # validation block itself is split, which is stated in the output.
    valid_scores = np.asarray(payload["validation"]["scores"], dtype=float)
    valid_labels = np.asarray(payload["validation"]["labels"], dtype=int)
    half = valid_scores.size // 2
    calib_scores, calib_labels = valid_scores[half:], valid_labels[half:]
    calibration_source = "second chronological half of the stored validation block"

    if calib_labels.sum() == 0:
        log.error(
            "the calibration partition contains no positive cases; conformal "
            "calibration on it would be meaningless"
        )
        return 1

    calibrator = SplitConformalCalibrator(
        alpha=alpha, verifier_penalty=float(config.get("stage4.verifier_penalty", 0.10))
    )
    calibration = calibrator.calibrate(calib_scores, calib_labels)

    test_scores = np.asarray(payload["test"]["scores"], dtype=float)
    test_labels = np.asarray(payload["test"]["labels"], dtype=int)
    coverage = calibrator.evaluate_coverage(test_scores, test_labels)

    out = {
        "model": args.model,
        "dataset": args.dataset,
        "alpha": alpha,
        "calibration": calibration.to_dict(),
        "test_coverage": coverage,
        "calibration_source": calibration_source,
        "scope_note": (
            "The marginal-coverage statement applies to the split-conformal layer only, "
            "under exchangeability of calibration and test data. Under the chronological "
            "splits used throughout this study, calibration and test data come from "
            "DIFFERENT time periods, so exchangeability is an approximation, not a "
            "property of the design. Empirical coverage is therefore the operative "
            "number, and the formal guarantee should not be quoted without this caveat."
        ),
    }
    path = result_path(args, "calibration", args.dataset,
                       f"{args.model}_seed{args.seed}_alpha{alpha}.json")
    write_result(
        path,
        make_manifest(args, experiment="calibrate", model=args.model, partition="calibration+test"),
        out,
        overwrite=args.overwrite,
    )
    log.info(
        "alpha=%.2f | nominal coverage %.3f | empirical test coverage %.4f | mean set size %.3f",
        alpha, 1 - alpha, coverage["empirical_coverage"], coverage["mean_prediction_set_size"],
    )
    log.info("wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
