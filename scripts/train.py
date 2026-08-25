"""Train VR-FraudNet or a baseline on one dataset with one seed.

Usage
-----
    python scripts/train.py --dataset D1 --config configs/d1.yaml --seed 42
    python scripts/train.py --dataset D2 --config configs/d2.yaml --seed 42 \
        --model lightgbm

What this script trains
-----------------------
* Stage 0 spectral filter bank (graph datasets only), snapshot-restricted so no
  future edge is visible.
* Stage 1 LightGBM triage with the focal, class-balanced objective.
* Routing thresholds on the model-selection half of the validation segment.
* Stage 4 isotonic mixer on the same half.
* Split-conformal calibration on the held-out calibration half.

What this script does NOT train
-------------------------------
Stage 2. Fine-tuning Meta-Llama-3.1-8B-Instruct requires the base weights, a
GPU, and a rationale training corpus that must be constructed from the
review-band transactions of each dataset. ``--stage stage2`` prints the exact
requirements and exits non-zero rather than pretending to train.

Outputs
-------
``results/predictions/<dataset>/<model>_seed<seed>.json`` containing validation
and test scores plus the frozen operating threshold. Every downstream analysis
consumes these files; nothing downstream recomputes a model.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from _common import add_common_arguments, bootstrap, make_manifest, result_path

from vrfraudnet.baselines.registry import BASELINE_ORDER, build_baseline, score_of
from vrfraudnet.data import load_prepared
from vrfraudnet.evaluation.metrics import select_operating_threshold
from vrfraudnet.io_utils import write_result
from vrfraudnet.models.stage0_spectral import Stage0Config, precompute_filter_bank
from vrfraudnet.models.stage1_triage import (
    ThresholdGate,
    TriageClassifier,
    TriageParams,
    select_thresholds,
)
from vrfraudnet.models.stage4_calibration import (
    IsotonicMixer,
    MixerInputs,
    SplitConformalCalibrator,
)

log = logging.getLogger("train")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser)
    parser.add_argument("--model", default="vr_fraudnet",
                        choices=("vr_fraudnet", *BASELINE_ORDER))
    parser.add_argument("--stage", default="all",
                        choices=("all", "stage0", "stage1", "stage2", "stage4"))
    parser.add_argument("--use-manuscript-thresholds", action="store_true",
                        help="use the per-dataset (T_low, T_high) pairs from S4.8.1 "
                             "instead of reselecting them on validation data")
    args = parser.parse_args()

    if args.stage == "stage2":
        return _stage2_requirements()

    config, data_root = bootstrap(args)
    prepared = load_prepared(args.dataset, config, root=data_root)
    split = prepared.split

    features = prepared.features.to_numpy(dtype=np.float64)
    labels = prepared.labels

    if config.get("stage0.enabled", False) and prepared.graph is not None:
        features = _append_graph_embedding(features, prepared, config)

    x_train, y_train = features[split.train], labels[split.train]
    x_valid, y_valid = features[split.validation], labels[split.validation]
    x_test, y_test = features[split.test], labels[split.test]

    if args.model == "vr_fraudnet":
        scores_valid, scores_test, extra = _train_vr_fraudnet(
            args, config, x_train, y_train, x_valid, y_valid, x_test, prepared
        )
    else:
        scores_valid, scores_test, extra = _train_baseline(
            args, x_train, y_train, x_valid, y_valid, x_test
        )

    threshold = select_operating_threshold(y_valid, scores_valid)
    log.info("frozen operating threshold (validation fraud-F1 maximiser): %.6f", threshold)

    payload = {
        "model": args.model,
        "dataset": args.dataset,
        "operating_threshold": float(threshold),
        "validation": {"scores": scores_valid.tolist(), "labels": y_valid.tolist()},
        "test": {"scores": scores_test.tolist(), "labels": y_test.tolist()},
        "split_sizes": split.sizes,
        "leakage_controls": prepared.leakage.to_dict(),
        **extra,
    }
    path = result_path(
        args, "predictions", args.dataset, f"{args.model}_seed{args.seed}.json"
    )
    write_result(
        path,
        make_manifest(
            args,
            experiment="train",
            model=args.model,
            partition="validation+test",
            notes="scores are continuous fraud scores; fraud is the positive class",
        ),
        payload,
        overwrite=args.overwrite,
    )
    log.info("wrote %s", path)
    return 0


def _train_vr_fraudnet(args, config, x_train, y_train, x_valid, y_valid, x_test, prepared):
    """Stage 1 + routing + Stage 4 for the full model."""
    params = TriageParams(
        n_estimators=int(config.require("stage1.n_estimators")),
        max_depth=int(config.require("stage1.max_depth")),
        num_leaves=int(config.require("stage1.num_leaves")),
        learning_rate=float(config.require("stage1.learning_rate")),
        min_child_samples=int(config.require("stage1.min_child_samples")),
        feature_fraction=float(config.require("stage1.feature_fraction")),
        bagging_fraction=float(config.require("stage1.bagging_fraction")),
        bagging_freq=int(config.require("stage1.bagging_freq")),
        reg_alpha=float(config.require("stage1.reg_alpha")),
        reg_lambda=float(config.require("stage1.reg_lambda")),
        early_stopping_rounds=int(config.require("stage1.early_stopping_rounds")),
        num_threads=int(config.get("stage1.num_threads", 0)),
    )
    triage = TriageClassifier(
        params,
        seed=args.seed,
        focal_gamma=float(config.get("stage1.focal_gamma", 2.0)),
        cost_weight=float(config.get("objectives.cost_sensitive_weight", 0.50)),
        class_balanced_beta=float(config.get("stage1.class_balanced_beta", 0.9999)),
    )
    triage.fit(x_train, y_train, x_valid, y_valid)

    p_valid = triage.predict_proba(x_valid)
    p_test = triage.predict_proba(x_test)

    if args.use_manuscript_thresholds:
        gate = ThresholdGate(
            float(config.require("stage1.threshold_low")),
            float(config.require("stage1.threshold_high")),
        )
        threshold_source = "manuscript S4.8.1"
    else:
        gate = select_thresholds(
            p_valid,
            y_valid,
            max_escalation_rate=float(config.get("stage1.max_escalation_rate", 0.05)),
        )
        threshold_source = "reselected on the validation partition"
    log.info(
        "routing gate T_low=%.3f T_high=%.3f (%s); validation escalation rate %.4f",
        gate.t_low, gate.t_high, threshold_source, gate.escalation_rate(p_valid),
    )

    # Stage 4. Without a Stage 2 adapter there is no rationale-side probability
    # and no verifier status, so the mixer degenerates to a monotone calibration
    # of the triage score. That is stated in the result file rather than hidden.
    mixer = IsotonicMixer().fit(
        MixerInputs(p_valid, None, None), y_valid, partition="validation"
    )
    calibrated_valid = mixer.predict(MixerInputs(p_valid, None, None))
    calibrated_test = mixer.predict(MixerInputs(p_test, None, None))

    conformal_summary = None
    calibration_idx = prepared.split.calibration
    if calibration_idx is not None and calibration_idx.size >= 20:
        features_all = prepared.features.to_numpy(dtype=np.float64)
        p_calib = triage.predict_proba(features_all[calibration_idx])
        calibrated_calib = mixer.predict(MixerInputs(p_calib, None, None))
        calibrator = SplitConformalCalibrator(
            alpha=float(config.get("stage4.conformal_alpha", 0.05)),
            verifier_penalty=float(config.get("stage4.verifier_penalty", 0.10)),
        )
        conformal_summary = calibrator.calibrate(
            calibrated_calib, prepared.labels[calibration_idx]
        ).to_dict()

    extra = {
        "routing": {
            "t_low": gate.t_low,
            "t_high": gate.t_high,
            "source": threshold_source,
            "validation_escalation_rate": gate.escalation_rate(p_valid),
            "test_escalation_rate": gate.escalation_rate(p_test),
        },
        "stage2_active": False,
        "stage2_note": (
            "no LoRA adapter supplied, so no rationale-side probability and no "
            "verifier status entered the Stage 4 mixer. Ablation A6 (w/o Large LLM) "
            "is therefore the configuration actually trained here."
        ),
        "conformal": conformal_summary,
    }
    return calibrated_valid, calibrated_test, extra


def _train_baseline(args, x_train, y_train, x_valid, y_valid, x_test):
    """Fit one manuscript baseline under identical splits and preprocessing."""
    model = build_baseline(args.model, seed=args.seed)
    if args.model == "isolation_forest":
        # Unsupervised: fitted on the training partition without labels.
        model.fit(x_train)
    elif hasattr(model, "fit") and args.model in {"mlp", "cnn1d", "lstm", "tabtransformer"}:
        model.fit(x_train, y_train, x_valid=x_valid, y_valid=y_valid)
    else:
        model.fit(x_train, y_train)

    return (
        score_of(model, x_valid),
        score_of(model, x_test),
        {
            "baseline_hyperparameters_provenance": (
                "ASSUMPTION - the manuscript publishes no baseline hyperparameters "
                "(docs/KNOWN_LIMITATIONS.md L-18)"
            )
        },
    )


def _append_graph_embedding(features, prepared, config):
    """Concatenate the Stage 0 representation, snapshot-restricted per timestep."""
    stage0 = Stage0Config(
        n_filters=int(config.get("stage0.n_filters", 4)),
        beta_order=int(config.get("stage0.beta_order", 4)),
        output_dim=int(config.get("stage0.output_dim", 64)),
    )
    graph = prepared.graph
    # Restrict to edges observable at or before the latest TRAINING time, so no
    # future edge can influence a training-time embedding.
    train_cutoff = float(np.max(prepared.times[prepared.split.train]))
    snapshot = graph.snapshot(train_cutoff)
    log.info(
        "Stage 0: %d of %d edges observable at the training cutoff t=%.1f",
        snapshot.edge_index.shape[1], graph.edge_index.shape[1], train_cutoff,
    )
    node_features = np.zeros((graph.num_nodes, features.shape[1]), dtype=np.float64)
    node_index = prepared.node_index
    if node_index is None:
        node_index = np.arange(min(graph.num_nodes, features.shape[0]))
    node_features[node_index[: features.shape[0]]] = features[: node_index.size]

    bank = precompute_filter_bank(
        snapshot.edge_index, graph.num_nodes, node_features, stage0
    )
    # Without a trained encoder the filter bank is summarised by its per-filter
    # mean, which is a deterministic, honest placeholder for z_g. Training the
    # learned mixing coefficients requires the torch module in stage0_spectral.
    pooled = bank.mean(axis=2).T[node_index[: features.shape[0]]]
    return np.hstack([features, pooled])


def _stage2_requirements() -> int:
    print(
        "Stage 2 training is not performed by this script.\n\n"
        "Requirements the manuscript implies (S4.3.1):\n"
        "  1. Access to meta-llama/Meta-Llama-3.1-8B-Instruct base weights.\n"
        "  2. A GPU with >= 40 GB memory for bfloat16 LoRA fine-tuning.\n"
        "  3. A rationale training corpus built from the REVIEW-BAND transactions of\n"
        "     the training partition, with target rationales constructed from evidence\n"
        "     observable at each transaction's prediction time, and with malformed,\n"
        "     inconsistent or verifier-rejected targets removed.\n\n"
        "The manuscript does not describe how the target rationales were authored -\n"
        "whether by rule, by a teacher model, or by human annotation. That is the\n"
        "single largest gap in the reproducibility package (docs/KNOWN_LIMITATIONS.md\n"
        "L-25, REQUIRES AUTHOR CONFIRMATION).\n\n"
        "This repository will not ship or generate a fabricated adapter."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
