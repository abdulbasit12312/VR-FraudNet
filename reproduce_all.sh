#!/usr/bin/env bash
#
# Master reproducibility driver.
#
#   bash reproduce_all.sh              print the plan without running anything
#   bash reproduce_all.sh --execute    actually run it
#
# The default is a dry run on purpose: the full sweep is on the order of several
# hundred GPU-hours (see docs/REPRODUCIBILITY.md, section 5). Read the plan, run
# one seed end to end, and only then launch the sweep.
#
# Steps that the manuscript does not permit to be reproduced are printed with a
# BLOCKED marker and the audit finding that blocks them. They are not executed,
# and no substitute number is produced for them.

set -euo pipefail

EXECUTE=0
[[ "${1:-}" == "--execute" ]] && EXECUTE=1

SEEDS=(42 123 456 789 2024 3141 2718 1001 1729 4096)
PRIMARY=(D1 D2 D3)
STRESS=(D4 D5)
MODELS=(logistic_regression isolation_forest mlp cnn1d lstm random_forest \
        xgboost lightgbm tabtransformer vr_fraudnet)

run() {
  if [[ $EXECUTE -eq 1 ]]; then
    echo "+ $*"
    "$@"
  else
    echo "    $*"
  fi
}

section() { printf '\n=== %s ===\n' "$1"; }
blocked() { printf '    [BLOCKED: %s] %s\n' "$1" "$2"; }

section "0. Environment and integrity checks"
run python -m pytest -q
run python scripts/audit_manuscript.py

section "1. Dataset preparation (manuscript Table 1, Table 2)"
for D in "${PRIMARY[@]}" "${STRESS[@]}"; do
  run python scripts/prepare_data.py --dataset "$D" --config "configs/${D,,}.yaml"
done

section "2. Training and evaluation -> Table 5 (a-c)"
echo "    NOTE: baseline hyperparameters are ASSUMPTIONS (L-18). Baseline numbers"
echo "          produced here are NOT comparable to manuscript Table 5."
for D in "${PRIMARY[@]}"; do
  for S in "${SEEDS[@]}"; do
    for M in "${MODELS[@]}"; do
      run python scripts/train.py    --dataset "$D" --config "configs/${D,,}.yaml" --seed "$S" --model "$M"
      run python scripts/evaluate.py --dataset "$D" --config "configs/${D,,}.yaml" --seed "$S" --model "$M"
    done
  done
done

section "3. Calibration -> Supplementary Table S4 (coverage rows)"
for D in "${PRIMARY[@]}"; do
  for S in "${SEEDS[@]}"; do
    run python scripts/calibrate.py --dataset "$D" --config "configs/${D,,}.yaml" --seed "$S" --alpha 0.05
    run python scripts/calibrate.py --dataset "$D" --config "configs/${D,,}.yaml" --seed "$S" --alpha 0.10
  done
done

section "4. Statistical testing -> Table 6"
run python scripts/stats.py --test wilcoxon
echo "    Table 6(a) note: pooled AND per-dataset tests are reported (audit A-04)."
for D in "${PRIMARY[@]}"; do
  echo "    Tables 6(b-d): the seed policy must be declared explicitly (audit A-03)."
  run python scripts/stats.py --test delong --dataset "$D" --seed-policy single_seed --seed 42
done

section "5. Temporal robustness -> Table 9"
for S in "${SEEDS[@]}"; do
  run python scripts/robustness.py --dataset D1 --config configs/d1.yaml --seed "$S" --protocol month_drift
  run python scripts/robustness.py --dataset D4 --config configs/d4.yaml --seed "$S" --protocol shock --shock-timestep 43
done
echo "    Table 9(a) note: the drift reference partition is ambiguous (audit A-16)."
echo "    Table 9(c) note: the shock timestep is an assumption (L-19)."

section "6. Budgeted evasion -> Table 10"
for D in D2 D3 D4; do
  for S in "${SEEDS[@]}"; do
    run python scripts/robustness.py --dataset "$D" --config "configs/${D,,}.yaml" --seed "$S" \
        --protocol evasion --budgets 1 2 4
  done
done
blocked "A-06" "Table 10 'eps4 inject' column: no benchmark has an untrusted text field, and no tabular baseline consumes text."
blocked "A-07" "Table 10 'eps1 split' on D1: BAF rows are account applications, not payments."

section "7. Cross-dataset transfer -> Table 8"
blocked "A-14" "Table 8 as published: no feature-space alignment is specified."
echo "    To produce an interpretable transfer matrix under a DECLARED alignment:"
for SRC in "${PRIMARY[@]}" "${STRESS[@]}"; do
  for TGT in "${PRIMARY[@]}" "${STRESS[@]}"; do
    echo "    python scripts/transfer.py --source $SRC --target $TGT \\"
    echo "        --source-config configs/${SRC,,}.yaml --target-config configs/${TGT,,}.yaml \\"
    echo "        --alignment rank_projection --seed 42"
  done
done

section "8. Latency -> Supplementary Table S3"
run python scripts/latency_bench.py --dataset D1 --config configs/d1.yaml --seed 42 --pathway common
blocked "A-17" "escalation-path p99: the published 175.23 ms is below the decoding bandwidth floor for an 8B model."

section "9. Component ablations -> Table 7"
blocked "L-25" "A1/A3/A6 require a trained Stage 2 rationale model; the training corpus is undescribed."
echo "    A2/A4/A5/A7/A8/A9 require checkpoint export; see configs/ablations.yaml."

section "10. Cost-sensitive evaluation -> Supplementary Table S1"
blocked "A-10" "the FN/FP/TP cost values are absent from the main text; configs/costs.yaml ships with nulls."

section "11. Table generation"
run python scripts/make_tables.py --table all

printf '\n'
if [[ $EXECUTE -eq 1 ]]; then
  echo "Done. Tables in results/tables/, raw results in results/."
else
  echo "Dry run complete. Re-run with --execute to perform these steps."
fi
echo "Before reporting anything, read docs/MANUSCRIPT_AUDIT.md and docs/KNOWN_LIMITATIONS.md."
