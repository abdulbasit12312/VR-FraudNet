# Manuscript-to-code map

Every major table, figure and experimental claim in the manuscript, mapped to
the code that implements it, the configuration that parameterises it, the data
partition it consumes, the metric implementation, the seeds, and the expected
output file.

**Status key**

| Status | Meaning |
|---|---|
| ✅ implemented | Runs end to end once the dataset is present |
| ⚠️ partial | Runs, but a manuscript-specified input is missing (see the note) |
| ⛔ blocked | Not implemented; the specification is inconsistent or an essential artefact is absent |

Seeds are always the ten from Section 4.8.1: `42, 123, 456, 789, 2024, 3141,
2718, 1001, 1729, 4096` (`vrfraudnet.seeds.MANUSCRIPT_SEEDS`).

---

## Section 3 — data

| Manuscript item | Code | Config | Partition | Output | Status |
|---|---|---|---|---|---|
| Table 1, D1 split | `data/splits.py::chronological_holdout_by_group` | `configs/d1.yaml` `split.*` | months 0–5 / 6–7 | `results/prepared/D1/report.json` | ✅ |
| Table 1, D2 split | `data/splits.py::chronological_fraction_split` | `configs/d2.yaml` | temporal 60/20/20 | `results/prepared/D2/report.json` | ✅ |
| Table 1, D3 split | `data/splits.py::quantile_time_split` | `configs/d3.yaml` | `TransactionDT` q=0.8 | `results/prepared/D3/report.json` | ✅ |
| Table 1 / revised Table 8(d), D3 5-fold expanding CV | `data/splits.py::expanding_window_folds(cut_quantiles=...)` | `configs/d3.yaml` `split.cv_cut_quantiles` | cuts at q = 0.60/0.70/0.80/0.85/0.90, test-to-end | `results/robustness/D3/*` | ⚠️ folds now MANUSCRIPT (L-03 superseded); the fold driver is not wired into `scripts/robustness.py` |
| Table 1, D4 strict-inductive | `data/splits.py::strict_inductive_split` | `configs/d4.yaml` | t 1–34 / 35–49 | `results/prepared/D4/report.json` | ⚠️ validation range is L-04 |
| Table 1, D5 node-arrival | `data/splits.py::node_arrival_split` | `configs/d5.yaml` | labelled nodes only | `results/prepared/D5/report.json` | ⚠️ proportions are L-05 |
| Table 2, D1 preprocessing | `data/baf.py::prepare` | `configs/d1.yaml` `preprocess.*` | train months only | same report | ⚠️ feature count contradiction A-08 |
| Table 2, D2 preprocessing | `data/amlworld.py::prepare` | `configs/d2.yaml` | train segment only | same report | ✅ |
| Table 2, D3 preprocessing | `data/ieee_cis.py::prepare` | `configs/d3.yaml` | train fold only | same report | ⚠️ feature count contradiction A-09 |
| Table 2, D4 preprocessing | `data/elliptic_pp.py::prepare` | `configs/d4.yaml` | train timesteps only | same report | ✅ |
| Table 2, D5 preprocessing | `data/dgraph.py::prepare` | `configs/d5.yaml` | train fold only | same report | ✅ |
| Table 2, leakage controls | `data/preprocess_base.py::TrainOnlyPipeline` | — | all | `leakage_controls` in every result | ✅ |

Command:

```bash
python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml
```

Tests: `tests/test_splits_leakage.py`, `tests/test_preprocess_train_only.py`.

---

## Section 4 — method

| Manuscript item | Code | Config | Status |
|---|---|---|---|
| Eq. 1–2, normalised Laplacian | `models/stage0_spectral.py::normalised_laplacian` | — | ✅ |
| S4.1, K=4 Beta-kernel filters | `models/stage0_spectral.py::beta_kernel_coefficients` | `stage0.n_filters`, `stage0.beta_order` | ⚠️ order is L-08 |
| S4.1, 32-d time embedding | `models/stage0_spectral.py::time_embedding` | `stage0.time_embedding_dim` | ⚠️ parameterisation is L-10 |
| S4.1, 2-layer GELU coefficient MLP | `models/stage0_spectral.py::build_torch_module` | `stage0.mlp_hidden_dim` | ⚠️ width is L-09 |
| S4.1, temporal cutoff | `data/types.py::TemporalGraph.snapshot`, `stage0_spectral.py::assert_no_future_edges` | — | ✅ |
| Eq. 3–4, triage input and probability | `models/stage1_triage.py::TriageClassifier` | `stage1.*` | ✅ |
| S4.2, focal + class-balanced surrogate | `losses/focal.py::focal_objective_factory`, `losses/cost_sensitive.py::class_balanced_weights` | `stage1.focal_gamma` | ⚠️ γ is L-20, β is L-12 |
| Eq. 5, threshold gate | `models/stage1_triage.py::ThresholdGate` | `stage1.threshold_low/high` | ✅ corrected form, A-05 |
| S4.8.1, per-dataset thresholds | `models/stage1_triage.py::MANUSCRIPT_THRESHOLDS` | `configs/d*.yaml` | ✅ |
| S4.3.1, LoRA configuration | `models/stage2_rationale.py::Stage2Config.from_yaml`, `scripts/train_stage2_lora.py` | `configs/stage2_lora.yaml` (canonical) | ⛔ manuscript adapter NOT AVAILABLE (L-31); corpus unspecified (L-25); see `docs/STAGE2_LORA.md` |
| S4.3.1, adapter validation / loading | `scripts/validate_stage2_adapter.py`, `models/stage2_rationale.py::RationaleModel.load` | `configs/stage2_lora.yaml` | ✅ interface; raises when no adapter is supplied |
| S4.3.2, rationale schema | `schemas/rationale_schema.json` | — | ✅ |
| S4.3.3, grammar-masked decoding | `models/grammar.py::RationaleGrammar` | — | ✅ own implementation, L-07 |
| S4.3.4, training-only retrieval | `retrieval/index.py::RetrievalIndex`, `scripts/build_retrieval_index.py` | `retrieval/config/retrieval.yaml` | ✅ exact search (L-33); encoder revision unpinned (L-29); see `docs/RETRIEVAL_SETUP.md` |
| S4.3.5, failure handling | `models/stage2_rationale.py::finalise_generation` | — | ✅ |
| Eq. 6–7, verifier | `verifier/engine.py::DeterministicVerifier` | `verifier/rules/claim_primitives.yaml` | ✅ |
| S4.4, five claim primitives | `verifier/primitives.py` | same rule file | ✅ |
| S4.4, cross-claim consistency | `verifier/engine.py::_check_consistency` (CR1–CR6) | same rule file | ⚠️ risk direction is L-14 |
| Table 4, E1/E2 examples | `examples/rationale_accepted.json`, `examples/rationale_rejected.json` | — | ✅ |
| S4.5, isotonic mixer | `models/stage4_calibration.py::IsotonicMixer` | — | ✅ |
| S4.5, split-conformal | `models/stage4_calibration.py::SplitConformalCalibrator` | `stage4.conformal_alpha` | ✅ |
| S4.5, verifier-aware variant | same class, `verifier_penalty` | `stage4.verifier_penalty` | ⚠️ magnitude is L-16 |
| S4.6.1, cost-sensitive loss | `losses/cost_sensitive.py` | `configs/costs.yaml` | ⛔ costs missing, A-10 |
| S4.6.2, adversarial training | `adversarial/edits.py` (edit functions) | `objectives.adversarial_*`, `adversarial/families/edit_families.yaml` | ⚠️ training-time selection loop not wired into `scripts/train.py` (L-34); injection family blocked, A-06 |
| S4.6.3, counterfactual loss | `losses/counterfactual.py`, `scripts/train_stage2_lora.py::build_counterfactual` | `objectives.counterfactual_weight` | ⚠️ target reconstruction rule is L-32 |
| S4.6.4, ±50% λ sensitivity | `configs/ablations.yaml` `A8` | — | ⚠️ requires Stage 2 |
| S4.7, nine baselines | `baselines/registry.py`, `baselines/neural.py` | `configs/baselines.yaml` | ⚠️ hyperparameters are L-18 |
| S4.8.2, latency protocol | `evaluation/latency.py::measure_pathway` | — | ⚠️ escalation path blocked, A-17 |
| S4.8.3, metric definitions | `evaluation/metrics.py` | — | ✅ |

---

## Section 5 — results

### Table 5(a–c) — primary performance ⚠️

| Field | Value |
|---|---|
| Code | `tables/build_tables.py::table_5_primary_performance` |
| Metrics | `evaluation/metrics.py::compute_all`, `aggregate_seeds` |
| Partition | test partition of D1, D2, D3 |
| Seeds | all ten |
| Output | `results/tables/table_5.md` |
| Blockers | baseline hyperparameters L-18; **D3 Recall@top-1% column withheld (A-01)** |

```bash
for S in 42 123 456 789 2024 3141 2718 1001 1729 4096; do
  for M in logistic_regression isolation_forest mlp cnn1d lstm random_forest \
           xgboost lightgbm tabtransformer vr_fraudnet; do
    python scripts/train.py    --dataset D1 --config configs/d1.yaml --seed $S --model $M
    python scripts/evaluate.py --dataset D1 --config configs/d1.yaml --seed $S --model $M
  done
done
python scripts/make_tables.py --table 5
```

### Table 6(a) — Wilcoxon ✅ (with a reported caveat)

| Field | Value |
|---|---|
| Code | `statistics/wilcoxon.py::pooled_and_per_dataset`, `statistics/holm.py::holm_bonferroni` |
| Input | `results/metrics/**` |
| Output | `results/statistics/wilcoxon_auprc.json` |
| Caveat | pooled and per-dataset both reported (A-04); Table 6(a) means disagree with Table 5 (A-02) |

```bash
python scripts/stats.py --test wilcoxon
```

### Tables 6(b–d) — DeLong ⚠️

| Field | Value |
|---|---|
| Code | `statistics/delong.py::delong_test` |
| Input | `results/predictions/<dataset>/*.json` |
| Output | `results/statistics/delong_<dataset>_<policy>.json` |
| Blocker | seed policy unspecified in the paper (A-03, L-01) |

```bash
python scripts/stats.py --test delong --dataset D1 --seed-policy single_seed --seed 42
```

### Table 7(a–b) — ablations ⛔

A1, A3 and A6 require a trained Stage 2 model (L-25). A2, A4, A5, A7, A8, A9
require checkpoint export. Definitions live in `configs/ablations.yaml`.

### Table 8(a–c) — transfer ⛔

Blocked by A-14: no alignment specified. `scripts/transfer.py` will run under an
explicitly declared alignment, and the result is labelled as this repository's
construction rather than a reproduction.

### Table 9(a) — D1 month drift ⚠️

| Field | Value |
|---|---|
| Code | `evaluation/temporal.py::month_drift` |
| Output | `results/robustness/D1/month_drift_*.json` |
| Blocker | reference partition ambiguous (A-16) |

### Table 9(b) — D3 expanding window ⚠️

| Field | Value |
|---|---|
| Code | `evaluation/temporal.py::expanding_window_summary`, `late_fold_degradation`; folds from `data/splits.py::expanding_window_folds(cut_quantiles=MANUSCRIPT_D3_CUT_QUANTILES)` |
| Blocker | fold definition resolved by revised Table 8(d); the per-fold training driver is not wired into `scripts/robustness.py`, and the full-model rows need Stage 2 (L-25, L-31) |

### Table 9(c) — D4 shock ⚠️

| Field | Value |
|---|---|
| Code | `evaluation/temporal.py::ShockResult`, `split_shock_windows` |
| Output | `results/robustness/D4/shock_*.json` |
| Blocker | shock timestep (L-19) |

```bash
python scripts/robustness.py --dataset D4 --config configs/d4.yaml --seed 42 \
    --protocol shock --shock-timestep 43
```

### Table 10(a–c) — budgeted evasion ⚠️

| Field | Value |
|---|---|
| Code | `adversarial/edits.py`, `adversarial/applicability.py` |
| Output | `results/robustness/<dataset>/evasion_*.json` |
| Blockers | injection family undefined (A-06); split undefined on D1 (A-07) |

### Section 5.7 / Supplementary S1–S4 ⛔

| Supplementary table | Code | Blocker |
|---|---|---|
| S1 cost-sensitive | `evaluation/cost_eval.py` | cost values absent (A-10) |
| S2 subgroup fairness | — | protected-attribute definition (L-22); target values absent (L-27) |
| S3 efficiency and latency | `evaluation/latency.py` | escalation path implausible (A-17) |
| S4 calibration and safeguards | `models/stage4_calibration.py`, `models/pipeline.py::escalation_statistics` | requires Stage 2 (L-25) |

---

## Figures

| Figure | Content | Status |
|---|---|---|
| Figure 1 | Problem setting and property table | ⛔ contradicts the text on "certified bounds" (A-20) |
| Figure 2 | Inference pipeline | ✅ implemented as `models/pipeline.py::VRFraudNet`; routing proportions questioned (A-21) |
| Figure 3 | Stage 0 encoder | ✅ `models/stage0_spectral.py` |
| Figure 4 | Stage 1 triage and gate | ✅ `models/stage1_triage.py` |

---

## Reproducibility summary

| Manuscript artefact | Reproducible from the paper alone? |
|---|---|
| Table 5 | No — L-18 (baselines), A-01 (D3 metric) |
| Table 6(a) | Structure yes, values no — A-02 |
| Tables 6(b–d) | No — A-03, L-01 |
| Table 7 | No — L-25 |
| Table 8 | No — A-14 |
| Table 9(a) | Structure yes — A-16 must be resolved |
| Table 9(b) | Approximately — L-03 |
| Table 9(c) | Approximately — L-19 |
| Table 10 | Partially — A-06, A-07 |
| Supplementary S1–S4 | No — L-26, L-27 |
