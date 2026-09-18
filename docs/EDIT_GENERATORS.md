# Edit generators (adversarial edit families)

The structured fraud-edit families of manuscript Sections 4.6.2 and 5.6: where
each is implemented, what it consumes, what it may never change, on which
datasets and models it is defined, how randomness is seeded, and how it is
tested. Applicability restrictions are stated here as they are enforced in
code; none is relaxed to make an experiment appear runnable.

| Item | Where |
|---|---|
| Declarative registry (families, invariants, applicability by dataset, training settings) | [`adversarial/families/edit_families.yaml`](../adversarial/families/edit_families.yaml) |
| Applicability rules (dataset capabilities, text-consuming models, `NotApplicableError`) | [`src/vrfraudnet/adversarial/applicability.py`](../src/vrfraudnet/adversarial/applicability.py) |
| Edit implementations, budgeted composition, evasion rate | [`src/vrfraudnet/adversarial/edits.py`](../src/vrfraudnet/adversarial/edits.py) |
| Driver | [`scripts/robustness.py --protocol evasion`](../scripts/robustness.py) |
| Tests | [`tests/test_adversarial_edits.py`](../tests/test_adversarial_edits.py), [`tests/test_reviewer_artifacts.py`](../tests/test_reviewer_artifacts.py) (registry ↔ implementation ↔ documentation mapping) |

## Common contract

* **Input**: one transaction row (`Mapping[str, Any]`), an `EditContext`
  (dataset id, model name, mutable and immutable column sets, training-derived
  feature bounds, and the optional amount / timestamp / entity / text
  columns), and a `numpy.random.Generator`.
* **Output**: `EditResult(family, applied, rows, reason)`. `applied=False`
  with a `reason` is an explicit refusal, never a silent no-op; the evasion
  rate excludes refused rows from numerator and denominator.
* **Immutable under every family** (S4.6.2): fraud label, protected
  attributes, immutable identifiers, dataset-split assignment, outcome-defining
  fields. `EditContext.assert_mutable` raises `ConfigurationError` on any
  attempt.
* **Randomness**: every edit takes an explicit generator; `scripts/robustness.py`
  derives it as `rng_for(seed, "evasion", dataset)` from the run seed via the
  SHA-256 sub-seed scheme in `vrfraudnet.determinism`, so a given (seed,
  dataset) yields the same edits on every platform. No global RNG is used.
* **Budgets** (S5.6): `apply_budgeted_edits(row, context, budget, families, rng)`
  applies `budget` edits **from the named family** in sequence (a per-family
  column at B = 2 means two edits of that family); mixed-family composition is
  available by passing several families. The manuscript does not say which
  reading it used (audit A-13).
* **Applicability**: `check_applicable(family, dataset, model, declared_text_field)`
  is called at the top of every edit function and raises `NotApplicableError`
  with the reason. `applicable_families(dataset, model)` enumerates what is
  defined for a pair; `tests/test_adversarial_edits.py::test_applicable_families_matrix_is_stable`
  pins the matrix.

## Revised manuscript scope

The revised manuscript evaluates the **predictive path** (LightGBM,
TabTransformer, the ablations and the full model) under four structured
families — split, delay, mule, feat — and states that prompt injection "is
not included as a predictive-path edit because the structured predictive
models ... do not consume free-text input"; injection is evaluated separately
through the Stage 2 / Stage 3 pathway. `configs/base.yaml
adversarial_evaluation.predictive_path_families` and
`edit_families.yaml predictive_path_families` record this. The five-family
registry is kept because the injection generator exists and is gated, not
because it can run on these benchmarks (see ε₄ below).

---

## ε₁ split — transaction splitting

| | |
|---|---|
| Implementation | `edits.py::split_edit` |
| Purpose | divide one payment into two records that together carry the original amount (S4.6.2) |
| Inputs | `context.amount_column`; the row's amount must be positive |
| Tunable | none beyond the budget; the split fraction is drawn from U(0.30, 0.70) |
| Invariant | the two parts sum to the original total to floating-point tolerance; a violation returns `applied=False` |
| Applicable | D2, D3 |
| Not applicable | **D1** (BAF rows are account applications with no transferred amount — audit A-07), D4 (no transaction amount in the feature set), D5 (nodes are users, not payments) |
| Randomness | the fraction, from the supplied generator |
| Tests | `test_split_preserves_the_total_amount`, `test_split_is_not_applicable_to_d1` |

## ε₂ delay — execution delay

| | |
|---|---|
| Implementation | `edits.py::delay_edit` |
| Purpose | postpone execution while preserving chronological validity (S4.6.2) |
| Inputs | `context.timestamp_column` |
| Tunable | `EditContext.max_delay_seconds`, default 86 400 s — **ASSUMPTION L-17**, the manuscript gives no maximum |
| Invariant | time never moves backwards; a backdated result is refused |
| Applicable | D2, D3, D4, D5 |
| Not applicable | **D1** (month granularity only; a delay inside a month changes nothing the model sees, and a delay across months changes the partition) |
| Randomness | the delay, from U(0, max_delay) |
| Tests | `test_delay_never_moves_time_backwards` |

## ε₃ mule — mule-routing modification

| | |
|---|---|
| Implementation | `edits.py::mule_edit` |
| Purpose | reroute through an intermediary entity (S4.6.2) |
| Inputs | `context.entity_columns`, `context.admissible_entities` — the pool **must** come from the training graph |
| Tunable | none |
| Invariant | the substituted entity is drawn from the training-graph pool and differs from the original; without a pool the edit refuses (`test_mule_edit_refuses_without_a_training_entity_pool`) |
| Applicable | D2, D4, D5 |
| Not applicable | **D1, D3** (no entity graph, so admissible entity and edge types cannot be drawn from a training graph) |
| Randomness | choice of entity column and of replacement entity |
| Tests | `test_mule_edit_uses_only_admissible_entities`, `test_mule_is_not_applicable_without_a_graph` |

## ε₄ inject — prompt injection

| | |
|---|---|
| Implementation | `edits.py::inject_edit` (gated) |
| Purpose | place a payload in an untrusted free-text field so it reaches a text-consuming model (S4.6.2) |
| Inputs | `context.text_column` — a *declared* genuine untrusted text field; `--injection-text-field` on the driver |
| Preconditions | (1) the dataset has an untrusted free-text field; (2) the target model consumes text (`TEXT_CONSUMING_MODELS` = the Stage 2 rationale pathway only) |
| Applicable | **none of D1–D5** as shipped: no benchmark carries an untrusted free-text field (`DATASET_CAPABILITIES[*].untrusted_text_field is None`, `test_no_dataset_declares_an_untrusted_text_field`) |
| Not applicable | any of the nine structured baselines, on any dataset (they consume no text) |
| Manuscript status | the revised manuscript removed injection from the predictive-path tables and evaluates it "separately through the Stage 2 rationale-generation and Stage 3 deterministic-verifier pathway". That protocol requires a text field to inject into; the manuscript does not name one and the benchmarks have none. **Audit A-06 therefore remains open for the rationale-pathway evaluation.** This repository does not synthesise a text column to make the experiment run. |
| Randomness | payload choice, when it runs |
| Tests | `test_injection_is_not_applicable_to_any_benchmark`, `test_injection_is_not_applicable_to_text_free_models`, `test_injection_is_permitted_once_both_preconditions_hold` |

## ε₅ feat — bounded feature perturbation

| | |
|---|---|
| Implementation | `edits.py::feature_perturbation_edit` |
| Purpose | perturb one numeric feature within training-derived limits (S4.6.2) |
| Inputs | `context.feature_bounds` (per-column `(min, max)` **fitted on the training partition**), `context.mutable_columns` |
| Tunable | perturbation fraction of the training range, ±10 % — **ASSUMPTION** recorded in `edit_families.yaml` |
| Invariant | the perturbed value stays inside the training-derived bounds (clipped); immutable columns are never candidates |
| Applicable | D1, D2, D3, D4, D5 |
| Not applicable | — |
| Randomness | choice of column and of perturbation |
| Tests | `test_feature_perturbation_stays_inside_training_bounds`, `test_immutable_columns_cannot_be_edited` |

---

## Applicability matrix (as enforced)

| Dataset | split | delay | mule | inject | feat |
|---|---|---|---|---|---|
| D1 BAF | ✗ A-07 | ✗ month granularity | ✗ no graph | ✗ A-06 | ✓ |
| D2 AMLworld | ✓ | ✓ | ✓ | ✗ A-06 | ✓ |
| D3 IEEE-CIS | ✓ | ✓ | ✗ no graph | ✗ A-06 | ✓ |
| D4 Elliptic++ | ✗ no amount | ✓ | ✓ | ✗ A-06 | ✓ |
| D5 DGraph-Fin | ✗ not payments | ✓ | ✓ | ✗ A-06 | ✓ |

The revised manuscript's Tables 9(a–c) report all four structured families for
every listed model without naming the dataset(s) the evasion evaluation was run
on. Under the matrix above, a table with a non-empty `split` **and** `mule`
column can only have been produced on D2. That is recorded, not resolved, in
`MANUSCRIPT_AUDIT.md` (revision reconciliation, A-24).

## Adversarial training (S4.6.2), distinct from evaluation

`edit_families.yaml adversarial_training`: budget B = 2, 25 % of fraudulent
examples per epoch, up to four valid candidates (one per applicable structured
family; the original submission said five), keep the candidate with the
largest predictive loss, keep the original row, weight 0.50. Training-time
examples come from the training partition only and are not reused at test
time.

**Implementation status: the training-time selection loop (generate
candidates per fraudulent row, score them with the current model, keep the
max-loss candidate, add it with weight 0.50) is not wired into
`scripts/train.py`.** The edit functions above are what such a loop would
call, and the settings are recorded in configuration, but `scripts/train.py`
trains Stage 1 without adversarial augmentation. This is an implementation gap
of the package, not a manuscript ambiguity, and is tracked as L-34 in
`KNOWN_LIMITATIONS.md`. Ablation A4 ("w/o adversarial training") is therefore
the configuration `scripts/train.py` actually produces today.

## Registry ↔ implementation ↔ documentation

`tests/test_reviewer_artifacts.py::test_edit_family_registry_documentation_and_code_agree`
asserts that the families in `edit_families.yaml`, `EDIT_FAMILIES` in
`applicability.py`, the `EDIT_FUNCTIONS` table in `edits.py`, and the `## ε`
headings of this document are the same five names, and that every family
declares a `status_by_dataset` entry for all of D1–D5 which matches
`applicable_families(dataset, "lightgbm")`.
