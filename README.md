# VR-FraudNet

[![CI](https://github.com/abdulbasit12312/VR-FraudNet/actions/workflows/ci.yml/badge.svg)](https://github.com/abdulbasit12312/VR-FraudNet/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Reproducibility package for **"Grounding language models with deterministic
verifiers for fraud detection"**.

VR-FraudNet is a five-stage verifier-grounded framework for fraud detection: a
time-conditioned spectral graph encoder, a LightGBM triage classifier with
threshold routing, a schema-constrained rationale language model, a fixed
deterministic verifier, and an isotonic probability mixer with split-conformal
calibration. Its organising principle is that **the language model proposes and
the verifier disposes**: a rationale-side probability can only influence an
automated decision if a non-trainable verifier has confirmed every claim against
the transaction, temporal and graph evidence.

```
                     ┌──────────────────────────────────────────────┐
  transaction ──────►│ Stage 0  time-conditioned spectral encoder    │  z_g ∈ ℝ⁶⁴
  graph · time       └──────────────────────────────────────────────┘
                                        │
                     ┌──────────────────▼──────────────────────────┐
                     │ Stage 1  LightGBM triage + threshold gate    │
                     └───┬───────────────┬───────────────────┬─────┘
                accept ◄─┘        escalate│                   └─► reject
                                          ▼
                     ┌──────────────────────────────────────────────┐
                     │ Stage 2  schema-constrained rationale LM      │
                     │          grammar-masked, greedy, ≤ 8 claims   │
                     └──────────────────┬───────────────────────────┘
                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │ Stage 3  fixed deterministic verifier V∈{0,1} │
                     └──────┬────────────────────────────┬──────────┘
                       V=1  │                        V=0 │
                            ▼                            ▼
                     ┌────────────────────┐     ┌──────────────────────┐
                     │ Stage 4  isotonic  │     │ human-review          │
                     │ mixer + conformal  │     │ escalation            │
                     └────────────────────┘     └──────────────────────┘
```

---

## Reproducing the study

Reviewer-oriented map of every artefact needed to reproduce the study, with the
exact path, the command that exercises it, and what cannot be shared and why.
Full detail: [`docs/REVIEWER_REPRODUCIBILITY.md`](docs/REVIEWER_REPRODUCIBILITY.md)
and [`docs/ARTIFACT_AVAILABILITY.md`](docs/ARTIFACT_AVAILABILITY.md).

| Material | Where | Status |
|---|---|---|
| Deterministic verifier rules | [`verifier/rules/claim_primitives.yaml`](verifier/rules/claim_primitives.yaml); engine [`src/vrfraudnet/verifier/`](src/vrfraudnet/verifier/) | shared |
| Rationale JSON schema (Draft 2020-12) | [`schemas/rationale_schema.json`](schemas/rationale_schema.json) | shared |
| Grammar-masked decoding | [`src/vrfraudnet/models/grammar.py`](src/vrfraudnet/models/grammar.py) · [`docs/DECODING_GRAMMAR.md`](docs/DECODING_GRAMMAR.md) | shared |
| Stage 2 LoRA configuration, training, loading | [`configs/stage2_lora.yaml`](configs/stage2_lora.yaml) · [`scripts/train_stage2_lora.py`](scripts/train_stage2_lora.py) · [`scripts/validate_stage2_adapter.py`](scripts/validate_stage2_adapter.py) · [`docs/STAGE2_LORA.md`](docs/STAGE2_LORA.md) | shared |
| **Stage 2 LoRA adapter weights** | [`artifacts/lora/README.md`](artifacts/lora/README.md) | **NOT AVAILABLE** — the manuscript adapter is absent from every accessible project location; none is fabricated |
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` (gated, Llama 3.1 Community License) | not redistributed; **immutable revision not recorded** (L-29) |
| Stage 2 rationale training corpus | format in [`docs/STAGE2_LORA.md`](docs/STAGE2_LORA.md), synthetic example [`examples/stage2_corpus_example.jsonl`](examples/stage2_corpus_example.jsonl) | **NOT AVAILABLE; construction unspecified** (L-25) |
| Retrieval setup | [`retrieval/config/retrieval.yaml`](retrieval/config/retrieval.yaml) · [`src/vrfraudnet/retrieval/index.py`](src/vrfraudnet/retrieval/index.py) · [`scripts/build_retrieval_index.py`](scripts/build_retrieval_index.py) · [`docs/RETRIEVAL_SETUP.md`](docs/RETRIEVAL_SETUP.md) | shared; training-only, strictly-earlier, identifier-excluded |
| Edit generators and applicability rules | [`adversarial/families/edit_families.yaml`](adversarial/families/edit_families.yaml) · [`src/vrfraudnet/adversarial/`](src/vrfraudnet/adversarial/) · [`docs/EDIT_GENERATORS.md`](docs/EDIT_GENERATORS.md) | shared; injection undefined on every benchmark (A-06) |
| Experiment configurations | [`configs/`](configs/) — every value tagged MANUSCRIPT / ASSUMPTION / DERIVED | shared |
| Manuscript seed set | [`configs/manuscript_seeds.yaml`](configs/manuscript_seeds.yaml) ↔ `vrfraudnet.seeds.MANUSCRIPT_SEEDS` (test-synchronised) | shared |
| Environment | [`requirements.txt`](requirements.txt) · [`environment.yml`](environment.yml) · reference machine in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | shared |
| Versioned archive, manifest, checksums | [`REPRODUCIBILITY_MANIFEST.json`](REPRODUCIBILITY_MANIFEST.json) · [`docs/RELEASE.md`](docs/RELEASE.md) · tag `v1.0.0-reproducibility` | built by `scripts/build_reproducibility_bundle.py`; publication status in `docs/RELEASE.md` |
| What cannot be reproduced and why | [`docs/KNOWN_LIMITATIONS.md`](docs/KNOWN_LIMITATIONS.md) · [`docs/MANUSCRIPT_AUDIT.md`](docs/MANUSCRIPT_AUDIT.md) | — |

Three different things, kept apart throughout this repository:

1. **Implementation reproducibility** — every stage of the method is
   implemented, configured from the manuscript, and tested. This is complete
   except for the adversarial-training loop (L-34).
2. **Exact numerical reproduction of the manuscript** — **not possible from
   this package.** The Stage 2 adapter and its training corpus are unavailable
   (L-25, L-31), baseline hyper-parameters are unpublished (L-18), cost values
   are only in an unavailable supplement (L-26), and the D3 Recall@top-1%
   column is arithmetically unattainable (A-01).
3. **Experiments that can be run** — every Stage 2-free experiment, with the
   manuscript's seeds and configurations, once the datasets are obtained; the
   numbers will differ from the paper for the documented reasons and must be
   reported with those caveats.

Four levels of checking, from no data to the full sweep:

```bash
git clone https://github.com/abdulbasit12312/VR-FraudNet.git && cd VR-FraudNet
git checkout v1.0.0-reproducibility
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest jsonschema

# Level A - integrity, no data
python -m pytest -q
python examples/run_minimal_demo.py
python scripts/audit_manuscript.py --revision revised

# Level B - reviewer artefacts and the archive
python scripts/audit_reviewer_artifacts.py
python scripts/validate_rationale_schema.py
python scripts/build_reproducibility_bundle.py --check-manifest
python scripts/build_reproducibility_bundle.py --version 1.0.0
python scripts/verify_reproducibility_bundle.py dist/VR-FraudNet-reproducibility-1.0.0.tar.gz

# Level C - one dataset, one seed (datasets obtained per docs/DATA_AVAILABILITY.md)
python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml
python scripts/train.py        --dataset D1 --config configs/d1.yaml --seed 42
python scripts/evaluate.py     --dataset D1 --config configs/d1.yaml --seed 42
python scripts/calibrate.py    --dataset D1 --config configs/d1.yaml --seed 42

# Level D - the complete planned sweep (prints the plan; --execute runs it)
bash reproduce_all.sh
```

---

## What this repository is, and is not

**It is** a complete, executable implementation of the methodology: leakage-
controlled preprocessing for all five benchmarks, the spectral encoder, the
triage and routing layer, the exact machine-readable rationale schema, a
working grammar-masked decoder, the full deterministic verifier with its
declarative rule set, the training-only retrieval index, isotonic mixing,
split-conformal calibration, all five adversarial edit families, the temporal
and transfer protocols, and the statistical tests — with 264 tests covering the
leakage and correctness properties that matter.

**It is not** a claim to have reproduced the manuscript's numbers. It ships no
datasets, no checkpoints, no LoRA adapter, no predictions and no fabricated
results. Every table
builder reads result files produced by an actual run and refuses to emit
anything else.

Several manuscript experiments **cannot** be reproduced as published. Those are
documented, with the arithmetic, in
[`docs/MANUSCRIPT_AUDIT.md`](docs/MANUSCRIPT_AUDIT.md) — and the checks are
rerunnable:

```bash
python scripts/audit_manuscript.py
```

The headline findings:

| ID | Finding | Severity |
|---|---|---|
| A-01 | Table 5(c) Recall@top-1% exceeds its arithmetic ceiling (0.2857 at 3.50% prevalence) for **every** model | blocking |
| A-06 | Prompt-injection evasion rates are reported for text-free models on datasets with no text field | blocking |
| A-14 | Cross-dataset transfer is reported without any feature-space alignment being specified | blocking |
| A-17 | The escalation-path p99 of 175.23 ms is below the memory-bandwidth floor for 8B single-stream decoding | major |
| A-02 | Table 6(a) mean ΔAUPRC values disagree with the means implied by Table 5 | major |
| A-16 | Table 9(a)'s "Train AUPRC" reference equals Table 5(a)'s held-out test value, yet differs from the mean of its own month columns | major |

Where an experiment is blocked, the code raises
`ManuscriptInconsistencyError` or `NotApplicableError` naming the finding,
instead of guessing.

---

## Quick start

```bash
git clone https://github.com/abdulbasit12312/VR-FraudNet.git
cd VR-FraudNet
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q                  # 264 tests, no data required
python examples/run_minimal_demo.py  # synthetic end-to-end walk-through
python scripts/audit_manuscript.py   # rerun every consistency check
```

Then obtain the datasets ([`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md))
and run:

```bash
python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml
python scripts/train.py        --dataset D1 --config configs/d1.yaml --seed 42
python scripts/evaluate.py     --dataset D1 --config configs/d1.yaml --seed 42
python scripts/calibrate.py    --dataset D1 --config configs/d1.yaml --seed 42
python scripts/make_tables.py  --table 5
```

A documented plan for the whole sweep:

```bash
bash reproduce_all.sh             # print the plan
bash reproduce_all.sh --execute   # run it
```

---

## Repository layout

```
configs/       dataset, baseline, ablation and cost configurations
               every value tagged MANUSCRIPT / ASSUMPTION / DERIVED
schemas/       the machine-readable Stage 2 rationale schema
verifier/      declarative Stage 3 rule set (primitives, operators, CR1-CR6)
src/vrfraudnet/
  data/        registry, splits, leakage-controlled preprocessing (D1-D5)
  models/      Stage 0 encoder, Stage 1 triage, Stage 2 LM, grammar, Stage 4, pipeline
  verifier/    evidence model, five claim primitives, consistency engine
  retrieval/   training-only retrieval index
  adversarial/ edit families and the applicability matrix
  evaluation/  metrics, temporal, transfer, cost, latency
  statistics/  Wilcoxon, DeLong, Holm-Bonferroni, bootstrap
  baselines/   the nine baselines of Section 4.7
  tables/      table builders (read result files only)
scripts/       prepare_data, train, evaluate, calibrate, robustness, transfer,
               stats, make_tables, latency_bench, audit_manuscript,
               train_stage2_lora, validate_stage2_adapter, build_retrieval_index,
               validate_rationale_schema, audit_reviewer_artifacts,
               build_reproducibility_bundle, verify_reproducibility_bundle
tests/         264 tests: leakage, verifier, grammar, edits, metrics, statistics,
               determinism, repository hygiene, reviewer artefacts
artifacts/     Stage 2 adapter availability statement (no weights are shipped)
docs/          code map, reproducibility, data availability, limitations, audit,
               reviewer map, artefact availability, Stage 2, grammar, retrieval,
               edit generators, release
examples/      example rationales and a synthetic end-to-end demo
```

---

## Design commitments

**Leakage is enforced, not asserted.** `TrainOnlyPipeline` records the index set
it was fitted on and raises `LeakageError` if asked to fit on anything else or
to fit twice. Split constructors assert temporal ordering between every pair of
partitions. `TemporalGraph.snapshot(t)` is the only way Stage 0 sees a graph, and
`assert_no_future_edges` guards it. Retrieval records whose partition is not
`train` cannot be constructed at all.

**Provenance is explicit.** Every configuration value carries a tag:
`MANUSCRIPT` (with the section that states it), `ASSUMPTION` (with a
`docs/KNOWN_LIMITATIONS.md` id), or `DERIVED`. Every run prints its assumptions
before it starts. A test asserts that no `ASSUMPTION` cites a limitation id that
does not exist.

**Nothing is hard-coded.** `tests/test_repo_hygiene.py::test_no_manuscript_numbers_in_the_runtime_path`
fails if a manuscript AUPRC appears anywhere outside documentation and
`scripts/audit_manuscript.py`, whose purpose is to test those numbers.

**Undefined experiments refuse to run.** Injection against a gradient-boosted
tree, transaction splitting on an account-application dataset, transfer without
an alignment, cost analysis without costs — each raises with an explanation
rather than producing a number.

---

## The verifier

The Stage 3 rule set lives in
[`verifier/rules/claim_primitives.yaml`](verifier/rules/claim_primitives.yaml)
and is loaded, never learned. Five primitives — numeric comparison, set
membership, temporal relation, graph path, entity match — and six cross-claim
consistency rules:

| Rule | Requirement |
|---|---|
| CR1 | No two claims may assert an unsatisfiable interval on the same field |
| CR2 | Every `evidence_id` must be declared *and* resolve in the transaction's evidence |
| CR3 | The verdict must be supported by the verified claims |
| CR4 | No duplicate claims |
| CR5 | No temporal claim may reference an event after the transaction |
| CR6 | Between 1 and 8 claims |

A claim whose evidence is absent is `unverifiable`, which the manuscript treats
as rejection (Table 4, example E2) — and so does this implementation.

```python
from vrfraudnet.verifier import DeterministicVerifier, TransactionEvidence, GraphEvidence

verifier = DeterministicVerifier()
result = verifier.verify(rationale, evidence)
print(result.status, result.failure_reasons)   # 0 or 1, plus the audit trace
```

---

## Grammar-masked decoding

`vrfraudnet.models.grammar.RationaleGrammar` is an incremental parser over the
rationale schema. At each decoding step it answers "can this prefix still become
a schema-valid object?", which yields the token mask directly and makes the
masking behaviour testable without a language model:

```python
grammar.allowed_characters('{"verdict":"')     # {'f', 'l', 'u'}
grammar.token_mask('{"verdict":', ['"fraud"', '"maybe"'])   # [True, False]
```

---

## Citation

See [`CITATION.cff`](CITATION.cff). Cite the manuscript for the method and this
repository (tag `v1.0.0-reproducibility`, version 1.0.0) for the
implementation. The manuscript's DOI is unassigned at the time of this release
and must be filled in when it is known.

---

## Licence

MIT for the source code ([`LICENSE`](LICENSE)). The five benchmark datasets are
**not** covered by it and remain under their providers' terms — several are
non-commercial or require an individual agreement. See
[`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).
