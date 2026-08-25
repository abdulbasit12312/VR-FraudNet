# VR-FraudNet

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

## What this repository is, and is not

**It is** a complete, executable implementation of the methodology: leakage-
controlled preprocessing for all five benchmarks, the spectral encoder, the
triage and routing layer, the exact machine-readable rationale schema, a
working grammar-masked decoder, the full deterministic verifier with its
declarative rule set, the training-only retrieval index, isotonic mixing,
split-conformal calibration, all five adversarial edit families, the temporal
and transfer protocols, and the statistical tests — with 228 tests covering the
leakage and correctness properties that matter.

**It is not** a claim to have reproduced the manuscript's numbers. It ships no
datasets, no checkpoints, no predictions and no fabricated results. Every table
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

python -m pytest -q                  # 228 tests, no data required
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
               stats, make_tables, latency_bench, audit_manuscript
tests/         228 tests: leakage, verifier, grammar, edits, metrics, statistics,
               determinism, repository hygiene
docs/          code map, reproducibility, data availability, limitations, audit
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
repository for the implementation. The manuscript's DOI is unassigned at the
time of this release and must be filled in before an archival tag.

---

## Licence

MIT for the source code ([`LICENSE`](LICENSE)). The five benchmark datasets are
**not** covered by it and remain under their providers' terms — several are
non-commercial or require an individual agreement. See
[`docs/DATA_AVAILABILITY.md`](docs/DATA_AVAILABILITY.md).
