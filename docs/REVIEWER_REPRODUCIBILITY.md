# Reviewer reproducibility map

**The request this document answers** (reviewer / editor, revision round):

> Please also explain how readers can access the code needed to reproduce the
> study. This includes the verifier rules, JSON schema, decoding grammar, LoRA
> adapters, retrieval setup and edit generators. A versioned archive with
> configuration files and seeds would help readers check the results. If any
> materials cannot be shared, state the reason and explain what can be made
> available.

Every item named in the request is mapped below to an exact path in this
repository, the command that exercises it, its availability status, its
provenance, and its limitation if any. Paths are relative to the repository
root and all exist at the tagged commit; `tests/test_reviewer_artifacts.py::test_reviewer_matrix_paths_exist`
fails if any of them disappears.

**Release to cite for the revised manuscript:** tag `v1.0.0-manuscript-revision`,
archive `VR-FraudNet-reproducibility-1.0.0.tar.gz` with `SHA256SUMS` and
`REPRODUCIBILITY_MANIFEST.json` attached. Publication status and the exact
commit are recorded in [`RELEASE.md`](RELEASE.md).

---

## Artifact matrix

| Reviewer-requested material | Exact repository path | Command / example | Availability | Version / provenance | Limitation if any |
|---|---|---|---|---|---|
| Verifier rule definitions | `verifier/rules/claim_primitives.yaml` | `python scripts/validate_rationale_schema.py` | AVAILABLE AND SHARED | rule file `version: 1`; manuscript S4.3.2, S4.4; SHA-256 in manifest | float tolerance (L-13) and risk direction (L-14) are documented defaults |
| Verifier execution engine | `src/vrfraudnet/verifier/engine.py`, `src/vrfraudnet/verifier/primitives.py`, `src/vrfraudnet/verifier/evidence.py` | `python -m pytest -q tests/test_verifier.py`; `python examples/run_minimal_demo.py` | AVAILABLE AND SHARED | manuscript Eq. 6–7, S4.3.5, Table 4; engine loads the YAML, never hard-codes operators | unavailable graph evidence ⇒ `unverifiable` ⇒ reject (Table 4 E2) — implemented as stated |
| Rationale JSON schema | `schemas/rationale_schema.json` | `python scripts/validate_rationale_schema.py` | AVAILABLE AND SHARED | JSON Schema Draft 2020-12 with `$id`; manuscript S4.3.2 | — |
| Decoding grammar | `src/vrfraudnet/models/grammar.py`; documented in `docs/DECODING_GRAMMAR.md` | `python -m pytest -q tests/test_grammar_masking.py` | AVAILABLE AND SHARED | derived from the schema at construction; manuscript S4.3.3 | own parser (L-07); fixed key order (L-15) |
| Stage 2 LoRA configuration | `configs/stage2_lora.yaml`; explained in `docs/STAGE2_LORA.md` | `python -c "from vrfraudnet.models.stage2_rationale import Stage2Config; print(Stage2Config.from_yaml())"` | AVAILABLE AND SHARED | every value tagged MANUSCRIPT / ASSUMPTION / DERIVED; S4.3.1, S4.8.1, S4.8.2 | per-device batch size, accumulation steps, gradient clipping, AdamW betas, post-warm-up decay, per-seed adapter policy not stated (L-30) |
| Stage 2 LoRA training script | `scripts/train_stage2_lora.py` | `python scripts/train_stage2_lora.py --dataset D2 --config configs/d2.yaml --stage2-config configs/stage2_lora.yaml --corpus examples/stage2_corpus_example.jsonl --seed 42 --dry-run` | AVAILABLE AND SHARED | implements S4.3.1 filters, S4.6.3 counterfactuals, PEFT LoRA training | needs gated base weights, GPU, and a corpus (see next rows) |
| Stage 2 LoRA inference / loading path | `src/vrfraudnet/models/stage2_rationale.py::RationaleModel.load`, `::_decode` | `python scripts/validate_stage2_adapter.py --adapter <dir> --load` | AVAILABLE AND SHARED | greedy, temperature 0, grammar-masked, 384-token cap (S4.3.3) | raises `MissingArtefactError` when no adapter is supplied — by design |
| **LoRA adapter weights** | `artifacts/lora/README.md` (availability statement; no weights) | `python scripts/validate_stage2_adapter.py --adapter artifacts/lora/manuscript` → exit 2 | **NOT AVAILABLE** | searched: working tree, full git history, tags, branches, LFS, maintainers' project directories | the adapter used for the manuscript is not present in the accessible project materials; exact Stage 2 reproduction cannot be claimed |
| Base model identifier and immutable revision | `configs/stage2_lora.yaml` → `base_model.repository`, `base_model.revision` | `--base-model-revision <sha>` on the training script (recorded in the run record) | identifier AVAILABLE; revision **REQUIRES AUTHOR CONFIRMATION** | `meta-llama/Meta-Llama-3.1-8B-Instruct` (S4.3.1); Transformers 4.40.1 / PEFT 0.10.0 (S4.8.2) | HF commit not recorded (L-29); weights not redistributed (Llama 3.1 Community License) |
| Rationale-training data construction / provenance | `docs/STAGE2_LORA.md` §4 (format), §5 (status); `examples/stage2_corpus_example.jsonl` (synthetic, 2 records) | `python scripts/train_stage2_lora.py … --dry-run` validates a corpus | **NOT AVAILABLE; REQUIRES AUTHOR CONFIRMATION** | manuscript S4.3.1 states filters, not authorship | rule / teacher / human authorship unspecified (L-25); the one related project document is a pre-implementation plan, not a record |
| Retrieval encoder / model | `retrieval/config/retrieval.yaml` → `encoder`; `src/vrfraudnet/retrieval/index.py::default_encoder` | `python scripts/build_retrieval_index.py … --encoder-revision <sha>` | AVAILABLE BUT NOT REDISTRIBUTABLE (Hub, Apache-2.0) | `sentence-transformers/all-MiniLM-L6-v2`, dim 384, cosine (S4.3.4); sentence-transformers 2.7.0 (S4.8.2) | revision not recorded (L-29) |
| Retrieval configuration | `retrieval/config/retrieval.yaml`; `docs/RETRIEVAL_SETUP.md` | `python -m pytest -q tests/test_reviewer_artifacts.py -k retrieval` | AVAILABLE AND SHARED | k = 4, training-only, strictly-earlier, identifier exclusions (S4.3.4) | exact search instead of an unnamed FAISS index type (L-33) |
| Retrieval index builder | `scripts/build_retrieval_index.py`; `RetrievalIndex.save/load` | `python scripts/build_retrieval_index.py --dataset D2 --config configs/d2.yaml --corpus examples/stage2_corpus_example.jsonl --seed 42 --encoder test-hashing --output-dir /tmp/idx` | AVAILABLE AND SHARED | writes `records.json`, `embeddings.npy`, checksummed `index_manifest.json` | no index shipped: its input corpus is unavailable (L-25) |
| Retrieval eligibility / filtering rules | `src/vrfraudnet/retrieval/index.py::RetrievalRecord.__post_init__`, `::_is_eligible` | see leakage tests | AVAILABLE AND SHARED | S4.3.4 | — |
| Retrieval leakage tests | `tests/test_retrieval_leakage.py`; `tests/test_reviewer_artifacts.py::test_retrieval_index_refuses_every_non_training_partition` | `python -m pytest -q tests/test_retrieval_leakage.py` | AVAILABLE AND SHARED | — | — |
| Edit / adversarial generator registry | `adversarial/families/edit_families.yaml`; `src/vrfraudnet/adversarial/applicability.py::EDIT_FAMILIES`; `src/vrfraudnet/adversarial/edits.py::EDIT_FUNCTIONS` | `python -m pytest -q tests/test_adversarial_edits.py` | AVAILABLE AND SHARED | S4.6.2, S5.6 | adversarial-training loop not wired into `scripts/train.py` (L-34) |
| Individual edit families | `src/vrfraudnet/adversarial/edits.py::{split_edit,delay_edit,mule_edit,inject_edit,feature_perturbation_edit}`; `docs/EDIT_GENERATORS.md` | `python scripts/robustness.py --dataset D2 --config configs/d2.yaml --seed 42 --protocol evasion --budgets 1 2 4` | AVAILABLE AND SHARED | ε₁–ε₅ as in S4.6.2 | max delay (L-17) and perturbation fraction are documented defaults |
| Edit applicability rules | `src/vrfraudnet/adversarial/applicability.py::check_applicable`, `DATASET_CAPABILITIES`; matrix in `docs/EDIT_GENERATORS.md` | `python -c "from vrfraudnet.adversarial.applicability import applicability_report; print(applicability_report(['D1','D2','D3','D4','D5'], ['lightgbm']))"` | AVAILABLE AND SHARED | — | injection undefined on every benchmark (A-06); split undefined on D1 (A-07); neither is relaxed |
| Experiment configurations | `configs/base.yaml`, `configs/d1.yaml` … `configs/d5.yaml`, `configs/baselines.yaml`, `configs/ablations.yaml`, `configs/costs.yaml` | `python -m pytest -q tests/test_config_and_registry.py` | AVAILABLE AND SHARED | every leaf value provenance-tagged | baseline hyper-parameters are assumptions (L-18); cost values are `null` (L-26) |
| Seed list | `configs/manuscript_seeds.yaml` ↔ `src/vrfraudnet/seeds.py::MANUSCRIPT_SEEDS` | `python -m pytest -q tests/test_reviewer_artifacts.py -k seed` | AVAILABLE AND SHARED | the ten seeds of S4.8.1, unchanged | — |
| Environment / dependency lock | `requirements.txt`, `requirements-optional.txt`, `environment.yml`, `pyproject.toml`; reference machine and versions in `docs/REPRODUCIBILITY.md` §1 | `pip install -r requirements.txt` | AVAILABLE AND SHARED | manuscript S4.8.2 versions listed | bitwise reproduction across hardware is not attainable (L-28) |
| Main reproduction script | `reproduce_all.sh` (dry run by default) | `bash reproduce_all.sh`; `bash reproduce_all.sh --execute` | AVAILABLE AND SHARED | prints BLOCKED markers with the audit id for steps that cannot run | Stage 2-dependent steps are blocked |
| Release manifest | `REPRODUCIBILITY_MANIFEST.json` | `python scripts/build_reproducibility_bundle.py --check-manifest` | AVAILABLE AND SHARED | real SHA-256 of every listed file; git commit and tag recorded at packaging | — |
| SHA-256 checksum manifest | `SHA256SUMS` inside the archive and as a release asset | `python scripts/verify_reproducibility_bundle.py dist/VR-FraudNet-reproducibility-1.0.0.tar.gz`; or `sha256sum -c SHA256SUMS` inside the unpacked archive | AVAILABLE AND SHARED | generated by the bundle builder | — |
| Versioned GitHub Release / archive | tag `v1.0.0-manuscript-revision`; `dist/VR-FraudNet-reproducibility-1.0.0.tar.gz` (built, not committed) | `python scripts/build_reproducibility_bundle.py --version 1.0.0` | see `docs/RELEASE.md` | deterministic tarball (sorted entries, fixed mtime, no uid/gid) | contains no datasets, weights, indexes, results or secrets |

---

## "Materials that cannot be shared" — the reasons, in one place

| Material | Why it cannot be shared | What is made available instead |
|---|---|---|
| Stage 2 LoRA adapter used for the manuscript | The exact trained LoRA adapter checkpoint used in the original experiments is not included because a standalone archival copy of the fitted adapter was not retained as a release-ready artifact during the original experimental workflow. The repository provides the complete LoRA training configuration and implementation, including the base-model specification, target modules, rank, scaling factor, dropout, optimizer settings, learning rate, training epochs, sequence lengths, and predefined seeds, allowing the adaptation procedure to be rerun. Exact checkpoint-level reproduction of the original Stage 2 model is therefore not possible from the currently released materials. Not a licensing restriction. | the exact configuration (`configs/stage2_lora.yaml`), the training script, the loader, the validator, and the availability statement (`artifacts/lora/README.md`) |
| Stage 2 rationale training corpus | **absent**, and its construction method is not described in the manuscript | corpus format, loader with leakage checks, the S4.3.1 filters as code, the counterfactual generator, and a synthetic two-record format example |
| Llama-3.1-8B-Instruct weights | third-party licence (Llama 3.1 Community License), gated download | identifier, access instructions, and a `revision` field to pin the commit |
| all-MiniLM-L6-v2 weights | third-party (Apache-2.0), downloaded by the library | identifier, dimension, and a `--encoder-revision` flag |
| Benchmark datasets D1–D5 | provider terms (CC BY-NC-SA, CDLA-Sharing, Kaggle competition rules, research agreement) | provider links, file lists, checksum procedure (`docs/DATA_AVAILABILITY.md`) |
| Prediction vectors and result files behind the tables | not in the accessible project materials | every script needed to regenerate what can be regenerated; table builders that refuse to print anything else |
| Supplementary Tables S1–S4, cost values | not supplied to the maintainers | code paths exist; `configs/costs.yaml` refuses to run with `null` costs rather than guess |

---

## How to check the package in under two minutes

```bash
git clone https://github.com/abdulbasit12312/VR-FraudNet.git && cd VR-FraudNet
git checkout v1.0.0-manuscript-revision
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt pytest jsonschema

python scripts/audit_reviewer_artifacts.py             # every path in this table exists; README links resolve
python scripts/validate_rationale_schema.py            # schema, rules, examples, grammar
python scripts/build_reproducibility_bundle.py --check-manifest
python -m pytest -q                                    # full suite, no data required
```

Then read `docs/MANUSCRIPT_AUDIT.md` before comparing any number against the
paper.
