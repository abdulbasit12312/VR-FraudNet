# Artifact availability statement

Every artefact involved in producing or checking the manuscript's results,
classified with one status each. Nothing listed as available has been
fabricated; nothing listed as unavailable has been replaced by a substitute.

**Status vocabulary**

| Status | Meaning |
|---|---|
| AVAILABLE AND SHARED | in this repository and in the reproducibility archive, with a SHA-256 in `REPRODUCIBILITY_MANIFEST.json` |
| AVAILABLE BUT NOT REDISTRIBUTABLE | obtainable from a third party under its own terms; this repository records identifiers, versions and access steps, never the bytes |
| RECONSTRUCTABLE FROM PUBLISHED INFORMATION | not shipped, but a script in this repository builds it deterministically from shared inputs or from a third-party artefact the reader obtains |
| NOT AVAILABLE | absent from the accessible project materials; no script can rebuild it to the manuscript's specification |
| REQUIRES AUTHOR CONFIRMATION | the manuscript is silent or ambiguous on something the artefact depends on; the gap is named in `KNOWN_LIMITATIONS.md` |

---

| Artifact | Public? | Location | Restriction / reason | Reconstructable? | Verification |
|---|---|---|---|---|---|
| Source code (all five stages, verifier, retrieval, edits, evaluation, statistics, baselines) | AVAILABLE AND SHARED | `src/vrfraudnet/`, `scripts/`, MIT licence | — | n/a | `python -m pytest -q`; hashes in the manifest |
| Deterministic verifier rules (five primitives, CR1–CR6, risk direction, tolerances) | AVAILABLE AND SHARED | `verifier/rules/claim_primitives.yaml` | — | n/a | `python scripts/validate_rationale_schema.py`; `tests/test_verifier.py` |
| Verifier execution engine | AVAILABLE AND SHARED | `src/vrfraudnet/verifier/{engine,primitives,evidence}.py` | — | n/a | `tests/test_verifier.py` |
| Rationale JSON schema (Draft 2020-12) | AVAILABLE AND SHARED | `schemas/rationale_schema.json` | — | n/a | `python scripts/validate_rationale_schema.py` |
| Decoding grammar | AVAILABLE AND SHARED | `src/vrfraudnet/models/grammar.py`, `docs/DECODING_GRAMMAR.md` | own incremental parser; the manuscript names no library (L-07) | n/a | `tests/test_grammar_masking.py`, reviewer cases in `tests/test_reviewer_artifacts.py` |
| Experiment configurations (per dataset, base, baselines, ablations, costs) | AVAILABLE AND SHARED | `configs/*.yaml`, every value provenance-tagged | `configs/costs.yaml` ships `null` cost values — the amounts are only in Supplementary S1 (L-26, A-10) | n/a | `tests/test_config_and_registry.py` |
| Manuscript seed set | AVAILABLE AND SHARED | `configs/manuscript_seeds.yaml` ↔ `vrfraudnet.seeds.MANUSCRIPT_SEEDS` | — | n/a | `tests/test_reviewer_artifacts.py::test_seed_file_matches_code_constant` |
| Stage 2 LoRA configuration | AVAILABLE AND SHARED | `configs/stage2_lora.yaml`, `docs/STAGE2_LORA.md` | fields the manuscript omits are `null`, not guessed (L-29, L-30, L-31) | n/a | `tests/test_reviewer_artifacts.py` (config ↔ dataclass ↔ base.yaml) |
| Stage 2 training and adapter-validation scripts | AVAILABLE AND SHARED | `scripts/train_stage2_lora.py`, `scripts/validate_stage2_adapter.py` | a real run needs the gated base weights, a GPU and a corpus | n/a | `--dry-run` in CI on the synthetic example corpus |
| **Stage 2 LoRA adapter weights (as used for the manuscript)** | **NOT AVAILABLE** | `artifacts/lora/README.md` (statement only) | absent from the working tree, the full git history, all branches/tags, and the maintainers' project directories; no metadata, log or checksum survives | **No.** A new adapter can be trained under the documented configuration and is labelled a reconstruction, but the corpus it needs is also unavailable | `python scripts/validate_stage2_adapter.py --adapter artifacts/lora/…` exits 2 (missing) by design |
| Base model `meta-llama/Meta-Llama-3.1-8B-Instruct` | AVAILABLE BUT NOT REDISTRIBUTABLE | Hugging Face Hub, gated | Llama 3.1 Community License; weights (~16 GB bf16) are never placed in this repository or the archive | n/a | identifier recorded in `configs/stage2_lora.yaml`; **immutable revision not recorded by the manuscript (L-29)** |
| **Stage 2 rationale training corpus** (target rationales for review-band training transactions) | **NOT AVAILABLE** and **REQUIRES AUTHOR CONFIRMATION** | none | the manuscript does not state how target rationales were authored (rule / teacher model / human) — L-25; the only related project document is a pre-implementation plan with no execution record (`docs/STAGE2_LORA.md` §5) | **No.** Format, loader, filters and generator interfaces are shipped so a corpus the authors supply drops in unchanged | corpus format check: `scripts/train_stage2_lora.py --dry-run` |
| Retrieval configuration | AVAILABLE AND SHARED | `retrieval/config/retrieval.yaml`, `docs/RETRIEVAL_SETUP.md` | encoder revision not recorded (L-29); exact rather than approximate search (L-33) | n/a | `tests/test_reviewer_artifacts.py::test_retrieval_config_matches_runtime_constants` |
| Retrieval implementation and index builder | AVAILABLE AND SHARED | `src/vrfraudnet/retrieval/index.py`, `scripts/build_retrieval_index.py` | — | n/a | `tests/test_retrieval_leakage.py` |
| Retrieval encoder `sentence-transformers/all-MiniLM-L6-v2` | AVAILABLE BUT NOT REDISTRIBUTABLE | Hugging Face Hub (Apache-2.0) | not bundled; downloaded by `sentence-transformers` | n/a | repository and dimension recorded; revision pinned by the operator at build time |
| Retrieval index (per dataset) | NOT AVAILABLE / RECONSTRUCTABLE | would be `retrieval/index/<dataset>/` | built from the rationale corpus, which is not available | **Yes, once a corpus exists**: `scripts/build_retrieval_index.py` writes a checksummed, manifested index | `RetrievalIndex.load` verifies the checksums |
| Edit / adversarial generators and applicability rules | AVAILABLE AND SHARED | `src/vrfraudnet/adversarial/{edits,applicability}.py`, `adversarial/families/edit_families.yaml`, `docs/EDIT_GENERATORS.md` | prompt injection cannot run on any benchmark (A-06); split disabled on D1 (A-07); adversarial-training loop not wired (L-34) | n/a | `tests/test_adversarial_edits.py` |
| Raw benchmark datasets D1–D5 | AVAILABLE BUT NOT REDISTRIBUTABLE | providers listed in `docs/DATA_AVAILABILITY.md` | CC BY-NC-SA / CDLA-Sharing / Kaggle competition terms / research agreement; never committed (`tests/test_repo_hygiene.py::test_no_datasets_committed`) | n/a | operator records SHA-256 in `datasets/checksums/` |
| Processed datasets (post Table 2 preprocessing) | RECONSTRUCTABLE FROM PUBLISHED INFORMATION | `scripts/prepare_data.py` from the raw data | same third-party terms; feature counts A-08/A-09 and validation carve-outs L-02/L-04/L-05 are documented defaults | **Yes** | `results/prepared/<dataset>/report.json` |
| Stage 0 / Stage 1 / Stage 4 checkpoints | NOT AVAILABLE / RECONSTRUCTABLE | none shipped | never committed (`test_no_model_weights_committed`) | **Yes**, from the raw data with `scripts/train.py` — under documented defaults, so numbers are not expected to match the paper bitwise (`REPRODUCIBILITY.md` §6) | result-file manifests record config hash, seed, git revision |
| Per-transaction prediction vectors behind Tables 5–9 | NOT AVAILABLE | none | not part of the accessible project materials; this is also why the original DeLong statistics could not be retained (the revised manuscript withdrew them) | **Partially**: `scripts/train.py` + `scripts/evaluate.py` regenerate predictions for every run they can perform; Stage 2-dependent rows cannot be regenerated | `results/predictions/<dataset>/<model>_seed<seed>.json` |
| Result files behind the manuscript tables | NOT AVAILABLE | none | not in the accessible project materials; `scripts/make_tables.py` reads only `results/` produced by an actual run and has no path that prints manuscript values | see above | `tests/test_repo_hygiene.py::test_no_manuscript_numbers_in_the_runtime_path` |
| Supplementary Tables S1–S4 | NOT AVAILABLE | none | not supplied to the maintainers (L-27) | — | — |
| Environment / dependency information | AVAILABLE AND SHARED | `requirements.txt`, `requirements-optional.txt`, `environment.yml`, `pyproject.toml`; reference machine in `REPRODUCIBILITY.md` §1 | — | n/a | `python -c "from vrfraudnet.io_utils import _environment; print(_environment())"` |
| Reproducibility manifest and checksums | AVAILABLE AND SHARED | `REPRODUCIBILITY_MANIFEST.json`, `SHA256SUMS` (archive), `docs/RELEASE.md` | — | regenerated by `scripts/build_reproducibility_bundle.py` | `scripts/verify_reproducibility_bundle.py` |
| Versioned reproducibility archive | AVAILABLE AND SHARED (release asset) | GitHub Release `v1.0.0-reproducibility` → `VR-FraudNet-reproducibility-1.0.0.tar.gz`; see `docs/RELEASE.md` for the exact status of publication | excludes datasets, weights, indexes, results, secrets by construction | deterministic rebuild from the tagged commit | `scripts/verify_reproducibility_bundle.py <archive>` |

---

## What follows from the table

1. **The method is fully inspectable and executable** for every stage except
   the language-model weights: verifier, schema, grammar, retrieval, edits,
   preprocessing, triage, calibration, statistics.
2. **Exact numerical reproduction of the manuscript is not possible from this
   package**, for reasons that are stated per table in
   `MANUSCRIPT_AUDIT.md` and `KNOWN_LIMITATIONS.md`, chiefly: the Stage 2
   adapter and corpus are absent (L-25), baseline hyper-parameters are
   unpublished (L-18), cost values are in an unavailable supplement (L-26),
   and the D3 Recall@top-1% column is arithmetically unattainable (A-01).
3. **What a reader can do today**: verify the package (Level A/B in
   `REPRODUCIBILITY.md`), obtain the datasets, run every Stage 2-free
   experiment with the manuscript seeds and configurations (Level C/D), and
   compare against the paper with the caveats above stated explicitly.
