# VR-FraudNet v1.0.0-manuscript-revision

Reproducibility snapshot associated with the revised manuscript.

**Manuscript:** *Grounding language models with deterministic verifiers for fraud detection* (revised submission).

**Purpose.** Reviewer-requested reproducibility release: direct access to the deterministic verifier rules, the rationale JSON schema, the grammar-constrained decoder, the Stage 2 LoRA configuration and adapter-availability statement, the retrieval configuration and index builder, the adversarial edit generators, every experiment configuration, and the exact manuscript seed set, packaged as a versioned archive with per-file SHA-256 checksums and a manifest.

**Commit.** The commit this tag points at: `git rev-list -n 1 v1.0.0-manuscript-revision`. The `git.commit` field of the attached `REPRODUCIBILITY_MANIFEST.json` records the same value.

## Reviewer-requested artefacts

| Requested | In this release |
|---|---|
| Verifier rules | `verifier/rules/claim_primitives.yaml` + `src/vrfraudnet/verifier/` |
| JSON schema | `schemas/rationale_schema.json` |
| Decoding grammar | `src/vrfraudnet/models/grammar.py`, `docs/DECODING_GRAMMAR.md` |
| LoRA adapters | configuration `configs/stage2_lora.yaml`, training `scripts/train_stage2_lora.py`, validation `scripts/validate_stage2_adapter.py`, guide `docs/STAGE2_LORA.md`; **weights NOT AVAILABLE** — `artifacts/lora/README.md` |
| Retrieval setup | `retrieval/config/retrieval.yaml`, `src/vrfraudnet/retrieval/index.py`, `scripts/build_retrieval_index.py`, `docs/RETRIEVAL_SETUP.md` |
| Edit generators | `adversarial/families/edit_families.yaml`, `src/vrfraudnet/adversarial/`, `docs/EDIT_GENERATORS.md` |
| Configuration files | `configs/*.yaml`, every value provenance-tagged |
| Seeds | `configs/manuscript_seeds.yaml` (= `vrfraudnet.seeds.MANUSCRIPT_SEEDS`) |
| Versioned archive | `VR-FraudNet-reproducibility-1.0.0.tar.gz` + `SHA256SUMS` + `REPRODUCIBILITY_MANIFEST.json` (attached) |
| Materials that cannot be shared | `docs/ARTIFACT_AVAILABILITY.md`; summary below |

## Install and verify

```bash
tar -xzf VR-FraudNet-reproducibility-1.0.0.tar.gz && cd VR-FraudNet-reproducibility-1.0.0
sha256sum -c SHA256SUMS                      # or: python scripts/verify_reproducibility_bundle.py <archive>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest jsonschema
python -m pytest -q
python scripts/validate_rationale_schema.py
python scripts/audit_reviewer_artifacts.py
```

## One run

```bash
python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml
python scripts/train.py        --dataset D1 --config configs/d1.yaml --seed 42
python scripts/evaluate.py     --dataset D1 --config configs/d1.yaml --seed 42
python scripts/calibrate.py    --dataset D1 --config configs/d1.yaml --seed 42
```

Full sweep: `bash reproduce_all.sh` (prints the plan), `bash reproduce_all.sh --execute`.

## Unavailable materials and reasons

* **Stage 2 LoRA adapter used for the manuscript.** The exact trained LoRA adapter checkpoint used in the original experiments is not included because a standalone archival copy of the fitted adapter was not retained as a release-ready artifact during the original experimental workflow. The repository provides the complete LoRA training configuration and implementation, including the base-model specification, target modules, rank, scaling factor, dropout, optimizer settings, learning rate, training epochs, sequence lengths, and predefined seeds, allowing the adaptation procedure to be rerun. Exact checkpoint-level reproduction of the original Stage 2 model is therefore not possible from the currently released materials. Nothing has been fabricated or trained as a stand-in. (L-31)
* **Stage 2 rationale training corpus** — absent, and the manuscript does not state how target rationales were authored (rule, teacher model or human). The corpus format, loader, filters and training script are shipped so an authors' corpus drops in unchanged. (L-25)
* **Immutable revisions of `meta-llama/Meta-Llama-3.1-8B-Instruct` and `sentence-transformers/all-MiniLM-L6-v2`** — not recorded by the manuscript; configuration fields are `null` and the scripts record whatever the operator pins. (L-29)
* **Base-model and encoder weights** — third-party licences (Llama 3.1 Community License; Apache-2.0); obtained from the Hugging Face Hub, never bundled.
* **Prediction vectors, result files, Supplementary Tables S1–S4, cost values** — not in the accessible project materials. (L-26, L-27)
* **Adversarial-training loop** — configuration and edit functions shipped; the training-time selection loop is not wired into `scripts/train.py`. (L-34)

## Dataset access

D1–D5 are not redistributed. Provider links, files, licences and the checksum procedure: `docs/DATA_AVAILABILITY.md` (Kaggle CC BY-NC-SA / CDLA-Sharing / competition rules; Elliptic++ repository; DGraph-Fin research agreement).

## Manuscript consistency

`docs/MANUSCRIPT_AUDIT.md` records every internal inconsistency found, with a reconciliation section for the revised manuscript (`python scripts/audit_manuscript.py --revision revised`). Findings A-01, A-05, A-08/A-09, A-10, A-16, A-17 and the new A-22–A-24 remain open in the revised text.

## Citation

Cite the manuscript for the method and this release for the implementation: `CITATION.cff` (software v1.0.0, tag `v1.0.0-manuscript-revision`).

## Release status

**PREPARED, NOT PUBLISHED.** This file is updated to *published*, with the archive SHA-256 and the release URL, only after `gh release create` has succeeded and the assets are visible on GitHub.
