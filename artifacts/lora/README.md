# Stage 2 LoRA adapter — availability statement

**Status: the LoRA adapter used to produce the manuscript's results is NOT
PRESENT in the accessible project materials. No adapter is distributed with
this repository, and none has been fabricated to stand in for it.**

This directory is intentionally empty apart from this file. Anything that
appears here in a future release must carry a SHA-256 in
`REPRODUCIBILITY_MANIFEST.json` and a training record produced by
`scripts/train_stage2_lora.py`.

## What was searched

| Location | Result |
|---|---|
| Working tree of this repository | no `adapter_config.json`, no `*.safetensors`, no `*.bin` |
| Full git history (every commit, every blob) | no weight file was ever committed |
| Tags, branches, releases | none existed before the reproducibility release |
| Git LFS | not in use |
| Project working directories accessible to the maintainers (design specifications, planning notes, draft figures) | no adapter, no adapter metadata, no training log, no rationale corpus |

The design documents that were found are **pre-implementation plans** (dated
before the experiments). They mention an intended corpus-construction method,
but they contradict the manuscript on several Stage 2 settings and contain no
execution record. They are therefore not treated as evidence of what was done.
See `docs/KNOWN_LIMITATIONS.md` L-25.

## What this means for reproduction

| Claim | Can the repository support it? |
|---|---|
| The Stage 2 *method* (prompt construction, schema, grammar-masked greedy decoding, stopping rule, failure handling, verifier hand-off) is implemented and tested | **Yes** — `src/vrfraudnet/models/stage2_rationale.py`, `src/vrfraudnet/models/grammar.py`, `tests/test_grammar_masking.py`, `tests/test_reviewer_artifacts.py` |
| The exact LoRA configuration and every training hyper-parameter the manuscript states are available in machine-readable form | **Yes** — `configs/stage2_lora.yaml`, with provenance per value |
| An adapter can be trained under that configuration from a rationale corpus | **Yes** — `scripts/train_stage2_lora.py` (needs the gated base weights, a GPU, and a corpus) |
| An adapter can be validated against the configuration and its checksums recorded | **Yes** — `scripts/validate_stage2_adapter.py` |
| The adapter the authors trained can be downloaded | **No** — not present |
| Exact reproduction of any Stage 2-dependent number (full model rows of Tables 5, 7, 8, 9; verifier-safeguard rates in Supplementary S4) | **No** — blocked by the missing adapter *and* by the unspecified rationale corpus (L-25) |
| The immutable Hugging Face revision of `meta-llama/Meta-Llama-3.1-8B-Instruct` that was used | **Not recorded** (L-29). `configs/stage2_lora.yaml` leaves `base_model.revision: null`; a reconstruction must pin one explicitly with `--base-model-revision` and it is written into the run record |

## How an adapter would be published, if one becomes available

1. Confirm it is the authentic manuscript adapter (training record, date, config
   hash) — a newly trained replacement is labelled `RECONSTRUCTION`, never
   passed off as the original.
2. Run `python scripts/validate_stage2_adapter.py --adapter <dir> --load`.
3. Record `adapter_config.json`, `adapter_model.safetensors`, tokenizer
   additions if any, the base-model revision, the SHA-256 of every file, the
   training seed, and the licence note (Llama 3.1 Community License applies to
   derivatives) in `REPRODUCIBILITY_MANIFEST.json`.
4. Attach the archive to the GitHub Release; the repository itself never
   carries weights (`.gitignore` blocks them and
   `tests/test_repo_hygiene.py::test_no_model_weights_committed` enforces it).

## Base model access

`meta-llama/Meta-Llama-3.1-8B-Instruct` is gated on the Hugging Face Hub under
the Llama 3.1 Community License. It is not redistributed here. Accept the
licence on the Hub and authenticate with `huggingface-cli login` *outside* this
repository; never write a token into any file under version control.
