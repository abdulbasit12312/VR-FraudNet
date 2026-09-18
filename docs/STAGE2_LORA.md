# Stage 2: LoRA configuration, training, loading and adapter availability

Everything a reader needs to locate, inspect and execute the Stage 2
rationale-language-model pathway of manuscript Section 4.3, and an exact
statement of what cannot be reproduced and why.

| Item | Where |
|---|---|
| Canonical configuration (every hyper-parameter, with provenance) | [`configs/stage2_lora.yaml`](../configs/stage2_lora.yaml) |
| Runtime implementation (prompt, decoding loop, failure handling, adapter loading) | [`src/vrfraudnet/models/stage2_rationale.py`](../src/vrfraudnet/models/stage2_rationale.py) |
| Grammar-masked decoder | [`src/vrfraudnet/models/grammar.py`](../src/vrfraudnet/models/grammar.py), documented in [`DECODING_GRAMMAR.md`](DECODING_GRAMMAR.md) |
| Training script | [`scripts/train_stage2_lora.py`](../scripts/train_stage2_lora.py) |
| Adapter validation / loading smoke test | [`scripts/validate_stage2_adapter.py`](../scripts/validate_stage2_adapter.py) |
| Counterfactual objective (S4.6.3) | [`src/vrfraudnet/losses/counterfactual.py`](../src/vrfraudnet/losses/counterfactual.py) |
| Adapter weights | **not available** — [`artifacts/lora/README.md`](../artifacts/lora/README.md) |
| Rationale training corpus | **not available, construction unspecified** — [`KNOWN_LIMITATIONS.md` L-25](KNOWN_LIMITATIONS.md) |
| Corpus format (for a reconstruction) | Section 4 below; synthetic illustration in [`examples/stage2_corpus_example.jsonl`](../examples/stage2_corpus_example.jsonl) |

---

## 1. Configuration, value by value

All values below are taken from `configs/stage2_lora.yaml`. The `Provenance`
column is the tag recorded in that file. **Null means the manuscript is silent
and no project artefact records the value**; the training script refuses to
fill such a value in silently.

| Setting | Value | Provenance |
|---|---|---|
| Base model | `meta-llama/Meta-Llama-3.1-8B-Instruct` | MANUSCRIPT S4.3.1, Table 3 |
| Base-model immutable revision (HF commit) | **null** | not recorded — L-29 |
| Tokenizer | same repository | DERIVED |
| Tokenizer revision | **null** | not recorded — L-29 |
| Base weights | frozen | MANUSCRIPT S4.3.1 |
| PEFT implementation / version | `peft` 0.10.0, `transformers` 4.40.1, `torch` 2.2.1 | MANUSCRIPT S4.8.2 |
| LoRA rank `r` | 16 | MANUSCRIPT S4.3.1, S4.8.1 |
| `lora_alpha` | 32 | MANUSCRIPT S4.3.1, S4.8.1 |
| `lora_dropout` | 0.05 | MANUSCRIPT S4.3.1, S4.8.1 |
| Target modules | `q_proj, k_proj, v_proj, o_proj` | MANUSCRIPT S4.3.1 ("query, key, value, and output projection layers of the self-attention modules") |
| `modules_to_save` | **null** (none stated) | L-30 |
| Bias policy | `none` | ASSUMPTION L-30 |
| Task type | `CAUSAL_LM` | DERIVED |
| dtype | bfloat16 | MANUSCRIPT S4.3.1, S4.8.2 |
| Quantisation | none | DERIVED (none described) |
| Max input / output tokens | 1024 / 384 | MANUSCRIPT S4.3.1 |
| Optimiser | AdamW | MANUSCRIPT S4.3.1 |
| Learning rate | 2e-4 | MANUSCRIPT S4.8.1 (S4.3.1 typesets "2 x10^4"; audit A-11) |
| Weight decay | 0.01 | MANUSCRIPT S4.3.1 |
| Scheduler | linear warm-up over the first 5 % of steps | MANUSCRIPT S4.3.1 |
| Post-warm-up decay | **null** | L-30 (the script uses `transformers`' `linear` schedule, i.e. linear decay to zero, and records it) |
| Epochs | 3 | MANUSCRIPT S4.3.1, S4.8.1 |
| Effective batch size | 32 via gradient accumulation | MANUSCRIPT S4.3.1 |
| Per-device batch size / accumulation steps | **null** — must be passed as `--per-device-batch-size`; accumulation is derived and both are recorded | L-30 |
| Gradient clipping | **null** — `--gradient-clip-norm` optional, recorded | L-30 |
| AdamW betas / epsilon | **null** | L-30 |
| Counterfactual loss weight | 0.25, selected from {0.25, 0.50, 0.75, 1.00} | MANUSCRIPT S4.6.3, S4.6.4 |
| Checkpoint selection | lowest validation loss; schema-valid generation rate and verifier acceptance rate as secondary criteria | MANUSCRIPT S4.3.1, S4.8.1 |
| Seeds | the ten of `configs/manuscript_seeds.yaml` | MANUSCRIPT S4.8.1 |
| One adapter per seed, or one shared? | **null** | L-30 |
| Training examples | review-band transactions (`T_low ≤ P_t ≤ T_high`) of the **train** partition only | MANUSCRIPT S4.3.1 |
| Target filtering | malformed, inconsistent or verifier-rejected targets excluded | MANUSCRIPT S4.3.1 — implemented in `filter_training_targets` and `apply_manuscript_filters` |
| Target construction (who wrote the rationales) | **null** | **L-25 — REQUIRES AUTHOR CONFIRMATION** |
| Decoding | greedy, temperature 0, sampling off, grammar-masked, stop at complete object or 384 tokens; incomplete = verification failure | MANUSCRIPT S4.3.3 |
| Retrieved demonstrations | 4, training-only, see [`RETRIEVAL_SETUP.md`](RETRIEVAL_SETUP.md) | MANUSCRIPT S4.3.4 |
| Adapter save format | safetensors | ASSUMPTION L-31 |

`tests/test_reviewer_artifacts.py` checks that the dataclass defaults in
`stage2_rationale.py`, the `stage2` block of `configs/base.yaml`, and this
canonical file agree, so there is one Stage 2 configuration, not three.

---

## 2. Training

```bash
python scripts/train_stage2_lora.py \
  --dataset D1 --config configs/d1.yaml \
  --stage2-config configs/stage2_lora.yaml \
  --corpus <rationale_corpus_D1_train.jsonl> \
  --validation-corpus <rationale_corpus_D1_validation.jsonl> \
  --seed 42 --per-device-batch-size 4 \
  --base-model-revision <hf-commit-sha>
```

The script, in order:

1. loads the two configurations and seeds every RNG;
2. reads the corpus and **refuses any record whose `partition` is not `train`**
   (`LeakageError`, exit 3) or whose `dataset` differs from `--dataset`;
3. applies the S4.3.1 filters — review band (`T_low ≤ P_t ≤ T_high` from
   `configs/<dataset>.yaml`, S4.8.1), verifier acceptance against the record's
   own evidence, and grammar serialisability — and counts every drop;
4. generates one label-preserving counterfactual per surviving record from the
   training partition (S4.6.3; operationalisation stated as L-32);
5. builds prompts with `build_prompt` (schema hint → current evidence → triage
   probability → timestamp → retrieved demonstrations), optionally drawing
   demonstrations from an index built by `scripts/build_retrieval_index.py`;
6. writes `results/stage2/<dataset>/train_plan_seed<seed>.json` — corpus
   checksum, kept/dropped counts, every hyper-parameter, base-model revision
   label — **before** touching any weight;
7. with `--dry-run`, stops here (exit 0). This is what CI runs, on the
   synthetic two-record example corpus, to prove the interface;
8. otherwise fine-tunes with PEFT LoRA and a per-example loss weight of 0.25
   for counterfactual examples, selects the checkpoint by validation loss, and
   saves the adapter with a training record containing the SHA-256 of every
   file written.

An adapter produced this way is a **reconstruction under the documented
configuration**. Its record says `RECONSTRUCTION ... NOT the manuscript
adapter`. It is not the manuscript's adapter and must not be described as such.

Requirements for a real run: access to the gated base weights (Llama 3.1
Community License, accepted on the Hugging Face Hub, authenticated outside this
repository), one GPU with ≥ 40 GB memory (`REPRODUCIBILITY.md` §2), and
`pip install -r requirements-optional.txt`.

---

## 3. Validating and loading an adapter

```bash
python scripts/validate_stage2_adapter.py --adapter <dir> --config configs/stage2_lora.yaml
python scripts/validate_stage2_adapter.py --adapter <dir> --load      # real load through PEFT
```

Exit 2 when the directory does not exist (no adapter is shipped), exit 1 on a
metadata mismatch (rank, alpha, dropout, target modules, bias, base-model
identifier) or a load failure, exit 0 when the adapter is structurally
consistent with the canonical configuration. Every file's SHA-256 is printed.

`RationaleModel.load()` performs the same metadata check before loading and
raises `ConfigurationError` on a mismatch, `MissingArtefactError` when the path
does not exist. `tests/test_reviewer_artifacts.py` covers both.

**A passing validation does not establish authenticity.** The manuscript
adapter is absent, so there is no reference checksum.

---

## 4. Corpus format

One JSON object per line. Every field is required except `entity_ids` and
`_note`.

```json
{
  "record_id": "TX-000123",
  "dataset": "D1",
  "partition": "train",
  "timestamp": 1767225600.0,
  "triage_probability": 0.62,
  "evidence": { "transaction_id": "...", "timestamp": ..., "numeric": {...},
                "categorical": {...}, "entities": {...},
                "temporal_aggregates": {...}, "graph": {...} },
  "evidence_summary": "standardised transaction summary used for retrieval and in the prompt",
  "target": { "verdict": "...", "rationale_probability": 0.0,
              "claims": [...], "evidence_references": [...] },
  "entity_ids": ["ACC-..."]
}
```

* `evidence` uses the layout of `examples/evidence_example.json`; it is parsed
  by `TransactionEvidence.from_json`, which keeps the typed field groups
  explicit so a numeric identifier is never treated as a quantity.
* `target` must conform to `schemas/rationale_schema.json` and must be accepted
  by the verifier against `evidence`, otherwise it is dropped (S4.3.1).
* `partition` must be `train` for training and for the retrieval index, and
  `validation` for `--validation-corpus`. Test, temporal-shock,
  strict-inductive and adversarial records can never be loaded by these
  scripts.

`examples/stage2_corpus_example.jsonl` contains two **synthetic** records that
illustrate the format. They are not manuscript training examples and say so in
their `_note` field.

---

## 5. What the manuscript does not say about the corpus (L-25)

Section 4.3.1 states that target rationales were "constructed from
transaction, temporal, and graph evidence observable at the relevant
prediction time" and filtered by the verifier. It does not state whether they
were written by rules, by a teacher model, or by human annotators, nor how many
there were per dataset. Without that, the corpus cannot be rebuilt and Stage 2
cannot be retrained to the manuscript's specification.

The only related material found in the project archive is a pre-implementation
design specification that *planned* a teacher-model-generated corpus with a
human-adjudicated subset. That document predates the experiments, disagrees
with the manuscript on other Stage 2 settings (number of retrieved examples,
index type, number of seeds), and comes with no execution record: no teacher
model revision, no prompt template, no decoding settings, no filtering log, no
adjudication protocol, no corpus file. It is therefore **not** treated as
authoritative, and the status of L-25 remains **REQUIRES AUTHOR CONFIRMATION**.

If the authors supply the construction procedure, the corpus (or its
non-sensitive metadata and generator), and the adapter, all three slot into the
interfaces above without code changes: `--corpus`, `--retrieval-index`,
`--adapter`.
