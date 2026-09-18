# Retrieval setup

The training-only retrieval of manuscript Section 4.3.4: what is embedded, with
what, how similarity is computed, what may enter the index, what may be
returned for a given query, how to build and verify an index, and where the
implementation departs from the paper's wording.

| Item | Where |
|---|---|
| Canonical configuration | [`retrieval/config/retrieval.yaml`](../retrieval/config/retrieval.yaml) |
| Implementation | [`src/vrfraudnet/retrieval/index.py`](../src/vrfraudnet/retrieval/index.py) — `RetrievalRecord`, `RetrievalIndex`, `default_encoder` |
| Index builder | [`scripts/build_retrieval_index.py`](../scripts/build_retrieval_index.py) |
| Leakage tests | [`tests/test_retrieval_leakage.py`](../tests/test_retrieval_leakage.py), [`tests/test_reviewer_artifacts.py`](../tests/test_reviewer_artifacts.py) |
| Consumer | `scripts/train_stage2_lora.py --retrieval-index`, `RationaleModel` prompts (S4.3.4: demonstrations follow the current evidence) |

---

## 1. Encoder and similarity

| Setting | Value | Provenance |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | MANUSCRIPT S4.3.4 |
| Hub repository | `sentence-transformers/all-MiniLM-L6-v2` | the repository `sentence-transformers` resolves that name to |
| Immutable revision | **not recorded** (L-29); pin with `--encoder-revision`, recorded in the index manifest as given or as `"unpinned"` | — |
| Library / version | sentence-transformers 2.7.0 | MANUSCRIPT S4.8.2 |
| Embedding dimension | 384 (model card); asserted at build time | — |
| Normalisation | L2, applied to index vectors and to the query vector | required for cosine as an inner product |
| Similarity | cosine | MANUSCRIPT S4.3.4 |
| Backend named by the manuscript | FAISS (FAISS-GPU 1.8.0), index type not stated | MANUSCRIPT S4.3.4, S4.8.2 |
| Backend implemented | exact inner product over L2-normalised vectors in `numpy` | see below |
| Index training | none (flat exact index) | — |
| Ranking | descending similarity, `argsort(kind="stable")` | MANUSCRIPT S4.3.4 ("ordered by similarity") |
| Tie behaviour | earlier-inserted record first (stable sort) | ASSUMPTION L-33 |
| Number retrieved | 4; fewer when fewer are eligible; none when none is | MANUSCRIPT S4.3.4 |
| Device | exact search on CPU; the encoder may use a GPU | — |

**Why numpy and not FAISS, and why it does not change the result.** The
manuscript states FAISS with cosine similarity but no index type. A FAISS
`IndexFlatIP` over L2-normalised vectors is exact search: it returns the same
neighbours in the same order as the inner-product computation in
`RetrievalIndex.query`, because both evaluate the same dot products and sort
them. The only way the two could differ is an *approximate* FAISS index (IVF,
HNSW, PQ), which the manuscript does not claim and which would make the
retrieved demonstrations depend on index-construction randomness. This
repository therefore uses exact search and records the choice (L-33). If the
authors confirm an approximate index type, that becomes a documented
deviation to reconcile, not a silent one.

No substitute encoder is used when `sentence-transformers` is unavailable; the
code raises `MissingArtefactError` instead, because a different embedding would
change which demonstrations the language model sees. A deterministic
character-trigram encoder exists for **interface tests only**
(`hashing_encoder`, `--encoder test-hashing`); an index built with it carries
`reproduction_grade: false` in its manifest and the training script warns when
it is used.

## 2. What may enter the index (build-time eligibility)

| Rule | Manuscript | Enforcement |
|---|---|---|
| Training partition only | S4.3.4 | `RetrievalRecord.__post_init__` raises `LeakageError` for any other partition — a non-training record cannot be constructed at all |
| Same dataset | S4.3.4 ("constructed separately for each dataset") | `load_corpus` rejects a mismatched `dataset` field |
| Schema-valid and verifier-accepted rationale | S4.3.4 | `build_retrieval_index.py` runs `DeterministicVerifier.verify` on every target against its own evidence and skips rejections |
| Review-band examples | S4.3.1, S4.3.4 | Stage 1 band from `configs/<dataset>.yaml` (S4.8.1 thresholds); `--include-outside-review-band` overrides for diagnostics and is recorded |
| Excluded: validation, test, transfer, temporal-shock, strict-inductive, adversarial | S4.3.4 | consequence of the partition rule |

## 3. What may be returned (query-time eligibility)

`RetrievalIndex.query(summary, query_timestamp, query_transaction_id, query_entity_ids)`
filters the whole index before ranking:

| Rule | Manuscript | Test |
|---|---|---|
| `record.timestamp < query_timestamp` (strictly earlier) when timestamps exist | S4.3.4 | `test_only_strictly_earlier_examples_are_eligible` |
| `record.record_id != query_transaction_id` | S4.3.4 | `test_shared_transaction_identifier_is_excluded` |
| `record.entity_ids ∩ query_entity_ids = ∅` | S4.3.4 | `test_shared_entity_identifier_is_excluded` |
| at most 4 results | S4.3.4 | `test_at_most_four_examples_are_returned` |
| empty result when nothing is eligible — never a fallback to another partition | S4.3.4 | `test_retrieval_is_omitted_when_no_example_is_eligible` |

## 4. Query construction and use in the prompt

The query text is the **standardised transaction summary** of the current
transaction (`evidence_summary` in the corpus record), encoded with the same
model as the index. Retrieved records are serialised as their target rationale
JSON (grammar key order) and appended to the prompt **after** the current
transaction's evidence, under the heading "DEMONSTRATIONS (form only, not
evidence)" (`build_prompt`). A retrieved example is never citable: consistency
rule CR2 requires every `evidence_id` to resolve in the *current* transaction's
evidence dictionary, so a rationale that cites a demonstration is rejected by
Stage 3.

## 5. Record contents

Each indexed record (`RetrievalRecord`, persisted in `records.json`) holds:
`record_id`, `partition` (always `train`), `timestamp`, `summary`,
`probability_band` (`low` / `review` / `high` from the Stage 1 thresholds),
`rationale` (verdict, probability, claims, evidence references), `entity_ids`.
This matches S4.3.4's list: transaction summary, temporal/graph context (inside
the summary and rationale evidence references), Stage 1 band, rationale,
claims, evidence references.

## 6. Building, storing and verifying an index

```bash
python scripts/build_retrieval_index.py \
  --dataset D1 --config configs/d1.yaml \
  --corpus <rationale_corpus_D1_train.jsonl> --seed 42 \
  --encoder-revision <hf-commit-sha>
```

Output directory `retrieval/index/<dataset>/` (git-ignored) contains

| File | Content |
|---|---|
| `records.json` | every record, sorted keys, ASCII |
| `embeddings.npy` | float32, L2-normalised, one row per record |
| `index_manifest.json` | schema version, record count, dimension, encoder repository and revision, `reproduction_grade`, corpus SHA-256, `retrieval.yaml` SHA-256, review band, drop counts, and the SHA-256 of the two data files |

`RetrievalIndex.load(dir)` recomputes the two data-file checksums and refuses
to load an index whose files no longer match its manifest. The training script
reads the manifest to choose the query encoder (the manuscript encoder at the
recorded revision, or the test encoder with a warning).

No index is shipped: it would be built from the rationale corpus, which is not
available (L-25). The interface is exercised in CI on the synthetic two-record
corpus with the test encoder.

## 7. Requirements

CPU is sufficient for exact search and for encoding a corpus of the sizes
implied by a 5 % review band. `sentence-transformers` (and its `torch`
dependency) are in `requirements-optional.txt`; FAISS is not required by this
implementation.
