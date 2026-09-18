# Grammar-masked decoding

How the schema-constrained decoding of manuscript Section 4.3.3 is
implemented, how it relates to the JSON schema and the verifier, what it
enforces, what it does not, and how to check it.

| Item | Where |
|---|---|
| Canonical implementation | [`src/vrfraudnet/models/grammar.py`](../src/vrfraudnet/models/grammar.py) — `GrammarSpec`, `RationaleGrammar`, `ParseState` |
| Schema it is derived from | [`schemas/rationale_schema.json`](../schemas/rationale_schema.json) |
| Decoding loop that applies the mask | [`src/vrfraudnet/models/stage2_rationale.py`](../src/vrfraudnet/models/stage2_rationale.py) — `RationaleModel._decode`, `finalise_generation` |
| Tests | [`tests/test_grammar_masking.py`](../tests/test_grammar_masking.py), [`tests/test_reviewer_artifacts.py`](../tests/test_reviewer_artifacts.py) |
| Runnable check | `python scripts/validate_rationale_schema.py` (grammar section) |

---

## 1. Kind of grammar

**Hand-written, incremental, schema-derived.** `GrammarSpec.from_schema` reads
the enumerations and bounds (verdict labels, claim types, operators,
`minItems`/`maxItems` of `claims` and `evidence_references`) from the JSON
schema at construction time; nothing is duplicated in code, so the grammar and
the schema cannot drift on those values. The *structure* (which keys, in which
order, with which value shapes) is encoded as recursive-descent productions in
`RationaleGrammar`.

No third-party grammar library is used. The manuscript names none (L-07), so
the repository ships its own parser rather than attributing the work to a
backend it cannot verify was used.

## 2. The three parse states

`RationaleGrammar.parse(prefix)` classifies any generated prefix as

* `INVALID` — no extension of the prefix can become a schema-valid object;
* `PARTIAL` — the prefix is a proper prefix of at least one valid object;
* `COMPLETE` — the prefix is exactly one valid object (trailing whitespace only).

The token mask at a decoding step is then

```python
mask = [grammar.parse(generated + token) is not ParseState.INVALID for token in vocabulary]
```

(`RationaleGrammar.token_mask`). A token is admissible when the extended prefix
is still `PARTIAL` or `COMPLETE`. `RationaleGrammar.allowed_characters(prefix)`
is the same question asked over single characters and is what the tests use.

## 3. Key order (L-15)

The grammar fixes object key order:

* top level: `verdict`, `rationale_probability`, `claims`, `evidence_references`;
* inside a claim: `type`, `field`, `operator`, `value`, `evidence_id`;
* inside a graph-path value: `target`, then optional `max_hops`.

A fixed order strictly narrows the accepted language (every accepted string is
also schema-valid; some schema-valid strings with permuted keys are not
accepted). The manuscript does not say whether key order was constrained; this
is ASSUMPTION L-15 and the training script serialises targets in exactly this
order (`serialise_rationale`).

## 4. What is enforced during generation

| Constraint | Source | Enforced by |
|---|---|---|
| JSON syntax (braces, brackets, commas, quotes) | S4.3.3 | productions in `_object`, `_claim_object`, `_array` |
| Only the four top-level keys, only the five claim keys | S4.3.2 "unrestricted free-form fields ... are excluded" | `_exact_string` |
| `verdict ∈ {fraud, legitimate, uncertain}` | S4.3.2 | `_enum_string` over `GrammarSpec.verdicts` |
| `rationale_probability ∈ [0, 1]` | S4.3.2 | `_number(low=0, high=1)`; a prefix that can no longer land inside the range (e.g. `-0.1` or `1.5`) is `INVALID` immediately |
| claim `type` in the five primitives | S4.3.2 | `_enum_string` over `GrammarSpec.claim_types` |
| claim `operator` in the schema enum | S4.3.2 | `_enum_string` over `GrammarSpec.operators` |
| `1 ≤ len(claims) ≤ 8` | S4.3.2 | `_array(min_items, max_items)` from the schema |
| `1 ≤ len(evidence_references) ≤ 32` | schema | same |
| `max_hops ∈ [1, 4]` | schema / rule file | `_number(1, 4)` in `_graph_target_object` |
| identifier alphabet `[A-Za-z0-9_.:- ]`, ≤ 128 chars | schema patterns | `_any_string` |
| polymorphic `value` (number, string, boolean, array ≤ 32, `{target,max_hops}`) | schema `oneOf` | `_claim_value` |

## 5. What is deliberately NOT enforced by the grammar

* Operator-to-type compatibility (`lt` on a `set_membership` claim is
  grammar-valid). The rule file `verifier/rules/claim_primitives.yaml` owns
  that mapping and the Stage 3 verifier rejects the claim. Encoding it in the
  grammar would duplicate the rule set.
* Whether an `evidence_id` exists, whether a numeric assertion is true, whether
  the verdict is supported. Manuscript S4.3.3: "Grammar masking controls output
  structure but does not establish whether a generated claim is factually
  correct." That is Stage 3.

## 6. Decoding settings actually used

From `configs/stage2_lora.yaml` (`decoding`) and `RationaleModel._decode`:

| Setting | Value |
|---|---|
| Strategy | greedy (`argmax` over masked logits) |
| Temperature | 0 |
| Sampling | disabled |
| Beams | 1 |
| Max output tokens | 384 |
| Stop condition | `parse(generated) is COMPLETE`, or the token cap |
| Incomplete / unparseable output | `RationaleOutput.parsed = None`, `truncated = True` at the cap; treated as a verification failure downstream (S4.3.3, S4.3.5) |

The mask is applied to logits as `masked_fill(~mask, -inf)` before `argmax`,
so an inadmissible token can never be selected; the mask is recomputed from
the full generated prefix at every step.

## 7. Failure handling

`finalise_generation(text, truncated)`:

* truncated → `parse_error = "generation reached the 384-token output limit ..."`;
* not truncated but `json.loads` fails → `parse_error = "JSON decode error: ..."`
  (only possible if the mask was bypassed, since a `COMPLETE` prefix is valid
  JSON by construction);
* top-level value not an object → `parse_error` set.

`RationaleOutput.schema_valid_generation` is the quantity reported as the
"schema-valid generation rate", the secondary checkpoint-selection criterion
of S4.3.1. A failed output is handed to Stage 3 as a rejection; the
rationale-side probability is excluded from Stage 4 and the transaction is
escalated (S4.3.5).

## 8. Known deviations from the manuscript

* **Key order is fixed** (L-15) — the manuscript is silent.
* **Own parser rather than a named library** (L-07) — the manuscript is silent.
* **Number literals**: the grammar accepts decimal literals with an optional
  leading minus and no exponent. The schema permits any JSON number; exponent
  forms (`1e-3`) are excluded because a bounded probability never needs them
  and admitting them complicates the prefix range check. This narrows the
  accepted language without excluding any value the schema allows in
  `[0, 1]` at ordinary precision.
* **Tokenisation**: masking works on the *decoded surface form* of each
  vocabulary token (`_decoded_vocabulary`). A token whose surface form spans a
  grammar boundary (for example `":"`) is admissible whenever the whole span is
  admissible, which is the standard construction; no claim is made about
  byte-level tokens that decode to partial UTF-8 sequences, which are rejected.

## 9. Runnable examples

```bash
python scripts/validate_rationale_schema.py
```

covers, against the shipped schema: a complete schema-valid rationale
(`COMPLETE`), a valid prefix (`PARTIAL`), an invalid enum value, an illegal
key, an illegal operator, and an incomplete object. The same cases are unit
tests in `tests/test_reviewer_artifacts.py::test_grammar_reviewer_cases` and
the broader property tests in `tests/test_grammar_masking.py` (every prefix of
a valid object is `PARTIAL`, more than eight claims is rejected, the token mask
blocks violating tokens, `max_hops` bound, and so on).
