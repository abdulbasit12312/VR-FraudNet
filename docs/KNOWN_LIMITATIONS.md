# Known limitations

Everything in this repository that **cannot be reconstructed exactly from the
manuscript alone**. Each entry has a stable id (`L-nn`) that is cited from the
configuration files (`_provenance` blocks) and from the code, so any reader can
trace a default back to the fact that the paper did not specify it.

Two categories:

* **REQUIRES AUTHOR CONFIRMATION** — the manuscript is silent or ambiguous and
  the missing information materially changes results.
* **DOCUMENTED DEFAULT** — the manuscript is silent, but the choice is
  conventional, has a bounded effect, and is exposed in configuration.

Separate from this file, `docs/MANUSCRIPT_AUDIT.md` lists points where the
manuscript is *internally inconsistent* rather than merely incomplete.

---

## Missing experimental artefacts

### L-01 — DeLong statistics cannot be recomputed from the paper
**REQUIRES AUTHOR CONFIRMATION.** A DeLong test consumes two score vectors over
one labelled sample. Tables 6(b)–6(d) can therefore only be regenerated from the
per-transaction prediction files, which are not published. Related: audit
finding A-03 (which of the ten seeded prediction vectors was used).

### L-25 — the Stage 2 rationale training corpus is undescribed
**REQUIRES AUTHOR CONFIRMATION.** Section 4.3.1 says the model is fine-tuned on
review-band transactions with target rationales "constructed from transaction,
temporal, and graph evidence observable at the relevant prediction time", but it
never says *how* those target rationales were authored — by rule, by a teacher
model, or by human annotation. Without that, Stage 2 cannot be trained, and
therefore neither can the full model. This is the single largest gap in the
package. Consequences: ablations A1, A3 and A6 cannot be run; every reported
verifier-safeguard rate is unreproducible; and `scripts/train.py` trains what is
effectively the A6 ("w/o Large LLM") configuration.

*Status after the reviewer reproducibility update (2026-09):* **still open.**
The accessible project materials were searched for an execution record. The
only related document is a pre-implementation design specification that
*planned* a teacher-model-generated corpus with a human-adjudicated subset; it
predates the experiments, disagrees with the manuscript on other Stage 2
settings (five retrieved examples and an approximate index versus the
manuscript's four and cosine search; five seeds versus ten), and carries no
teacher-model revision, prompt, decoding settings, filtering log, adjudication
protocol or corpus file. It is not treated as evidence of what was done. The
corpus **format**, loader, S4.3.1 filters, counterfactual generator and training
script are now shipped (`docs/STAGE2_LORA.md`), so an authors' corpus can be
used without code changes. The adapter itself is separately absent (L-31 and
`artifacts/lora/README.md`).

### L-26 — no operational cost values
**REQUIRES AUTHOR CONFIRMATION.** See audit finding A-10. The false-negative
fraud cost, false-positive review cost and true-positive review cost appear only
in Supplementary Table S1, which was not supplied. `configs/costs.yaml` ships
with `null` values and the loader refuses to run until a human fills them in.

### L-27 — Supplementary Tables S1–S4 were not available
**REQUIRES AUTHOR CONFIRMATION.** The cost analysis (S1), subgroup fairness
(S2), efficiency and model size (S3), and calibration/safeguard behaviour (S4)
are all referenced from Section 5.7 but not reproduced in the main text. The
corresponding code paths exist; the target numbers do not.

---

## Under-specified experimental design

### L-02 — D3 validation carve-out
**DOCUMENTED DEFAULT.** Table 1 gives the D3 train/test boundary (0.8 quantile
of `TransactionDT`) but not how validation is carved out of training. The
chronological tail of the training period is used, matching D1's stated rule.
Default: 10%.

### L-03 — D3 expanding-window initial window size
**DOCUMENTED DEFAULT — superseded by the revised manuscript.** The original
submission stated five folds without the initial training window; the default
was 40% of the ordered sample with contiguous test blocks. The revised
manuscript's Table 8(d) now defines the folds exactly: fixed origin, cut at
`TransactionDT` quantiles 0.60 / 0.70 / 0.80 / 0.85 / 0.90, **test-to-end**
(every observation at or after the cut is the test partition), so Fold 3 is the
primary Table 5(c) holdout by construction. `configs/d3.yaml
split.cv_cut_quantiles` carries those values tagged MANUSCRIPT and
`expanding_window_folds(cut_quantiles=...)` implements the construction; the
legacy `cv_initial_fraction` path is retained only for comparison. What
remains undocumented is the within-window validation carve-out (10%, the D1
rule, L-02).

### L-04 — D4 validation timesteps
**DOCUMENTED DEFAULT.** Timesteps 1–34 train and 35–49 test are stated; no
validation range is. Default: the last 15% of the training timestep range.

### L-05 — D5 split proportions and node arrival time
**DOCUMENTED DEFAULT.** "Temporal node-arrival split" is stated without
proportions. Default: 60/20/20 over labelled nodes. When the DGraph-Fin release
carries no node timestamp, arrival time is derived as the earliest incident edge
time.

### L-11 — threshold-selection criterion
**DOCUMENTED DEFAULT.** Section 4.8.1 says thresholds were chosen to prioritise
"validation AUPRC and expected operational cost while limiting the proportion of
transactions sent to the language-model pathway". AUPRC is threshold-free, so it
cannot discriminate between threshold pairs, and the cost values are missing
(L-26). The implemented criterion maximises fraud captured inside the review
band per unit of escalation, subject to the 5% escalation cap of Figure 2. The
manuscript's own per-dataset pairs remain available via
`--use-manuscript-thresholds`.

### L-19 — D4 shock timestep
**DOCUMENTED DEFAULT.** Section 5.5 refers to a "pre/post dark-market-shutdown
shock decomposition" without naming the timestep. The Elliptic literature places
the shutdown at timestep 43; that value is the default and must be passed
explicitly via `--shock-timestep`.

### L-22 — which D1 attributes are "protected"
**DOCUMENTED DEFAULT.** Table 2 says "exclude protected attrs" for D1 without
enumerating them. Supplementary Table S2 refers to a protected *age* group, so
`customer_age` is excluded by default. The list is configurable. See also audit
finding A-08 on the resulting feature-count contradiction.

---

## Under-specified model configuration

### L-08 — Beta-kernel order
**DOCUMENTED DEFAULT.** "Four Beta-kernel spectral filters" fixes `K = 4` but
not the wavelet order. The BWGNN construction `W_{p,q} ∝ (L/2)^p (I − L/2)^q`
with `p + q = 4` and `p = 1..4` is used.

### L-09 — Stage 0 MLP width and dropout
**DOCUMENTED DEFAULT.** A "two-layer multilayer perceptron with GELU" is stated;
its hidden width is not. Default: 64. Dropout default: 0.0.

### L-10 — time-embedding parameterisation
**DOCUMENTED DEFAULT.** A 32-dimensional time embedding is stated; how time is
embedded is not. A parameter-free sinusoidal embedding is used, so the only
learned component of the temporal pathway is the MLP the manuscript describes.

### L-12 — class-balanced reweighting hyperparameter
**DOCUMENTED DEFAULT.** Table 2 says "focal + class-balanced" without the
effective-number `beta`. Default: 0.9999.

### L-20 — focal-loss gamma
**DOCUMENTED DEFAULT.** The focusing parameter is not stated. Default: 2.0, the
value from the original focal-loss paper.

### L-18 — baseline hyperparameters
**REQUIRES AUTHOR CONFIRMATION.** The manuscript names nine baselines and
publishes hyperparameters for none of them. Every setting in
`configs/baselines.yaml` was chosen by this repository. **No comparison against
manuscript Table 5 baseline numbers is valid until these are supplied.** The
TabTransformer implementation additionally diverges from the original
architecture, because after the manuscript's preprocessing every column reaching
the model is numeric and the original design attends over categorical
embeddings.

### L-24 — what replaces the triage signal in ablation A2
**REQUIRES AUTHOR CONFIRMATION.** "w/o Triage Tθ" removes the Stage 1
probability, but the Stage 4 mixer takes that probability as an input. The
manuscript does not say what the mixer consumes in this variant. Default: the
rationale-side probability alone.

---

## Under-specified rationale, verifier and decoding details

### L-07 — grammar-masking backend
**DOCUMENTED DEFAULT.** Grammar-masked decoding is described but no library is
named. This repository ships its own incremental parser
(`src/vrfraudnet/models/grammar.py`) rather than attributing the work to a
third-party backend it cannot verify was used.

### L-15 — object key order in the decoding grammar
**DOCUMENTED DEFAULT.** The grammar fixes key order. That strictly narrows the
accepted language; the manuscript does not say whether key order was
constrained.

### L-13 — floating-point tolerance for `eq`/`ne` claims
**DOCUMENTED DEFAULT.** Default: 1e-9, declared in
`verifier/rules/claim_primitives.yaml`.

### L-14 — risk direction of a verified claim
**DOCUMENTED DEFAULT.** Consistency rule CR3 requires knowing whether a verified
claim raises or lowers suspicion. The manuscript states that a verdict must be
"logically supported by verified claims" but gives no mapping. The mapping is
declared in `verifier/rules/claim_primitives.yaml` so it is auditable and
overridable.

### L-16 — verifier-aware conformal penalty
**DOCUMENTED DEFAULT.** Section 4.5 says the verifier-aware variant "assigns
greater uncertainty to verifier-rejected cases" without quantifying it. Default
score inflation: 0.10.

### L-06 — target-encoding smoothing weight
**DOCUMENTED DEFAULT.** "Smoothed target-encode" is stated; the smoothing weight
is not. Default: 20.

### L-23 — hash bucket count
**DOCUMENTED DEFAULT.** "64-bit hash" describes the hash function, not the
output cardinality. Default: 1024 buckets.

### L-17 — maximum admissible execution delay
**DOCUMENTED DEFAULT.** The delay edit family must "preserve chronological
validity" but no maximum delay is given. Default: 24 hours.

---

## Environment and reproducibility limits

### L-21 — thread count
**DOCUMENTED DEFAULT.** LightGBM is deterministic only for a fixed
`num_threads`; the manuscript does not report one. Default: 4. Changing it
changes histogram construction order and therefore the model.

### L-28 — hardware
The manuscript's results were produced on an AMD EPYC 7543, 256 GB RAM and one
NVIDIA A100 80GB. GPU floating-point reductions are not bitwise reproducible
across architectures, so exact numerical reproduction on different hardware
should not be expected even with identical seeds.

---

## Added by the reviewer reproducibility update

### L-29 — immutable revisions of the third-party models are not recorded
**REQUIRES AUTHOR CONFIRMATION.** The manuscript names
`Meta-Llama-3.1-8B-Instruct` (S4.3.1) and `all-MiniLM-L6-v2` (S4.3.4) and the
library versions (S4.8.2) but not the Hugging Face commit of either model. Both
repositories have received revisions since release (tokenizer and generation
configuration changes are documented on the Llama model card), so the
identifier alone does not fix the bytes. `configs/stage2_lora.yaml
base_model.revision`, `base_model.tokenizer_revision` and
`retrieval/config/retrieval.yaml encoder.revision` are therefore `null`; the
training and index-building scripts accept `--base-model-revision` /
`--encoder-revision`, and record `"unpinned"` when none is given, so the gap
is visible in every run record.

### L-30 — Stage 2 optimisation details the manuscript omits
**DOCUMENTED DEFAULT / REQUIRES AUTHOR CONFIRMATION.** Stated: AdamW, learning
rate 2e-4, weight decay 0.01, linear warm-up over the first 5% of steps, three
epochs, bfloat16, effective batch size 32 by gradient accumulation, checkpoint
by validation loss (S4.3.1, S4.8.1). Not stated: per-device batch size and
accumulation steps (only their product); the learning-rate schedule after
warm-up; gradient clipping for Stage 2 (1.0 is stated for Stage 0 only); AdamW
betas and epsilon; LoRA bias policy and `modules_to_save`; whether one adapter
was trained per seed or shared across seeds. `scripts/train_stage2_lora.py`
refuses to run without `--per-device-batch-size`, uses `transformers`' linear
schedule (warm-up then linear decay) and `bias="none"`, and records every one
of these choices in the run record.

### L-31 — Stage 2 adapter save format, and the adapter's absence
**DOCUMENTED DEFAULT** for the format: the manuscript does not state one;
safetensors (the PEFT default) is used and checked by
`scripts/validate_stage2_adapter.py`. **NOT AVAILABLE** for the artefact: The exact trained LoRA adapter checkpoint used in the original experiments is not included because a standalone archival copy of the fitted adapter was not retained as a release-ready artifact during the original experimental workflow. The repository provides the complete LoRA training configuration and implementation, including the base-model specification, target modules, rank, scaling factor, dropout, optimizer settings, learning rate, training epochs, sequence lengths, and predefined seeds, allowing the adaptation procedure to be rerun. Exact checkpoint-level reproduction of the original Stage 2 model is therefore not possible from the currently released materials.
Searched: the working tree, the full git history, every branch and tag, Git LFS,
and the maintainers' project directories. Exact reproduction of every Stage 2-dependent number is blocked by
this independently of L-25. Full statement: `artifacts/lora/README.md`.

### L-32 — counterfactual target construction
**DOCUMENTED DEFAULT.** S4.6.3 says that after one admissible evidence element
is modified, "claims directly affected by the change are updated or removed,
whereas claims supported by unchanged evidence retain their original structured
form", and that the rationale-side probability is "regularised", without giving
the rule or the functional form of the loss. `scripts/train_stage2_lora.py`
re-verifies the original claims against the modified evidence, keeps those that
still verify, removes those that do not, discards the pair when no claim
survives or the verdict is no longer supported, and applies the counterfactual
example as an additional supervised example with loss weight 0.25 (the stated
weight). The probability-stability penalty
(`losses/counterfactual.py::rationale_stability_penalty`) is available as a
reported metric but is not back-propagated, because doing so requires decoding
inside the training loop and the manuscript does not describe that.

### L-33 — retrieval index type and tie-breaking
**DOCUMENTED DEFAULT.** S4.3.4 states FAISS with cosine similarity but no index
type. Exact search (numpy inner product over L2-normalised vectors, equivalent
to `IndexFlatIP`) is used; an approximate index would make the retrieved
demonstrations depend on index-construction randomness. Ties are broken by
insertion order (stable sort). See `docs/RETRIEVAL_SETUP.md` §1.

### L-34 — adversarial training loop is not wired into Stage 1 training
**IMPLEMENTATION GAP (this package, not the manuscript).** S4.6.2 describes
training-time adversarial augmentation (budget 2, 25% of fraudulent examples
per epoch, keep the max-loss valid candidate, weight 0.50). The edit functions
and the configuration exist (`docs/EDIT_GENERATORS.md`), but `scripts/train.py`
does not run the selection loop, so the Stage 1 model it produces corresponds
to ablation A4 ("w/o adversarial training") rather than the full model. This is
stated so that no number from `scripts/train.py` is compared against a
full-model row without that caveat.

---

## Summary

| Category | Count | Effect |
|---|---|---|
| REQUIRES AUTHOR CONFIRMATION | 8 (L-01, L-18, L-24, L-25, L-26, L-27, L-29, L-30 in part, and A-* findings) | Blocks reproduction of specific tables |
| DOCUMENTED DEFAULT | 24 (incl. L-31 format, L-32, L-33) | Changes numbers; does not block execution |
| NOT AVAILABLE | 1 (L-31: the Stage 2 adapter) | Blocks every Stage 2-dependent number |
| IMPLEMENTATION GAP | 1 (L-34) | `scripts/train.py` produces the A4 configuration of Stage 1 |

The practical consequence: **Tables 5, 6, 7, 8, 9(b), 10 and Supplementary
S1–S4 cannot be reproduced from the manuscript as published.** Tables 9(a) and
9(c) can be reproduced in structure once the reference partition (audit A-16)
and shock timestep (L-19) are confirmed.
