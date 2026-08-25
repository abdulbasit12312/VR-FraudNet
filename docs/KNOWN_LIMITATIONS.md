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
**DOCUMENTED DEFAULT.** Five folds are stated; the initial training window is
not. Default: 40% of the ordered sample. This directly affects the per-fold
values in Table 9(b).

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

## Summary

| Category | Count | Effect |
|---|---|---|
| REQUIRES AUTHOR CONFIRMATION | 7 (L-01, L-18, L-24, L-25, L-26, L-27, and A-* findings) | Blocks reproduction of specific tables |
| DOCUMENTED DEFAULT | 21 | Changes numbers; does not block execution |

The practical consequence: **Tables 5, 6, 7, 8, 9(b), 10 and Supplementary
S1–S4 cannot be reproduced from the manuscript as published.** Tables 9(a) and
9(c) can be reproduced in structure once the reference partition (audit A-16)
and shock timestep (L-19) are confirmed.
