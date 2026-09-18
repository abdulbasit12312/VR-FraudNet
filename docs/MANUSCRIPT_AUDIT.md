# Manuscript-to-code audit

Findings that arose while implementing the manuscript. These are **internal
inconsistencies and specification errors**, distinct from the merely missing
details catalogued in `docs/KNOWN_LIMITATIONS.md`.

Every arithmetic claim below is reproducible:

```bash
python scripts/audit_manuscript.py          # human-readable
python scripts/audit_manuscript.py --json   # machine-readable
```

That script quotes published numbers **as audit inputs only**. Nothing in the
experimental pipeline reads them.

Severity key:

* **BLOCKING** — the experiment cannot be implemented as described; this
  repository refuses to run it and raises `ManuscriptInconsistencyError`.
* **MAJOR** — reviewer-visible; the reported number is unattainable, unverifiable
  or contradicted elsewhere in the paper.
* **MINOR** — a typographic or labelling error with no effect on conclusions.

---

## A-01 — Recall@top-1% on D3 exceeds its arithmetic ceiling — **BLOCKING**

Reviewing the top 1% of a population can capture at most `0.01 / prevalence` of
all positives. D3 IEEE-CIS has a declared fraud prevalence of 3.50% (Table 1),
so the ceiling is

```
0.01 / 0.035 = 0.2857
```

Manuscript Table 5(c) reports Recall@top-1% values from **0.4234** (Isolation
Forest) to **0.6234** (VR-FraudNet). *Every* value in that column exceeds the
ceiling, including all nine baselines. The values are not merely optimistic;
they are unattainable by any ranking function whatsoever.

Section 4.8.3 is explicit that the top-1% set is formed within each test fold
("The top-1% set was therefore determined independently for every temporal fold
and was never constructed by pooling scores across train, validation, and test
periods"), so a pooling artefact does not explain it either.

**Consistency check on the other datasets.** D1 (1.10%) has a ceiling of 0.909
and reports at most 0.6912 — fine. D2 (0.10%) has a ceiling above 1.0 — fine.
The problem is confined to D3.

**Required clarification.** Either (a) the D3 column is a Recall@top-k for a
different k, (b) it is a different metric altogether, or (c) the denominator was
not "all fraud cases in the fold". Until this is resolved the column must not be
published.

**Repository behaviour.** `vrfraudnet.evaluation.metrics.recall_at_top_k`
enforces the ceiling and `scripts/make_tables.py` withholds the column for D3
rather than reproducing it.

---

## A-02 — Table 6(a) mean AUPRC deltas disagree with Table 5 — **MAJOR**

Table 6(a) reports the "Mean Δ AUPRC (pp)" of VR-FraudNet against each
baseline, pooled over D1–D3. Those means are recomputable from Table 5:

| Baseline | Table 6(a) reports | Implied by Table 5 | Gap (pp) |
|---|---|---|---|
| Logistic Regression | +20.18 | +24.68 | 4.50 |
| Isolation Forest | +22.45 | +27.73 | 5.28 |
| MLP | +12.18 | +16.28 | 4.10 |
| 1D-CNN | +10.78 | +14.55 | 3.77 |
| LSTM | +9.04 | +12.38 | 3.34 |
| Random Forest | +8.32 | +10.90 | 2.58 |
| XGBoost | +5.45 | +7.31 | 1.86 |
| LightGBM | +4.94 | +6.05 | 1.11 |
| TabTransformer | +3.65 | +3.81 | 0.16 |

Every gap has the same sign — Table 6(a) understates the improvement — and the
gaps shrink monotonically as the baseline gets stronger. That pattern is
consistent with the two tables being computed over different quantities (for
example a per-seed mean of ratios versus a mean of differences, or a subset of
folds), not with a transcription slip.

**Required clarification.** State exactly which quantity Table 6(a)'s "Mean Δ
AUPRC" averages, and over which units.

---

## A-03 — DeLong tests are not defined for the reported design — **MAJOR**

A DeLong test compares two AUCs estimated on **one** labelled sample from **two**
score vectors. Section 4.8.1 reports ten seeded runs per model per dataset, so
there are ten candidate score vectors per model, and the manuscript does not say
which produced the `Z` values in Tables 6(b)–6(d). Three defensible readings
give different answers:

1. a single designated seed's predictions;
2. seed-averaged per-transaction scores, then one DeLong test;
3. ten DeLong tests, then some pooling of statistics.

Reading (3) additionally has no standard pooling rule for DeLong `Z`.

A second issue: DeLong on samples of 118,108 (D1 test), ~1.0 M (D2 test) and
118,108 (D3 test) transactions will return extremely small p-values for
essentially any consistent difference. The reported `p < 1e-15` entries are
therefore uninformative about practical significance and should be accompanied
by effect sizes or confidence intervals on ΔAUC.

**Repository behaviour.**
`vrfraudnet.statistics.delong.delong_test` requires an explicit `seed_policy`
and records it in the result file.

---

## A-04 — Pooling paired differences across datasets — **MAJOR**

Table 6(a) pools 10 seeds × 3 datasets = 30 paired observations into a single
Wilcoxon signed-rank test. The arithmetic is consistent (`V_max = 30·31/2 =
465`, as reported), but the inference is not:

* the ten seeds within a dataset are exchangeable;
* observations across datasets are not — the paired-difference distributions
  differ by dataset (D1 ≈ +2.4 pp, D2 ≈ +6.9 pp, D3 ≈ +2.2 pp), and the AUPRC
  scales differ by ~1.5×.

The pooled test answers "is the median paired improvement over this particular
three-dataset mixture positive", not "is the improvement consistent across fraud
datasets". With three datasets, a per-dataset test with n = 10 has a minimum
attainable one-sided p of 2⁻¹⁰ ≈ 9.8e-4, which is ample; there is no need to
pool.

**Repository behaviour.**
`vrfraudnet.statistics.wilcoxon.pooled_and_per_dataset` reports both.

---

## A-05 — Equation 5 (routing rule) is malformed — **MINOR**

As printed:

```
D_i = accept,   P_t < T_low
D_i = reject,   P_t < T_high
D_i = escalate, T_low ≤ P_t ≥ T_high
```

The second line duplicates the accept condition (any `P_t < T_low` also
satisfies `P_t < T_high`), and the third mixes inequality directions so that it
is satisfied by every `P_t ≥ T_high`. The surrounding prose and Figure 4 make
the intent unambiguous, and the corrected rule is implemented:

```
accept    if P_t <  T_low
escalate  if T_low ≤ P_t ≤ T_high
reject    if P_t >  T_high
```

---

## A-06 — Prompt-injection evasion rates against text-free models — **BLOCKING**

Section 4.6.2: "prompt-injection strings are introduced only into untrusted text
fields." No benchmark in this study has one:

| Dataset | Free-text field? |
|---|---|
| D1 BAF | No — 30 numeric/categorical account-application attributes |
| D2 AMLworld | No — amounts, currencies, bank/account ids, payment format |
| D3 IEEE-CIS | No — anonymised `V`/`C`/`D`/`M`/`id_*`; email domains are closed-vocabulary categoricals |
| D4 Elliptic++ | No — anonymised floats |
| D5 DGraph-Fin | No — 17 anonymised node attributes |

Separately, Logistic Regression, Isolation Forest, MLP, 1D-CNN, LSTM, Random
Forest, XGBoost, LightGBM and TabTransformer consume no text at any point, so an
injected string cannot reach them even if a text column existed.

Table 10 nevertheless reports an `ε₄ inject` evasion rate for every model at
every budget, including **0.5234** for LightGBM at B = 1 and **0.8123** at
B = 4 — the *highest* rate in that row, i.e. injection is reported as the most
effective attack against a model that cannot read text.

A related anomaly within the same column: the "A4 w/o Adversarial training"
ablation records injection rates of 0.0980 / 0.0612 / 0.0823 at B = 1/2/4 —
lower than the full model's own rates in two of three budgets, and roughly five
times lower than its own rates for every other edit family. Removing a defence
should not improve robustness.

**Required clarification.** Name the untrusted text field, and explain what an
injected string does to a gradient-boosted tree.

**Repository behaviour.**
`vrfraudnet.adversarial.applicability.check_applicable` raises
`NotApplicableError` for this family unless both preconditions hold.

---

## A-07 — Transaction splitting on D1 — **MAJOR**

BAF is a bank-account-**application** dataset: each row is an application for an
account, not a payment. Section 4.6.2 requires that "transaction-splitting edits
preserve the original total amount", which has no validity-preserving
interpretation for an application record; BAF has no transaction amount at all
(`proposed_credit_limit` and `intended_balcon_amount` are application
attributes, not transferred sums). The family is disabled for D1 pending
clarification.

---

## A-08 — D1 declared feature count contradicts its own leakage control — **MINOR**

Table 2 for D1 requires both:

* "Leakage mitigation 2: exclude protected attrs", and
* "Output to model: 30 features".

BAF Base ships 30 predictor columns before any exclusion. Excluding any
protected attribute necessarily leaves fewer than 30. The two rows cannot both
be true.

---

## A-09 — D3 declared feature count is not reconstructible — **MINOR**

From 394 transaction + 41 identity columns, removing `TransactionID`,
`isFraud` and `TransactionDT`, removing the twelve D-columns D2–D9 and D11–D14,
and replacing 339 V-columns with 50 principal components leaves roughly 130
model inputs, not the declared 109. Either additional columns were dropped
(unstated) or the count is wrong.

For contrast, D4's arithmetic is internally consistent: 183 − 89 = 94.

---

## A-10 — Cost values are absent from the main text — **MAJOR**

Section 4.6.1 normalises "the dataset-specific false-negative fraud cost and
false-positive review cost reported", and Section 5.7 claims VR-FraudNet obtains
"the lowest total loss and the highest reported savings" across D1–D3. The three
cost values appear nowhere in the main text; they are in Supplementary Table S1.
Without them the cost-sensitive weight selection, the operating-point analysis
and the entire operational claim are unverifiable.

---

## A-11 — Learning rate typeset as `2 × 10⁴` — **MINOR**

Section 4.3.1 gives the Stage 2 learning rate as "2 x10^4"; Section 4.8.1 gives
"0.0002". The former is a typesetting artefact of `2e-4`. Value used: 2e-4.

---

## A-12 — Effect size named only as "Δ" — **MINOR**

Table 6(a) reports an "Effect size Δ" without naming the estimator. Cliff's
delta is the effect size conventionally paired with a Wilcoxon signed-rank test
and is what this repository computes. Note that the reported values (1.00 for
Logistic Regression and Isolation Forest) are consistent with Cliff's delta at
complete dominance.

---

## A-13 — Meaning of a budget above 1 in Table 10 — **MINOR**

Tables 10(b) and 10(c) report a per-family column at budgets B = 2 and B = 4. It
is not stated whether "B = 2, ε₁ split" means two split edits or one split edit
plus one edit from another family. The repository implements the former (which
is what makes a per-family column meaningful) and exposes the latter.

---

## A-14 — Cross-dataset transfer has no defined alignment — **BLOCKING**

Table 8 reports a 5 × 5 zero-shot AUPRC matrix. The five benchmarks share no
feature space:

| | Features | Task |
|---|---|---|
| D1 | 30 account-application attributes | account-opening fraud |
| D2 | 9 graph attributes + entity metadata | money-laundering typology |
| D3 | ~109 anonymised transaction/identity columns | card-payment fraud |
| D4 | 94 anonymised transaction features | illicit Bitcoin flows |
| D5 | 17 anonymised node attributes | P2P borrower default |

A model trained on D1 is a function of D1's columns. Applying it to D3 is
undefined until an alignment is specified, and the manuscript describes none — no
shared encoder, no column matching, no projection, no adapter. Table 8 therefore
cannot be reproduced or even interpreted.

**Repository behaviour.** `scripts/transfer.py` requires `--alignment` and
records the choice in every result file. Calling the transfer code without one
raises `ManuscriptInconsistencyError`.

---

## A-15 — D5's label is not fraud — **MAJOR**

DGraph-Fin's node label denotes **loan default** on a P2P lending platform, not
fraud in the sense of D1–D4. Transferring a fraud detector to a default-risk
target changes the task, not just the domain, and the low D5 transfer numbers in
Table 8 are as easily explained by that as by domain shift.

---

## A-16 — D1 drift reference is ambiguous and numerically inconsistent — **MAJOR**

Table 9(a) heads its reference column "Train AUPRC" and gives **0.5247** for the
full model — exactly the value Table 5(a) reports as the **held-out test**
AUPRC on D1. Table 1 assigns months 6–7 to the D1 test partition, so the Table
5(a) figure should equal the combined month-6/month-7 performance. But the mean
of the Table 9(a) month values is

```
(0.5089 + 0.4945) / 2 = 0.5017
```

a 2.30 pp gap. The three numbers cannot all be correct. Either Table 9(a)'s
reference is a training-period figure that coincidentally equals the Table 5(a)
test figure, or the two tables disagree about what was measured. As printed, the
"drift" in Table 9(a) is measured from an unidentified baseline.

The same pattern recurs in Table 9(b), where Fold 3 (0.7456) is exactly the
Table 5(c) primary test result, and in Table 9(c), where the D4 "Train AUPRC"
(0.6823) is exactly the Table 8 in-domain diagonal.

**Repository behaviour.** `scripts/robustness.py` requires the reference
partition to be named and records it in the result file.

---

## A-17 — Escalation-path latency is below the decoding floor — **MAJOR**

Section 4.8.2 reports a p99 of **175.23 ms** on the escalation path, which
includes autoregressive generation from Meta-Llama-3.1-8B-Instruct at batch size
1 with a 384-token cap and greedy decoding, plus retrieval and verification.

Single-stream decoding is memory-bandwidth bound: each token requires reading
the full weight matrix. For 8B parameters in bfloat16 (~16 GB) against an
A100-80GB's ~2.0 TB/s HBM, the floor is roughly **8 ms per token** before any
framework overhead.

| Assumed output length | Implied ms/token | Verdict |
|---|---|---|
| 384 tokens (the stated cap) | 0.46 | ~17× faster than the bandwidth floor |
| 60 tokens (a short rationale) | 2.92 | ~2.7× faster than the floor |

Even a 20-token output would imply 8.8 ms/token, which is at the floor with zero
overhead. Reproducing 175.23 ms requires a smaller generator, a much shorter
output, speculative decoding, or an optimised serving stack (vLLM,
TensorRT-LLM) — none of which Section 4.8.2 describes; it names PyTorch 2.2.1,
Transformers 4.40.1 and PEFT 0.10.0.

**Required clarification.** State the mean and p99 generated-token count, and
the serving stack.

---

## A-18 — Ablation A6 "w/o Large LLM" is ambiguous — **MINOR**

The label suggests substituting a smaller generator, but no replacement model is
named, and the alternative reading (remove the rationale branch entirely) gives
a different experiment. Since A6 is the ablation with the *largest* reported
effect (−2.78 pp on D1, −3.52 pp on D2), its definition matters.

---

## A-19 — TabTransformer's D2 in-domain AUPRC differs across tables — **MINOR**

Table 5(b) reports 0.5137 for TabTransformer on D2. The Table 8(c) diagonal,
which is the same in-domain setting, reports 0.5187 — a 0.50 pp discrepancy. The
D1 (0.5012) and D3 (0.7234) diagonals agree with Table 5 exactly, so D2 is an
isolated inconsistency.

---

## A-20 — Figure 1 claims bounds the text disclaims — **MINOR**

Figure 1 panel (b) marks VR-FraudNet with a check for "Hallucination bound",
"Prompt-injection bound" and "Concept-drift bound", and panel (c) states "4
certified bounds · predeclared · falsifiable". Section 4 says the opposite:
"Only the split-conformal calibration component is described as having a formal
coverage bound under its stated assumptions. The remaining safeguards are
evaluated empirically." The figure should be brought into line with the text,
which is the defensible position.

---

## A-21 — Figure 2 routing proportions are operationally implausible — **MINOR**

Figure 2 shows the threshold gate sending ≈70% of transactions to ACCEPT, ≈25%
to REJECT and ≤5% to escalation. Rejecting a quarter of all traffic as fraud on
datasets whose prevalence is 0.10%–3.50% would imply a false-positive rate
between 22% and 25%, i.e. tens of thousands of declined legitimate transactions
per hundred thousand. Either the figure's proportions are illustrative rather
than measured, or "reject" means something other than "declined as fraud".

---

## Effect on this repository

| Finding | Repository response |
|---|---|
| A-01 | D3 Recall@top-1% column withheld; ceiling enforced in code |
| A-02 | Deltas recomputed from actual runs; discrepancy would resurface |
| A-03 | `--seed-policy` mandatory and recorded |
| A-04 | Pooled *and* per-dataset tests reported |
| A-05 | Corrected routing rule implemented and documented |
| A-06 | Injection family refuses to run; `NotApplicableError` |
| A-07 | Split family disabled for D1 |
| A-08, A-09 | Realised feature counts reported, never padded |
| A-10, A-27 | Cost file ships with nulls; loader refuses to guess |
| A-14 | `--alignment` mandatory; `ManuscriptInconsistencyError` otherwise |
| A-16 | Reference partition mandatory and recorded |
| A-17 | Plausibility check emitted with every latency measurement |

---

## Revision reconciliation — revised manuscript (2026-09)

The findings above were raised against the original submission. The revised
manuscript supplied with the reviewer reproducibility request changes the
status of several of them. Nothing above has been deleted or rewritten; this
section records, per finding, what the revision did. "Resolved" means the
inconsistency no longer exists in the revised text; "withdrawn" means the
affected result was removed rather than corrected; "open" means the revised
text still contains the problem.

| Finding | Status in the revised manuscript | Evidence |
|---|---|---|
| A-01 Recall@top-1% ceiling on D3 | **open** | Table 5(c) still reports values above 0.2857 for every model |
| A-02 Table 6 mean deltas | **resolved** | Table 6 now reports the values implied by Table 5 (+24.68, +27.73, +16.28, +14.55, +12.38, +10.90, +7.31, +6.05, +3.81 pp); `scripts/audit_manuscript.py --revision revised` passes this check |
| A-03 DeLong seed policy | **withdrawn** | S4.8.3: the DeLong statistics are "not retained in the revised analysis" because the underlying prediction vectors were not unambiguously preserved — which also confirms that per-transaction predictions are not available (ARTIFACT_AVAILABILITY.md) |
| A-04 pooling across datasets | **partly addressed** | the revised text repeatedly states the 30 pairs are not independent and the pooled test is a "secondary sensitivity analysis"; per-dataset tests are still not reported |
| A-05 Equation 5 malformed | **open** | the revised Eq. 5 still reads `D_i = reject, P_t < T_high` and `T_low ≤ P_t ≧ T_high` |
| A-06 injection against text-free models | **partly resolved** | the injection column is removed from the predictive-path tables (now Tables 9(a–c), four families) and S4.6.2 states why; the rationale-pathway injection evaluation (S5.7, Supplementary S4) still presupposes an untrusted text field that no benchmark has |
| A-07 splitting on D1 | **open** | unchanged |
| A-08 / A-09 feature counts | **open** | Table 2 unchanged ("30 features", "109 features") |
| A-10 cost values | **open** | still only in Supplementary S1 |
| A-11 learning-rate typesetting | **open** | S4.3.1 still prints "2 x10^4" |
| A-12 effect size unnamed | **open** | Table 6 column still headed "Effect size Δ" |
| A-13 meaning of B > 1 per family | **open** | unchanged |
| A-14 transfer without alignment | **withdrawn** | S5.4: "the previously reported full cross-dataset transfer matrices are not retained" |
| A-15 D5 label is default, not fraud | **moot** | transfer results withdrawn |
| A-16 D1 drift reference | **open** | the D1 drift table (captioned "Table 9(a)", referred to as Table 8(a)) still heads its reference column "Train AUPRC" with the value Table 5(a) reports as the held-out test AUPRC |
| A-17 escalation latency | **open** | S4.8.2 now gives hardware and timing protocol (batch 1, warm-up, 10 000 timed transactions, GPU synchronisation) but the same p99, and still no generated-token count or serving stack; the bandwidth argument is unchanged |
| A-18 A6 ambiguity | **open** | unchanged |
| A-19 TabTransformer D2 across tables | **moot** | Table 8(c) transfer diagonal withdrawn |
| A-20 Figure 1 bounds | **resolved** | Figure 1 panel (b) now states "Only the split-conformal component carries a formal coverage statement" |
| A-21 Figure 2 proportions | **open** | unchanged |

### New findings in the revised manuscript

**A-22 — Table numbering is inconsistent — MINOR.** The D1 drift table is
captioned "Table 9(a)" while the surrounding text calls it Table 8(a) and the
evasion tables are captioned 9(a–c); S4.6.2 and S5.6 refer to "Tables 10(a–c)"
for evasion; S4.8.3 refers to "Table 9(b)" for the expanding-window analysis
that is captioned 8(b). A reader cannot cite these tables unambiguously.

**A-23 — An editorial note was left in the text — MINOR.** S4.8.2 ends with
"The paper already reports these two p99 values but does not currently provide
the associated hardware and measurement protocol." — a revision note that
contradicts the paragraph it closes, which does provide them.

**A-24 — The evasion tables do not name their dataset — MAJOR.** Tables 9(a–c)
report split, delay, mule and feat columns for LightGBM, TabTransformer, two
ablations and the full model without stating which dataset(s) the attacks were
run on. Under S4.6.2's own validity conditions (splitting needs a payment
amount; mule routing needs an entity graph) the only benchmark on which both
`split` and `mule` are defined is D2 (see `docs/EDIT_GENERATORS.md`,
applicability matrix). Either the tables are D2-only and should say so, or a
dataset on which one of the two families is undefined contributed to a column,
which would make that column undefined.

**A-25 — Table 8(d) fold arithmetic is consistent — PASS.** The nominal
train/test counts (354,324 / 236,216; 413,378 / 177,162; 472,432 / 118,108;
501,959 / 88,581; 531,486 / 59,054) equal 590,540 × the stated quantiles to
rounding. This is recorded because it resolves L-03 and is implemented in
`configs/d3.yaml split.cv_cut_quantiles`.

**A-26 — S4.3.4 retrieval description is internally consistent with the
implementation — PASS.** Training-only, strictly-earlier, identifier
exclusions, four examples, omission when none is eligible: each maps to an
enforced rule and a test (`docs/RETRIEVAL_SETUP.md`).

### Effect on the audit script

`python scripts/audit_manuscript.py` keeps evaluating the original submission
by default. `python scripts/audit_manuscript.py --revision revised` evaluates
the revised manuscript: A-02 passes, A-19 is reported as not applicable, and
A-01, A-08/A-09, A-16 and A-17 still fail.
