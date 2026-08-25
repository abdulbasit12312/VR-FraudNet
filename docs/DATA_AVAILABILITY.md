# Data availability

**No dataset is redistributed in this repository.** Every benchmark must be
obtained from its original provider under that provider's licence and terms of
use. `.gitignore` blocks `datasets/raw/`, `datasets/interim/` and
`datasets/processed/` so that data cannot be committed by accident.

Expected layout after download:

```
datasets/
  raw/
    D1/  Base.csv
    D2/  HI-Small_Trans.csv
    D3/  train_transaction.csv  train_identity.csv
    D4/  txs_features.csv  txs_classes.csv  txs_edgelist.csv
    D5/  dgraphfin.npz
  checksums/
    *.sha256          # optional, recorded by you after download
```

Verify a prepared dataset against the characteristics manuscript Table 1
declares:

```bash
python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml --check-only
```

The script warns loudly when the realised row count, feature count or fraud
prevalence disagrees with Table 1.

---

## D1 — BAF: Bank Account Fraud (Base variant)

| | |
|---|---|
| Official source | <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022> |
| Reference | Jesus et al., *Turning the Tables: Biased, Imbalanced, Dynamic Tabular Datasets for ML Evaluation*, NeurIPS 2022 Datasets and Benchmarks |
| Licence | CC BY-NC-SA 4.0 — **non-commercial** |
| Access | Kaggle account required |
| Files needed | `Base.csv` (the Base variant only; the five bias-injected variants are not used) |
| Declared in Table 1 | 1,000,000 rows · 30 features · 1.10% fraud · 8 months |

Note: BAF is a **synthetic** dataset generated from a real account-opening fraud
dataset with differential-privacy guarantees. It is a bank-account-*application*
benchmark, which is why the transaction-splitting edit family is undefined for
it (audit finding A-07).

---

## D2 — AMLworld HI-Small (IBM synthetic AML transactions)

| | |
|---|---|
| Official source | <https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml> |
| Reference | Altman et al., *Realistic Synthetic Financial Transactions for Anti-Money Laundering Models*, NeurIPS 2023 Datasets and Benchmarks |
| Licence | CDLA-Sharing-1.0 |
| Files needed | `HI-Small_Trans.csv` (and `HI-Small_Patterns.txt` if you wish to confirm that the pattern-label oracle has been excluded) |
| Declared in Table 1 | 515,088 accounts · 5,078,345 transactions · 0.10% laundering · 10 days |

**Leakage note.** The accompanying pattern files identify the generator's
laundering typologies directly. Manuscript Table 2 drops them ("drop
pattern-label oracle") and `data/amlworld.py` enforces that.

---

## D3 — IEEE-CIS Fraud Detection (Vesta)

| | |
|---|---|
| Official source | <https://www.kaggle.com/competitions/ieee-fraud-detection/data> |
| Licence | Kaggle competition rules; **competition-use terms apply**, review before redistribution or commercial use |
| Access | Kaggle account plus acceptance of the competition rules |
| Files needed | `train_transaction.csv`, `train_identity.csv` (joined on `TransactionID`) |
| Declared in Table 1 | 590,540 transactions · 394 + 41 columns · 3.50% fraud · ~6 months |

**Leakage note.** The community "magic UID" feature (derived from
card/address/`D1` combinations) is a known leakage shortcut on this dataset.
Manuscript Table 2 disables it; `data/ieee_cis.py` constructs no such feature.

**Metric note.** At 3.50% prevalence, Recall@top-1% is bounded above by 0.2857.
See audit finding A-01.

---

## D4 — Elliptic++ transaction graph

| | |
|---|---|
| Official source | <https://github.com/git-disl/EllipticPlusPlus> |
| Underlying dataset | Elliptic Data Set, <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set> |
| Reference | Elmougy and Liu, *Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network*, KDD 2023 |
| Licence | See the repository; the original Elliptic data is CC BY-NC-SA 4.0 |
| Files needed | `txs_features.csv`, `txs_classes.csv`, `txs_edgelist.csv` |
| Declared in Table 1 | 203,769 transaction nodes · 234,355 edges · 183 features · 2.23% illicit · 49 timesteps |

**Preprocessing note.** The manuscript drops 89 pre-aggregated neighbourhood
features, leaving 94 local features — the one dataset whose declared feature
arithmetic is internally consistent.

---

## D5 — DGraph-Fin (Finvolution)

| | |
|---|---|
| Official source | <https://dgraph.xinye.com/dataset> |
| Reference | Huang et al., *DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection*, NeurIPS 2022 Datasets and Benchmarks |
| Licence | Research use only; **an application and agreement are required** |
| Access | Registration and approval by the dataset host |
| Files needed | `dgraphfin.npz` |
| Declared in Table 1 | 3,700,550 nodes · 4,300,999 directed edges · 17 node features · 11 edge types · 1.30% positive |

**Label-semantics note.** The DGraph-Fin node label denotes **loan default**, not
fraud in the sense of D1–D4. See audit finding A-15 before interpreting any
transfer result involving D5.

---

## Recording checksums

After downloading, record a fingerprint so a later run can prove it used the
same bytes:

```bash
sha256sum datasets/raw/D1/Base.csv > datasets/checksums/D1_Base.csv.sha256
```

`datasets/checksums/*.sha256` is the one path under `datasets/` that is *not*
ignored by git, so the fingerprints can be committed while the data cannot.

---

## Licence compatibility

The MIT licence in this repository covers **source code only**. Several of the
datasets above are non-commercial (CC BY-NC-SA) or require an individual
agreement. Obtain and use them under their own terms; nothing here grants any
right to them.
