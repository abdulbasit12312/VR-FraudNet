# Reproducibility guide

How to set up, run and interpret this package — and, equally important, what it
cannot reproduce and why.

---

## 1. Environment

### Option A — pip

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .                     # optional; scripts also work via PYTHONPATH
```

### Option B — conda

```bash
conda env create -f environment.yml
conda activate vr-fraudnet
```

### Optional extras

```bash
pip install -r requirements-optional.txt
```

Needed only for Stage 2 (`transformers`, `peft`), the retrieval encoder
(`sentence-transformers`, `faiss`), and large-graph message passing
(`torch-geometric`). Everything else — Stage 0/1/3/4, all baselines, all
metrics, all statistics, all adversarial edits — runs without them.

### Reference environment (manuscript Section 4.8.2)

| Component | Manuscript value |
|---|---|
| CPU | AMD EPYC 7543, 32 cores |
| RAM | 256 GB |
| GPU | 1 × NVIDIA A100 80GB |
| OS | Ubuntu 22.04 LTS |
| Python | 3.11.7 |
| CUDA / cuDNN | 12.1 / 8.9 |
| PyTorch | 2.2.1 |
| PyTorch Geometric | 2.5.2 |
| LightGBM / XGBoost | 4.3.0 / 2.0.3 |
| scikit-learn | 1.4.2 |
| Transformers / PEFT | 4.40.1 / 0.10.0 |
| Sentence-Transformers | 2.7.0 |
| FAISS | 1.8.0 (GPU) |

Verify what you actually have:

```bash
python -c "import vrfraudnet, sys; from vrfraudnet.io_utils import _environment; print(_environment())"
```

The same information is written into the manifest of every result file.

---

## 2. Hardware requirements

| Task | Minimum | Comfortable |
|---|---|---|
| Tests and the synthetic demo | 4 GB RAM, any CPU | — |
| D1 preprocessing and Stage 1 | 8 GB RAM | 16 GB |
| D3 preprocessing (435 columns × 590 k rows) | 16 GB RAM | 32 GB |
| D2 graph construction (5.1 M edges) | 32 GB RAM | 64 GB |
| D5 graph (3.7 M nodes, 4.3 M edges) | 64 GB RAM | 128 GB |
| Stage 2 LoRA fine-tuning | 40 GB GPU | 80 GB GPU |
| Stage 2 inference at batch size 1 | 20 GB GPU | 40 GB GPU |

Disk: roughly 25 GB for the five raw datasets, plus room for prediction files
(one JSON per dataset × model × seed; the D2 test partition alone is ~1 M
scores).

---

## 3. Data

See `docs/DATA_AVAILABILITY.md`. Nothing is bundled; nothing is generated as a
substitute. A missing dataset raises `MissingArtefactError` naming the file and
the download location.

```bash
python scripts/prepare_data.py --dataset D1 --config configs/d1.yaml --check-only
```

---

## 4. Running the pipeline

### Sanity check (no data required)

```bash
python -m pytest -q
python examples/run_minimal_demo.py
python scripts/audit_manuscript.py
```

The demo runs on `numpy.random` output. **Nothing it prints is a scientific
result.**

### One dataset, one seed

```bash
python scripts/train.py     --dataset D1 --config configs/d1.yaml --seed 42
python scripts/evaluate.py  --dataset D1 --config configs/d1.yaml --seed 42
python scripts/calibrate.py --dataset D1 --config configs/d1.yaml --seed 42
```

### Full sweep

```bash
bash reproduce_all.sh              # prints the plan
bash reproduce_all.sh --execute    # actually runs it
```

### Statistics and tables

```bash
python scripts/stats.py --test wilcoxon
python scripts/stats.py --test delong --dataset D1 --seed-policy single_seed --seed 42
python scripts/make_tables.py --table all
```

`make_tables.py` reads only `results/`. If a run is missing it prints the exact
command that produces it and exits non-zero. It has no path that prints
manuscript values.

---

## 5. Expected runtime

Wall-clock on the reference machine, per seed. Multiply by 10 seeds and by 10
models for a complete Table 5 sweep.

| Stage | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| Preprocessing | ~2 min | ~25 min | ~12 min | ~3 min | ~30 min |
| Stage 0 filter bank | — | ~15 min | — | ~4 min | ~45 min |
| Stage 1 triage | ~4 min | ~20 min | ~10 min | ~2 min | ~25 min |
| Neural baseline (each) | ~15 min | ~60 min | ~40 min | ~8 min | ~90 min |
| Stage 2 fine-tuning | ~6–10 h on one A100 | | | | |

A complete Table 5 sweep (10 models × 10 seeds × 3 datasets) is on the order of
**several hundred GPU-hours**. Budget accordingly, and run one seed end to end
before launching a sweep.

---

## 6. Seeds and determinism

The ten seeds of Section 4.8.1 are frozen in
`vrfraudnet.seeds.MANUSCRIPT_SEEDS`. Every script warns when run with a seed
outside that set, because a mean ± standard deviation computed from other seeds
is not comparable to the paper.

`vrfraudnet.determinism.set_global_seed` pins:

* `PYTHONHASHSEED`
* `random.seed`
* `numpy.random.seed`
* `torch.manual_seed` / `torch.cuda.manual_seed_all`
* `torch.use_deterministic_algorithms(True, warn_only=True)`
* `torch.backends.cudnn.deterministic = True`, `benchmark = False`
* `CUBLAS_WORKSPACE_CONFIG=:4096:8`

Independent random streams are derived from the single run seed with
`derive_subseed(seed, *tags)`, a SHA-256 based derivation that is stable across
processes and platforms (unlike Python's salted `hash`).

### Limits to exact reproducibility

1. **GPU reductions.** Floating-point reduction order differs across GPU
   architectures, CUDA versions and thread counts. Bitwise identity across
   different hardware is not attainable, with or without seeding.
2. **LightGBM thread count.** Determinism holds for a *fixed* `num_threads`.
   The manuscript does not report one; this repository defaults to 4 (L-21).
   Changing it changes histogram construction order and therefore the model.
3. **BLAS backend.** MKL, OpenBLAS and Accelerate produce slightly different
   results for the same linear algebra.
4. **Library versions.** scikit-learn's `IsotonicRegression` tie-breaking and
   LightGBM's split selection have both changed between minor versions. Pin the
   versions in `requirements.txt`.
5. **Dataset revisions.** Kaggle datasets are occasionally re-uploaded. Record
   SHA-256 checksums (see `docs/DATA_AVAILABILITY.md`).

Practical expectation: on identical hardware and library versions, per-seed
AUPRC should reproduce to roughly 1e-6. Across different hardware, expect
agreement to about 1e-3, which is well inside the seed-to-seed standard
deviations the manuscript reports (0.0016–0.0108).

---

## 7. What this package cannot reproduce

Read `docs/KNOWN_LIMITATIONS.md` and `docs/MANUSCRIPT_AUDIT.md` in full before
claiming any reproduction. The headline items:

| Artefact | Obstacle |
|---|---|
| Table 5 baselines | No published baseline hyperparameters (L-18) |
| Table 5(c) Recall@top-1% | Values exceed the arithmetic ceiling (A-01) |
| Tables 6(b–d) | DeLong seed policy unspecified (A-03) |
| Table 7 ablations | Stage 2 training corpus undescribed (L-25) |
| Table 8 transfer | No feature-space alignment specified (A-14) |
| Table 10 injection column | No untrusted text field exists (A-06) |
| Supplementary S1 | Cost values absent from the main text (A-10) |
| Escalation latency | Reported p99 is below the decoding floor (A-17) |

---

## 8. Result-file contract

Every result file carries a manifest:

```json
{
  "manifest": {
    "schema_version": "vrfraudnet-result/1",
    "experiment": "train",
    "dataset": "D1",
    "model": "vr_fraudnet",
    "seed": 42,
    "config_path": "configs/d1.yaml",
    "config_sha256": "...",
    "partition": "validation+test",
    "git_revision": "abc1234",
    "created_utc": "2026-08-25T12:00:00+00:00",
    "environment": { "python": "3.11.7", "lightgbm": "4.3.0", "...": "..." }
  },
  "results": { "...": "..." }
}
```

Manifests deliberately exclude username, hostname and absolute paths: result
files are meant to be shared, and none of those fields is scientifically
relevant. `tests/test_config_and_registry.py` enforces that.

Existing result files are never overwritten silently; pass `--overwrite`
deliberately.

---

## 9. Reporting a reproduction

When reporting numbers obtained with this package, state:

1. the commit hash (`git rev-parse --short HEAD`);
2. the seeds used, and whether they are the manuscript's ten;
3. every ASSUMPTION-tagged setting that differed from the shipped defaults —
   `summarise_assumptions` prints them at the top of every run;
4. the dataset checksums;
5. which audit findings from `docs/MANUSCRIPT_AUDIT.md` affect the table you
   are reporting.

Without (3) and (5), a comparison against the published numbers is not
interpretable.
