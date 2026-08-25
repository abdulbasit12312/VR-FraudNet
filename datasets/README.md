# datasets/

**No data is stored here and none is committed.** `.gitignore` blocks
`raw/`, `interim/`, `processed/` and every common data extension.

Expected layout after you download the benchmarks yourself
(see `docs/DATA_AVAILABILITY.md`):

```
datasets/raw/D1/Base.csv
datasets/raw/D2/HI-Small_Trans.csv
datasets/raw/D3/train_transaction.csv
datasets/raw/D3/train_identity.csv
datasets/raw/D4/txs_features.csv
datasets/raw/D4/txs_classes.csv
datasets/raw/D4/txs_edgelist.csv
datasets/raw/D5/dgraphfin.npz
```

`checksums/` is the one subdirectory that IS tracked. Record a fingerprint for
each file you download so a later run can prove it used the same bytes:

```bash
sha256sum datasets/raw/D1/Base.csv > datasets/checksums/D1_Base.csv.sha256
```
