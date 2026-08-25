# results/

Every artefact this repository produces lands here. **Nothing in this directory
is committed** (see `.gitignore`) — results belong to a run, not to the source
tree, and committing them would make it impossible to tell a reproduction from a
recollection.

```
results/
  prepared/<dataset>/report.json            split sizes, leakage controls, feature counts
  predictions/<dataset>/<model>_seed<N>.json validation + test scores, frozen threshold
  metrics/<dataset>/<model>_seed<N>.json     AUPRC, ROC-AUC, F1, MCC, Recall@top-1%
  calibration/<dataset>/...                  split-conformal thresholds and coverage
  robustness/<dataset>/...                   drift, shock and budgeted-evasion results
  transfer/<source>_to_<target>_<align>.json cross-dataset transfer cells
  statistics/wilcoxon_auprc.json             paired tests, Holm-adjusted
  statistics/delong_<dataset>_<policy>.json  DeLong comparisons
  latency/<dataset>/<pathway>_seed<N>.json   per-transaction latency
  tables/table_<n>.md                        regenerated manuscript tables
```

Every file carries a manifest recording the experiment, dataset, model, seed,
config path and SHA-256, git revision, timestamp and library versions. Manifests
deliberately exclude username, hostname and absolute paths.

Existing files are never overwritten silently; pass `--overwrite` deliberately.
