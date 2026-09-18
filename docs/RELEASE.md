# Release: reproducibility archive

How the versioned reproducibility archive requested by the reviewer is built,
verified and published, and the exact status of the current release.

## Versioning

| Field | Value |
|---|---|
| Release version | `1.0.0` (`pyproject.toml`, `CITATION.cff`) |
| Git tag | `v1.0.0-manuscript-revision` |
| Archive | `VR-FraudNet-reproducibility-1.0.0.tar.gz` |
| Release assets | the archive, `SHA256SUMS`, `REPRODUCIBILITY_MANIFEST.json` |
| Commit | the commit the tag points at: `git rev-list -n 1 v1.0.0-manuscript-revision` |

The archive's own SHA-256 is published in the GitHub Release notes and in the
`SHA256SUMS` asset. It is deliberately **not** written into any file inside
the repository, because a file that names the hash of the archive that
contains it cannot be consistent.

No LoRA adapter asset is attached: the manuscript adapter is not available
(`artifacts/lora/README.md`).

## What the archive contains

Exactly the files selected by `vrfraudnet.reproducibility.BUNDLE_INCLUDE`:
source (`src/`, `scripts/`, `tests/`, `examples/`), every configuration
(`configs/`, `schemas/`, `verifier/rules/`, `retrieval/config/`,
`adversarial/families/`, `evaluation/protocols/`, `statistics/tests/`), all
documentation (`docs/`, `artifacts/lora/README.md`, the dataset and results
READMEs), the environment files, `reproduce_all.sh`, the CI workflow, plus
`REPRODUCIBILITY_MANIFEST.json` and `SHA256SUMS` generated at packaging time.

Excluded by construction and refused by the builder if found: raw or processed
datasets, model weights and checkpoints, retrieval indexes, result and
prediction files, caches, and any file matching a secret or local-path
pattern.

## Building

```bash
python scripts/build_reproducibility_bundle.py --version 1.0.0 --tag v1.0.0-manuscript-revision
# writes dist/VR-FraudNet-reproducibility-1.0.0.tar.gz and dist/SHA256SUMS and
# dist/REPRODUCIBILITY_MANIFEST.json, prints the archive SHA-256
```

The tarball is deterministic: sorted entries, fixed modification time
(`SOURCE_DATE_EPOCH`, else the commit time), uid/gid 0, no user names, gzip
header without timestamp. Building twice from the same tree gives the same
bytes.

`REPRODUCIBILITY_MANIFEST.json` is also committed at the repository root so a
reader who clones without the release can check the tree. CI runs

```bash
python scripts/build_reproducibility_bundle.py --check-manifest
```

which rebuilds the hash table from the working tree and fails if any bundled
file changed without the manifest being regenerated
(`--write-manifest` regenerates it).

## Verifying

```bash
python scripts/verify_reproducibility_bundle.py dist/VR-FraudNet-reproducibility-1.0.0.tar.gz
```

Checks, without extracting to disk: every required file is present; every
manifest hash matches the archive bytes; every archive file is in the manifest;
every `components` hash matches; `SHA256SUMS` matches; the seed file equals the
manifest seeds and the code constant; the JSON schema and every YAML/JSON file
parse; no data, weight, cache or credential file is present; no secret pattern
appears in any text file. Alternatively, unpack and run `sha256sum -c
SHA256SUMS`.

## Publishing

```bash
git tag -a v1.0.0-manuscript-revision -m "Reproducibility release for the revised manuscript"
git push origin v1.0.0-manuscript-revision
gh release create v1.0.0-manuscript-revision \
  dist/VR-FraudNet-reproducibility-1.0.0.tar.gz dist/SHA256SUMS dist/REPRODUCIBILITY_MANIFEST.json \
  --title "v1.0.0-manuscript-revision" --notes-file docs/RELEASE_NOTES_v1.0.0-manuscript-revision.md
```

Authentication is through `gh auth login` or a credential helper configured
outside the repository. No token is ever written into this repository.

## Status of the current release

See the "Release status" section at the end of
`docs/RELEASE_NOTES_v1.0.0-manuscript-revision.md`, which is updated when the release is actually
published. Until it says *published*, the release has been **prepared, not
published**, and the archive should be built locally from the tag with the
command above.

## Optional: DOI via Zenodo

A fixed GitHub Release already satisfies the request for an immutable
archive. For a citable DOI, the repository owner can enable the GitHub–Zenodo
integration (https://zenodo.org/account/settings/github/) and re-publish the
release; Zenodo then mints a DOI for that exact tag. This requires the owner's
Zenodo login and is not something the build scripts can do.
