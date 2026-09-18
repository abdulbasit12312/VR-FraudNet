"""Build the training-only retrieval index for one dataset (manuscript S4.3.4).

Usage
-----
    python scripts/build_retrieval_index.py \\
        --dataset D1 --config configs/d1.yaml \\
        --corpus <rationale_corpus_train.jsonl> --seed 42

    # Interface check without downloading the encoder (CI only; the resulting
    # index is NOT the manuscript's and says so in its manifest):
    python scripts/build_retrieval_index.py ... --encoder test-hashing

What is indexed
---------------
Exactly what the manuscript permits (S4.3.4): records from the TRAIN partition
of the named dataset whose target rationale is schema-valid and accepted by the
deterministic verifier, restricted to the review band defined by the Stage 1
thresholds. A record from any other partition raises ``LeakageError`` before
anything is written. Validation, test, temporal-shock, strict-inductive and
adversarial-evaluation examples can therefore never enter the index.

Where it goes
-------------
``retrieval/index/<dataset>/`` (git-ignored): ``records.json``,
``embeddings.npy`` and ``index_manifest.json`` with SHA-256 of both data
files, the encoder repository, its resolved revision (``"unpinned"`` when the
operator did not pin one, L-29), the embedding dimension, the corpus checksum
and the retrieval configuration checksum.

Query-time eligibility (strictly earlier timestamp, no shared transaction or
entity identifier) is enforced by ``RetrievalIndex.query`` at run time, not at
build time, because it depends on the query.

Exit codes: 0 success, 2 missing input, 3 leakage.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from _common import REPO_ROOT, add_common_arguments, bootstrap

from vrfraudnet.errors import ConfigurationError, LeakageError, MissingArtefactError
from vrfraudnet.io_utils import sha256_of_file
from vrfraudnet.retrieval.index import (
    ENCODER_DIMENSION,
    ENCODER_REPOSITORY,
    RetrievalIndex,
    RetrievalRecord,
    default_encoder,
    hashing_encoder,
)
from vrfraudnet.verifier import DeterministicVerifier
from vrfraudnet.verifier.evidence import TransactionEvidence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_stage2_lora import load_corpus  # noqa: E402

log = logging.getLogger("build_retrieval_index")

RETRIEVAL_CONFIG = REPO_ROOT / "retrieval" / "config" / "retrieval.yaml"


def probability_band(p: float, t_low: float, t_high: float) -> str:
    """Stage 1 probability band label stored with each record (S4.3.4)."""
    if p < t_low:
        return "low"
    if p > t_high:
        return "high"
    return "review"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_arguments(parser)
    parser.add_argument("--corpus", type=Path, required=True,
                        help="JSONL rationale corpus, TRAIN partition only (docs/STAGE2_LORA.md)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="default retrieval/index/<dataset>")
    parser.add_argument("--encoder", choices=("manuscript", "test-hashing"), default="manuscript",
                        help="'manuscript' = sentence-transformers/all-MiniLM-L6-v2 (S4.3.4); "
                             "'test-hashing' = deterministic character-trigram encoder for "
                             "interface tests ONLY, never for a reproduction")
    parser.add_argument("--encoder-revision", default=None,
                        help="immutable Hugging Face revision of the encoder (L-29); recorded")
    parser.add_argument("--include-outside-review-band", action="store_true",
                        help="index verifier-accepted TRAIN records regardless of their Stage 1 "
                             "band (the manuscript indexes review-band examples; this flag is "
                             "for diagnostics and is recorded in the manifest)")
    args = parser.parse_args()

    config, _ = bootstrap(args)
    t_low = float(config.require("stage1.threshold_low"))
    t_high = float(config.require("stage1.threshold_high"))
    verifier = DeterministicVerifier()

    records = load_corpus(args.corpus, dataset=args.dataset, partition="train")
    if args.encoder == "manuscript":
        encoder = default_encoder(args.encoder_revision)
        encoder_name = getattr(encoder, "model_name", ENCODER_REPOSITORY)
        encoder_revision = getattr(encoder, "revision", args.encoder_revision or "unpinned")
        dimension = int(getattr(encoder, "dimension", ENCODER_DIMENSION))
        if dimension != ENCODER_DIMENSION:
            raise ConfigurationError(
                f"encoder reports dimension {dimension}; all-MiniLM-L6-v2 has {ENCODER_DIMENSION}"
            )
    else:
        encoder = hashing_encoder()
        encoder_name = "TEST-ONLY character-trigram hashing encoder (NOT the manuscript encoder)"
        encoder_revision = "n/a"
        dimension = 256
        log.warning("--encoder test-hashing: this index is an interface check, not a reproduction")

    index = RetrievalIndex(encoder=encoder)
    counts = {"loaded": len(records), "indexed": 0, "verifier_rejected": 0,
              "outside_review_band": 0}
    for record in records:
        evidence = TransactionEvidence.from_json(record.evidence)
        accepted = verifier.verify(record.target, evidence).accepted
        if not accepted:
            counts["verifier_rejected"] += 1
            continue
        band = probability_band(record.triage_probability, t_low, t_high)
        if band != "review" and not args.include_outside_review_band:
            counts["outside_review_band"] += 1
            continue
        index.add(
            RetrievalRecord(
                record_id=record.record_id,
                partition=record.partition,           # raises LeakageError unless "train"
                timestamp=record.timestamp,
                summary=record.evidence_summary,
                probability_band=band,
                rationale=record.target,
                entity_ids=frozenset(record.entity_ids),
            ),
            verifier_accepted=True,
        )
        counts["indexed"] += 1
    index.build()

    output_dir = args.output_dir or (REPO_ROOT / "retrieval" / "index" / args.dataset)
    index.save(
        output_dir,
        manifest_extra={
            "dataset": args.dataset,
            "seed": args.seed,
            "encoder": encoder_name,
            "encoder_revision": encoder_revision,
            "encoder_dimension_expected": dimension,
            "reproduction_grade": args.encoder == "manuscript",
            "corpus_file": args.corpus.name,
            "corpus_sha256": sha256_of_file(args.corpus),
            "retrieval_config_sha256": sha256_of_file(RETRIEVAL_CONFIG),
            "review_band": {"t_low": t_low, "t_high": t_high, "source": "manuscript S4.8.1"},
            "include_outside_review_band": bool(args.include_outside_review_band),
            "counts": counts,
            "eligibility_at_query_time": [
                "strictly earlier timestamp than the query",
                "no shared transaction identifier",
                "no shared direct entity identifier",
            ],
        },
    )
    manifest = json.loads((output_dir / "index_manifest.json").read_text(encoding="utf-8"))
    log.info("index written to %s: %s", output_dir, json.dumps(counts))
    print(json.dumps({k: manifest[k] for k in ("n_records", "embedding_dimension", "encoder",
                                                "encoder_revision", "files")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (MissingArtefactError, ConfigurationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    except LeakageError as exc:
        print(f"leakage: {exc}", file=sys.stderr)
        raise SystemExit(3)
