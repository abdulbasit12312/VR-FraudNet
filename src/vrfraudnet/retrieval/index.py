"""Training-only retrieval index (manuscript Section 4.3.4).

Manuscript specification
------------------------
* The index is built per dataset from the *training partition only*.
* Each record holds a standardised transaction summary, temporal and graph
  context, the Stage 1 probability band, the structured rationale, the claims
  and the evidence references.
* Only schema-valid and verifier-accepted training examples are indexed.
* Validation, test, transfer, temporal-shock and adversarial-evaluation examples
  are excluded.
* For datasets with explicit timestamps, an indexed example is eligible only if
  it occurred strictly before the query transaction.
* Examples sharing a transaction identifier or a direct entity identifier with
  the query are excluded, to prevent identifier-based retrieval leakage.
* Summaries are encoded with ``all-MiniLM-L6-v2`` and stored in FAISS under
  cosine similarity; the four most similar eligible examples are retrieved and
  ordered by similarity. Fewer than four is allowed; retrieval is omitted
  entirely when no eligible example exists.

The leakage guarantees are enforced structurally in :class:`RetrievalIndex`, not
left to caller discipline: adding a record whose partition is not ``train``
raises immediately.

Similarity backend
------------------
The manuscript names FAISS (S4.3.4, S4.8.2: FAISS-GPU 1.8.0) without naming an
index type. This implementation computes cosine similarity as an exact inner
product over L2-normalised vectors with ``numpy``. That is mathematically
identical to a FAISS ``IndexFlatIP`` over the same normalised vectors (exact
search, no quantisation, no approximate neighbour graph), so the retrieved set
and its order are the same; only the execution engine differs. An approximate
FAISS index (IVF, HNSW) would NOT be identical and is deliberately not used,
because the manuscript does not state one (docs/KNOWN_LIMITATIONS.md L-33).
The embedding model's immutable revision is not recorded by the manuscript
(L-29); ``build_retrieval_index.py`` records whatever revision was resolved.

Serialisation
-------------
:meth:`RetrievalIndex.save` writes ``records.json`` (every field of every
record, including the rationale), ``embeddings.npy`` (float32, L2-normalised)
and ``index_manifest.json`` (encoder name, revision, dimension, record count,
SHA-256 of the two data files). :meth:`RetrievalIndex.load` refuses a directory
whose checksums do not match its manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from vrfraudnet.errors import ConfigurationError, LeakageError, MissingArtefactError

#: Manuscript S4.3.4.
ENCODER_MODEL = "all-MiniLM-L6-v2"
#: Hugging Face repository of the manuscript encoder (sentence-transformers namespace).
ENCODER_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
#: Output dimension of all-MiniLM-L6-v2 (model card); checked at build time.
ENCODER_DIMENSION = 384
N_RETRIEVED = 4
#: Similarity backend actually used (see module docstring).
SIMILARITY_BACKEND = "numpy exact inner product over L2-normalised vectors (== FAISS IndexFlatIP)"

INDEX_MANIFEST_FILE = "index_manifest.json"
INDEX_RECORDS_FILE = "records.json"
INDEX_EMBEDDINGS_FILE = "embeddings.npy"


@dataclass
class RetrievalRecord:
    """One indexed training example."""

    record_id: str
    partition: str
    timestamp: float | None
    summary: str
    probability_band: str
    rationale: dict[str, Any]
    entity_ids: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.partition != "train":
            raise LeakageError(
                f"retrieval record {self.record_id!r} belongs to partition "
                f"{self.partition!r}; manuscript S4.3.4 permits training examples only"
            )


class RetrievalIndex:
    """Cosine-similarity index over verifier-accepted training rationales."""

    def __init__(
        self,
        *,
        encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
        n_retrieved: int = N_RETRIEVED,
    ) -> None:
        self.records: list[RetrievalRecord] = []
        self.embeddings: np.ndarray | None = None
        self.n_retrieved = int(n_retrieved)
        self._encoder = encoder
        self.manifest: dict[str, Any] | None = None

    # -- construction -------------------------------------------------------
    def add(self, record: RetrievalRecord, *, verifier_accepted: bool) -> None:
        """Add one record. Only verifier-accepted, schema-valid examples qualify."""
        if not verifier_accepted:
            return
        self.records.append(record)
        self.embeddings = None  # invalidate

    def build(self) -> "RetrievalIndex":
        """Encode every summary and normalise for cosine similarity."""
        if not self.records:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return self
        encoder = self._encoder or default_encoder()
        matrix = np.asarray(encoder([r.summary for r in self.records]), dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.embeddings = matrix / np.maximum(norms, 1e-12)
        return self

    # -- query --------------------------------------------------------------
    def query(
        self,
        summary: str,
        *,
        query_timestamp: float | None = None,
        query_transaction_id: str | None = None,
        query_entity_ids: frozenset[str] | None = None,
    ) -> list[RetrievalRecord]:
        """Return up to ``n_retrieved`` eligible examples, ordered by similarity.

        Eligibility (manuscript S4.3.4):
        1. training partition only (guaranteed at insertion time);
        2. strictly earlier timestamp than the query, when timestamps exist;
        3. no shared transaction identifier;
        4. no shared direct entity identifier.
        """
        if self.embeddings is None:
            self.build()
        if not self.records:
            return []  # "Retrieval is omitted when no eligible training example exists."

        eligible = [
            i
            for i, record in enumerate(self.records)
            if _is_eligible(
                record, query_timestamp, query_transaction_id, query_entity_ids or frozenset()
            )
        ]
        if not eligible:
            return []

        encoder = self._encoder or default_encoder()
        vector = np.asarray(encoder([summary]), dtype=np.float32)[0]
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        similarities = self.embeddings[eligible] @ vector
        order = np.argsort(-similarities, kind="stable")[: self.n_retrieved]
        return [self.records[eligible[int(i)]] for i in order]

    def __len__(self) -> int:
        return len(self.records)

    # -- persistence --------------------------------------------------------
    def save(self, directory: str | Path, *, manifest_extra: dict[str, Any] | None = None) -> Path:
        """Write records, embeddings and a checksummed manifest to ``directory``."""
        if self.embeddings is None:
            self.build()
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        records_path = directory / INDEX_RECORDS_FILE
        embeddings_path = directory / INDEX_EMBEDDINGS_FILE
        payload = [
            {**asdict(r), "entity_ids": sorted(r.entity_ids)} for r in self.records
        ]
        records_path.write_text(
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=True), encoding="utf-8"
        )
        np.save(embeddings_path, np.asarray(self.embeddings, dtype=np.float32))
        manifest = {
            "schema_version": "vrfraudnet-retrieval-index/1",
            "n_records": len(self.records),
            "n_retrieved": self.n_retrieved,
            "embedding_dimension": int(self.embeddings.shape[1]) if len(self.records) else 0,
            "normalisation": "l2",
            "similarity": "cosine",
            "backend": SIMILARITY_BACKEND,
            "partition": "train",
            "files": {
                INDEX_RECORDS_FILE: _sha256(records_path),
                INDEX_EMBEDDINGS_FILE: _sha256(embeddings_path),
            },
            **(manifest_extra or {}),
        }
        (directory / INDEX_MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        encoder: Callable[[Sequence[str]], np.ndarray] | None = None,
    ) -> "RetrievalIndex":
        """Load an index written by :meth:`save`, verifying its checksums.

        ``encoder`` is needed only to embed *queries*; it must be the same model
        the index was built with, which the manifest names.
        """
        directory = Path(directory)
        manifest_path = directory / INDEX_MANIFEST_FILE
        if not manifest_path.exists():
            raise MissingArtefactError(
                f"no retrieval index at {directory} ({INDEX_MANIFEST_FILE} missing). Build one with:\n"
                "    python scripts/build_retrieval_index.py --dataset D1 --config configs/d1.yaml "
                "--corpus <corpus.jsonl> --seed 42"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in manifest.get("files", {}).items():
            actual = _sha256(directory / name)
            if actual != expected:
                raise ConfigurationError(
                    f"retrieval index file {name} has SHA-256 {actual}, manifest says {expected}; "
                    "refusing to load a modified index"
                )
        records_raw = json.loads((directory / INDEX_RECORDS_FILE).read_text(encoding="utf-8"))
        index = cls(encoder=encoder, n_retrieved=int(manifest.get("n_retrieved", N_RETRIEVED)))
        for raw in records_raw:
            raw = dict(raw)
            raw["entity_ids"] = frozenset(raw.get("entity_ids", []))
            index.records.append(RetrievalRecord(**raw))
        index.embeddings = np.load(directory / INDEX_EMBEDDINGS_FILE)
        if index.embeddings.shape[0] != len(index.records):
            raise ConfigurationError("embeddings and records disagree on the record count")
        index.manifest = manifest
        return index


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_eligible(
    record: RetrievalRecord,
    query_timestamp: float | None,
    query_transaction_id: str | None,
    query_entity_ids: frozenset[str],
) -> bool:
    if query_transaction_id is not None and record.record_id == query_transaction_id:
        return False
    if query_entity_ids and record.entity_ids & query_entity_ids:
        return False
    if query_timestamp is not None and record.timestamp is not None:
        if record.timestamp >= query_timestamp:
            return False
    return True


def default_encoder(revision: str | None = None) -> Callable[[Sequence[str]], np.ndarray]:
    """Return the ``all-MiniLM-L6-v2`` sentence encoder (manuscript S4.3.4).

    ``revision`` pins the Hugging Face commit of the encoder. The manuscript
    does not record one (L-29); when ``None`` the hub's current default is
    resolved and the built index records ``"unpinned"``.

    Raises when ``sentence-transformers`` is unavailable rather than silently
    substituting a different embedding, because a different encoder would change
    which demonstrations the language model sees.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise MissingArtefactError(
            f"the manuscript's retrieval encoder ({ENCODER_MODEL}) requires "
            "sentence-transformers. Install it with:\n"
            "    pip install -r requirements-optional.txt\n"
            "No substitute encoder is used, because a different embedding would "
            "change the retrieved demonstrations."
        ) from exc

    model = SentenceTransformer(ENCODER_REPOSITORY, revision=revision)

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False)

    encode.model_name = ENCODER_REPOSITORY  # type: ignore[attr-defined]
    encode.revision = revision or "unpinned"  # type: ignore[attr-defined]
    encode.dimension = int(model.get_sentence_embedding_dimension())  # type: ignore[attr-defined]
    return encode


def hashing_encoder(dim: int = 256) -> Callable[[Sequence[str]], np.ndarray]:
    """Deterministic bag-of-character-trigrams encoder for tests only.

    Never used in a manuscript reproduction path: it exists so the index's
    leakage and ordering logic can be unit-tested without downloading a model.
    """

    def encode(texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f"  {text}  "
            for i in range(len(padded) - 2):
                trigram = padded[i : i + 3]
                digest = hashlib.blake2b(trigram.encode("utf-8"), digest_size=4).digest()
                out[row, int.from_bytes(digest, "big") % dim] += 1.0
        return out

    return encode
